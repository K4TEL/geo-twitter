import pandas as pd
import numpy as np
from geopy import distance
from shapely.geometry import Point, LineString
import geopy.distance
import geopandas as gpd
from geopandas import GeoDataFrame
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, multivariate_normal
import scipy.sparse as sparse
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from scipy.linalg import cholesky
import imageio
import os
import shutil
import moviepy.editor as mp
from moviepy.editor import *

from utils.regressor import *
from utils.benchmarks import *
from utils.result_manager import *


def GaussianModel(means, sigma):
    return dist.MultivariateNormal(torch.from_numpy(means), torch.from_numpy(sigma))


def GaussianWeights(weights):
    return dist.Categorical(torch.from_numpy(weights))


def get_gm_family(means, sigma, weights):
    gaussian = GaussianModel(means, sigma)
    gmm_weights = GaussianWeights(weights)
    return dist.MixtureSameFamily(gmm_weights, gaussian)


class ResultVisuals():
    def __init__(self, manager):
        self.manager = manager
        self.cluster = self.manager.cluster
        self.feature = self.manager.feature
        self.prefix = self.manager.prefix

        self.df = self.manager.df
        self.pred_columns = self.manager.pred_columns
        self.size = self.manager.size
        self.true = self.manager.true

        self.prob = self.manager.prob
        self.outcomes = self.manager.outcomes
        self.cov = self.manager.cov
        self.weighted = self.manager.weighted

        self.dist = self.manager.dist

        self.covariances = self.manager.covariances

        self.palette = {1: 'darkgreen',
                        2: 'goldenrod',
                        3: 'darkorange',
                        4: 'crimson',
                        5: 'darkred'}

        self.zorder = {'XS': 5,
                       'S': 4,
                       'M': 3,
                       'L': 2,
                       'XL': 1}

        self.world_model = "naturalearth_lowres"
        self.city_model = "naturalearth_cities"

    def prob_map_animation(self, frames=42, gif=False, clean=True, video=True):
        frames = self.size if frames > self.size else frames
        xmin, xmax = -180, 180
        ymin, ymax = -90, 90
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        XX = np.array([xx.ravel(), yy.ravel()]).T

        map_frames_files = []
        frame_dir = f"results/img/gaussian_{self.prefix}_N{frames}_{datetime.today().strftime('%Y-%m-%d')}"
        Path(f"./{frame_dir}").mkdir(parents=True, exist_ok=True)

        for i in range(frames):
            loss_lh = self.df.loc[i, f'lh_loss']

            fig, ax = plt.subplots(figsize=(20, 20))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            if self.outcomes > 1:
                gmm = get_gm_family(self.manager.means[i, :].reshape(self.outcomes, 2),
                                    self.manager.covs[i, :].reshape(self.outcomes, 2, 2),
                                    self.manager.weights[i, :].reshape(-1))
                Z = gmm.log_prob(torch.from_numpy(XX)).numpy()
                gmm_lh = gmm.log_prob(torch.from_numpy(self.true[i, :])).numpy()
            else:
                gm = GaussianModel(self.manager.means[i, :].reshape(2),
                                    self.manager.covs[i, :].reshape(2, 2))
                Z = gm.log_prob(torch.from_numpy(XX)).numpy()
                gmm_lh = gm.log_prob(torch.from_numpy(self.true[i, :])).numpy()

            zmin = np.min(Z)
            zmax = np.max(Z)
            print(f"{i}\tMin: {zmin}\tMax: {zmax}")
            Z = Z.reshape(xx.shape)

            #contour = ax.contour(xx, yy, Z, levels=np.linspace(zmin, zmax, 500), cmap='RdYlGn_r', linewidths=0.5)
            contour = ax.contourf(xx, yy, Z, levels=np.linspace(zmin, zmax, 250), cmap='Spectral_r', alpha=0.7)
            fig.colorbar(contour, orientation="horizontal", pad=0.2)

            map = Basemap(ax=ax)
            map.drawcoastlines(linewidth=0.5, color="gray")
            map.drawcountries(linewidth=0.7, color="gray")
            map.drawparallels(np.arange(ymin, ymax, 30.))
            map.drawmeridians(np.arange(xmin, xmax, 30.))

            if self.outcomes > 1:
                map.scatter(self.manager.means[i, :, 0], self.manager.means[i, :, 1], s=1, c="black", zorder=10)
            else:
                map.scatter(self.manager.means[i, 0], self.manager.means[i, 1], s=1, c="black", zorder=10)

            map.scatter(self.true[i, 0], self.true[i, 1], color="white", s=10, zorder=11)
            plt.axvline(x=self.true[i, 0], color="white", linestyle='--', lw=3)
            plt.axhline(y=self.true[i, 1], color="white", linestyle='--', lw=3)
            fig.text(0.5, 0.04, f"{self.true[i, :]} --- LLH: {loss_lh} --- GMM: {gmm_lh}", ha='center', va='center')

            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            ax.set_yticks(range(-90, 90, 30))
            ax.set_xticks(range(-180, 180, 30))
            plt.title(f'{self.prefix} - Log-likelihood contour of Gaussian Model with {self.outcomes} means - frame {i}')

            if gif or video:
                filename = f"{frame_dir}/plot-{i}.png"
                map_frames_files.append(filename)
                plt.savefig(filename, dpi=100)
                print(f"VAL\tSAVE\t{i} - Gaussian model of {self.size} samples is drawn to file: {filename}")

        if gif:
            print(f"VAL\tCreating gif animation for {frames} gaussian plots")
            gif_filename = f"results/gif/gaussian_maps_{self.prefix}_N{frames}_{datetime.today().strftime('%Y-%m-%d')}.gif"
            with imageio.get_writer(gif_filename, mode="I") as writer:
                for f in map_frames_files:
                    image = imageio.imread(f)
                    writer.append_data(image)
            print(f"VAL\tGIF animation for {frames} gaussian plots saved to {gif_filename}")

        if video:
            mp4_filename = f"results/mp4/gaussian_maps_{self.prefix}_N{frames}_{datetime.today().strftime('%Y-%m-%d')}.mp4"
            clip = ImageSequenceClip(map_frames_files, fps=4)
            clip.write_videofile(mp4_filename, fps=24)

        if clean:
            print(f"VAL\tRemoving {frames} gaussian plot frames")
            shutil.rmtree(f"./{frame_dir}", ignore_errors=True)

    def gaus_map(self, index=42, save=True):
        xmin, xmax = -180, 180
        ymin, ymax = -90, 90
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        XX = np.array([xx.ravel(), yy.ravel()]).T

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        #print(means, covs, true)

        if self.df.size > 0:
            loss_lh = self.df.loc[index, f'lh_loss']

        if self.outcomes > 1:
            gmm = get_gm_family(self.manager.means[index, :].reshape(self.outcomes, 2),
                                self.manager.covs[index, :].reshape(self.outcomes, 2, 2),
                                self.manager.weights[index, :].reshape(-1))
            Z = gmm.log_prob(torch.from_numpy(XX)).numpy()
            if self.true.size > 0:
                gmm_lh = gmm.log_prob(torch.from_numpy(self.true[index, :])).numpy()
        else:
            gm = GaussianModel(self.manager.means[index, :].reshape(2),
                                self.manager.covs[index, :].reshape(2, 2))
            Z = gm.log_prob(torch.from_numpy(XX)).numpy()
            if self.true.size > 0:
                gmm_lh = gm.log_prob(torch.from_numpy(self.true[index, :])).numpy()

        zmin = np.min(Z)
        zmax = np.max(Z)
        print(f"Log-likelihood:\tMin: {zmin}\tMax: {zmax}")
        Z = Z.reshape(xx.shape)

        #contour = ax.contour(xx, yy, Z, levels=np.linspace(zmin, zmax, 500), cmap='RdYlGn_r', linewidths=0.5)
        contour = ax.contourf(xx, yy, Z, levels=np.linspace(zmin, zmax, 250), cmap='Spectral_r', alpha=0.7)
        fig.colorbar(contour, orientation="horizontal", pad=0.2)

        map = Basemap(ax=ax)
        map.drawcoastlines(linewidth=0.5, color="gray")
        map.drawcountries(linewidth=0.7, color="gray")
        map.drawparallels(np.arange(ymin, ymax, 30.))
        map.drawmeridians(np.arange(xmin, xmax, 30.))

        if self.outcomes > 1:
            map.scatter(self.manager.means[index, :, 0], self.manager.means[index, :, 1], s=1, c="black", zorder=10)
        else:
            map.scatter(self.manager.means[index, 0], self.manager.means[index, 1], s=1, c="black", zorder=10)

        if self.true.size > 0:
            map.scatter(self.true[index, 0], self.true[index, 1], color="white", s=10, zorder=11)
            plt.axvline(x=self.true[index, 0], color="white", linestyle='--', lw=1)
            plt.axhline(y=self.true[index, 1], color="white", linestyle='--', lw=1)
            fig.text(0.5, 0.04, f"{self.true[index, :]} --- LLH: {loss_lh} --- GMM: {gmm_lh}", ha='center', va='center')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        ax.set_yticks(range(-90, 90, 30))
        ax.set_xticks(range(-180, 180, 30))
        plt.title(f'{self.prefix} - Log-likelihood contour of Gaussian Model with {self.outcomes} means - sample {index}')

        if save:
            filename = f"results/img/gaussian_sample_map_{self.prefix}_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.png"
            plt.savefig(filename, dpi=300)
            print(f"VAL\tSAVE\tGaussian model of {self.size} samples is drawn to file: {filename}")
        if not self.cluster:
            plt.show()

    def text_map_result(self, index=0, save=True):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))

        xmin, xmax = -180, 180
        ymin, ymax = -90, 90
        step_big = 45.
        ticks_big_x, ticks_big_y = range(int(xmin), int(xmax), int(step_big)), \
                                   range(int(ymin), int(ymax), int(step_big))
        tick_labels_big_x, tick_labels_big_y = [str(x) for x in ticks_big_x], \
                                               [str(y) for y in ticks_big_y]

        ind = np.argwhere(np.round(self.manager.weights[index, :] * 100, 2) > 0)
        significant = self.manager.means[index, ind].reshape(-1, 2)
        margin_lon, margin_lat = 10, 5
        min_lon, max_lon = min(significant[:, 0]) - margin_lon, max(significant[:, 0]) + margin_lon
        min_lat, max_lat = min(significant[:, 1]) - margin_lat, max(significant[:, 1]) + margin_lat
        step_zoom = 5.0
        ticks_zoom_x, ticks_zoom_y = range(int(min_lon), int(max_lon), int(step_zoom)), \
                                     range(int(min_lat), int(max_lat), int(step_zoom))
        tick_labels_zoom_x, tick_labels_zoom_y = [str(x) for x in ticks_zoom_x], \
                                                 [str(y) for y in ticks_zoom_y]

        if self.prob:
            xxb, yyb = np.mgrid[xmin:xmax:400j, ymin:ymax:400j]
            XX_big = np.array([xxb.ravel(), yyb.ravel()]).T

            xx, yy = np.mgrid[min_lon:max_lon:400j, min_lat:max_lat:400j]
            XX_zoom = np.array([xx.ravel(), yy.ravel()]).T

            if self.outcomes > 1:
                gmm = get_gm_family(self.manager.means[index, :].reshape(self.outcomes, 2),
                                    self.manager.covs[index, :].reshape(self.outcomes, 2, 2),
                                    self.manager.weights[index, :].reshape(-1))
                Z_big = gmm.log_prob(torch.from_numpy(XX_big)).numpy()
                Z_zoom = gmm.log_prob(torch.from_numpy(XX_zoom)).numpy()
            else:
                gm = GaussianModel(self.manager.means[index, :].reshape(2),
                                    self.manager.covs[index, :].reshape(2, 2))
                Z_big = gm.log_prob(torch.from_numpy(XX_big)).numpy()
                Z_zoom = gm.log_prob(torch.from_numpy(XX_zoom)).numpy()

            Z_big, Z_zoom = Z_big.reshape(xx.shape), Z_zoom.reshape(xx.shape)
            Zb_pdf, Zz_pdf = np.exp(Z_big), np.exp(Z_zoom)
            zbmin, zzmin = np.min(Zb_pdf), np.min(Zz_pdf)
            zbmax, zzmax = np.max(Zb_pdf), np.max(Zz_pdf)

            # print(f"Big PDF:\tMin: {zbmin}\tMax: {zbmax}")
            # print(f"Zoom PDF:\tMin: {zzmin}\tMax: {zzmax}")

            Zb_pdf = np.ma.array(Zb_pdf, mask=Zb_pdf < 1e-5)
            Zz_pdf = np.ma.array(Zz_pdf, mask=Zz_pdf < 1e-5)

        ax_big = ax[0]

        ax_big.set_xlim(xmin, xmax)
        ax_big.set_ylim(ymin, ymax)

        map_big = Basemap(ax=ax_big, projection='mill', resolution='l')
        map_big.drawcoastlines(linewidth=0.5, color="black", zorder=2)
        map_big.drawcountries(linewidth=0.7, color="black", zorder=3)
        map_big.drawparallels(np.arange(ymin, ymax, step_big), labels=tick_labels_big_y)
        map_big.drawmeridians(np.arange(xmin, xmax, step_big), labels=tick_labels_big_x)
        map_big.drawmapboundary(fill_color='lightgrey', zorder=0)
        map_big.fillcontinents(color='white', lake_color='lightgrey', zorder=1)

        if self.prob:
            contour_big = map_big.contourf(xxb, yyb, Zb_pdf, levels=np.linspace(zbmin, zbmax, 250),
                                           cmap='Spectral_r', alpha=0.7, zorder=9, latlon=True)
            plt.colorbar(contour_big, ax=ax_big, orientation="horizontal", pad=0.2)

        if self.outcomes > 1:
            for i in range(self.outcomes):
                if np.round(self.manager.weights[index, i] * 100, 2) > 0:
                    map_big.scatter(self.manager.means[index, i, 0],
                                    self.manager.means[index, i, 1],
                                    color=self.palette[i+1],
                                    s=10 * self.manager.weights[index, i],
                                    latlon=True,
                                    zorder=9999)
        else:
            map_big.scatter(self.manager.means[index, 0],
                            self.manager.means[index, 1],
                            latlon=True,
                            s=10, c="black", zorder=9999)

        ax_zoom = ax[1]
        ax_zoom.set_xlim(min_lon, max_lon)
        ax_zoom.set_ylim(min_lat, max_lat)

        map_zoom = Basemap(ax=ax_zoom, projection='mill',
                            llcrnrlat = min_lat,
                            llcrnrlon = min_lon,
                            urcrnrlat = max_lat,
                            urcrnrlon = max_lon, resolution='h')
        map_zoom.drawcoastlines(linewidth=0.5, color="black", zorder=2)
        map_zoom.drawcountries(linewidth=0.7, color="black", zorder=3)
        map_zoom.drawmapboundary(fill_color='lightgrey', zorder=0)
        map_zoom.drawparallels(np.arange(min_lat, max_lat, step_zoom), labels=tick_labels_zoom_y)
        map_zoom.drawmeridians(np.arange(min_lon, max_lon, step_zoom), labels=tick_labels_zoom_x)
        map_zoom.fillcontinents(color='white', lake_color='lightgrey', zorder=1)

        if self.prob:
            contour_zoom = map_zoom.contourf(xx, yy, Zz_pdf, levels=np.linspace(zzmin, zzmax, 250),
                                             cmap='Spectral_r', alpha=0.7, zorder=9, latlon=True)
            plt.colorbar(contour_zoom, ax=ax_zoom, orientation="horizontal", pad=0.2)

        if self.outcomes > 1:
            for i in range(self.outcomes):
                if np.round(self.manager.weights[index, i] * 100, 2) > 0:
                    map_zoom.scatter(self.manager.means[index, i, 0],
                                     self.manager.means[index, i, 1],
                                     latlon=True,
                                     label=f"Out {i+1}: { ', '.join(map(str, self.manager.means[index, i]))} - {round(self.manager.weights[index, i] * 100, 2)}%",
                                     color=self.palette[i+1],
                                     s=10 * self.manager.weights[index, i],
                                     zorder=9999)
        else:
            map_zoom.scatter(self.manager.means[index, 0],
                             self.manager.means[index, 1],
                             latlon=True,
                             s=10, c="black", zorder=9999)

        title = f'{self.prefix} - PDF contour of Gaussian Model with {self.outcomes} means - sample {index}'
        if self.manager.text:
            title += f"\nText: {self.manager.text}\n"
        elif self.size > 0:
            title += f"\nText: {self.manager.df.loc[self.manager.df.index[index], 'text']}\n"

        plt.legend(loc='upper center', title="Predicted outcomes", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
        plt.suptitle(title)

        if save:
            filename = f"results/img/text_map_{self.prefix}_{datetime.today().strftime('%Y-%m-%d')}.png"
            plt.savefig(filename, dpi=300)
            print(f"VAL\tSAVE\tGaussian model of {self.size} samples is drawn to file: {filename}")
        if not self.cluster:
            plt.show()

    def density(self, save=True):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set(xscale="log")
        ax.set_xlim(1e-1, 4e+4)

        for i in range(self.outcomes):
            label = f"{i+1}: d = {round(self.df[f'O{i+1}_dist'].median())} km; "
            if self.weighted:
                label += f" w = {round(self.df[f'O{i+1}_weight'].median(), 5)};"

            sns.kdeplot(self.df[f'O{i+1}_dist'], ax=ax, label=label, bw_adjust=.5,
                         color=self.palette[i+1], linewidth=2, fill=True, alpha=0.1)

            #self.df[f'O{i+1}_dist'].plot.kde(color=self.palette[i+1], label=label, ax=ax, ind=self.size)

            # peak = self.df[f'O{i+1}_dist'].median()
            # plt.axvline(peak, color=self.palette[i+1], linestyle='--', lw=1)
            # ymin, ymax = ax.get_ylim()
            # plt.text(peak, ymax, f"{round(peak, 2)}km", color=self.palette[i+1], fontweight="bold")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', title="Median per outcome", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
        plt.title(f'{self.prefix} - Density Plot for Distance error')
        plt.xlabel('Error (km)')
        plt.ylabel('Density')

        if save:
            filename = f"results/img/density_{self.prefix}_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.png"
            plt.savefig(filename, dpi=300)
            print(f"VAL\tSAVE\tDistance density of {self.size} samples is drawn to file: {filename}")
        if not self.cluster:
            plt.show()

    def cum_dist(self, best=True, threshold=200, save=True):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set(xscale="log")
        ax.set_xlim(1e-1, 4e+4)

        best_outcome_dist = "O1_dist"

        if self.outcomes == 1 or best:
            sns.ecdfplot(self.df, x=best_outcome_dist, ax=ax)

            x, y = ax.get_lines()[0].get_data()
            segments = np.array([x[:-1], y[:-1], x[1:], y[1:]]).T.reshape(-1, 2, 2)
            norm = LogNorm(self.df[best_outcome_dist].min(), self.df[best_outcome_dist].max())
            lc = LineCollection(segments, cmap='RdYlGn_r', norm=norm)
            lc.set_array(x[:-1])
            lc.set_linewidth(2)
            ax.get_lines()[0].remove()
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax)

            plt.axvline(threshold, color="red", linestyle='dashed', lw="1")
            prop = self.df[best_outcome_dist][self.df[best_outcome_dist] < threshold].count() / self.size
            plt.axhline(prop, color="black")
            plt.text(1, prop + 0.005, f"{round(prop * 100, 2)}%", color="black", fontweight="bold")

            ax.fill_between(x, prop, y, where=x >= threshold, color='red', alpha=0.1, hatch='xx')
            ax.fill_between(x, y, where=y <= prop, color='green', alpha=0.1, hatch='xx')
            ax.fill_between(x, 0, prop, where=y >= prop, color='green', alpha=0.1, hatch='xx')

        else:
            for i in range(self.outcomes):
                label = f"{i+1}: distance = {round(self.df[f'O{i+1}_dist'].mean())} km; "
                if self.weighted:
                    label += f" weight = {round(self.df[f'O{i+1}_weight'].mean(), 5)};"

                sns.ecdfplot(self.df, x=f'O{i+1}_dist', ax=ax, label=label, color=self.palette[i+1], lw=2)

                prop = self.df[f'O{i+1}_dist'][self.df[f'O{i+1}_dist'] < threshold].count() / self.size
                plt.axhline(prop, color=self.palette[i+1], lw=1)
                if prop > 0:
                    plt.text(1, prop + 0.005, f"{round(prop * 100, 2)}%", color=self.palette[i+1], fontweight="bold")

                x, y = ax.lines[i*2].get_data()
                ax.fill_between(x, y, where=y <= prop, color=self.palette[i+1], alpha=0.1, hatch='xx')
                ax.fill_between(x, 0, prop, where=y >= prop, color=self.palette[i+1], alpha=0.1, hatch='xx')
            plt.axvline(threshold, color="red", linestyle='dashed', lw="1")
            plt.text(threshold, -0.03, f"{threshold}km", color="red", fontweight="bold")

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            ax.legend(loc='upper center', title="Mean per outcome", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

        plt.title(f'{self.prefix} - Cumulative distribution for distance error with {threshold} km threshold')
        plt.xlabel('log Distance error (km)')
        plt.ylabel('Proportion')

        if save:
            filename = f"results/img/cum_dist_{self.prefix}_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.png"
            plt.savefig(filename, dpi=300)
            print(
                f"VAL\tSAVE\tError distance cumulative distribution of {self.size} samples is drawn to file: {filename}")
        if not self.cluster:
            plt.show()

    def interact_lines(self, size=500, scope="world", save=True):
        self.df = self.df.sample(n=size, random_state=42, ignore_index=True)
        self.size = len(self.df.index)

        fig = go.Figure()

        fig.add_trace(
            go.Scattergeo(
                lon=self.df["longitude"],
                lat=self.df["latitude"],
                mode='markers',
                marker=dict(
                    size=2,
                    color='rgb(0, 0, 0)',
                    line=dict(
                        width=3,
                        color='rgba(68, 68, 68, 0)'
                    )
                )
            )
        )

        for o in range(self.outcomes):
            lon, lat = [], []
            for i in range(self.size):
                lon.append(np.array(self.df.loc[i, f"O{o+1}_point"])[0])
                lat.append(np.array(self.df.loc[i, f"O{o+1}_point"])[1])

            fig.add_trace(
                go.Scattergeo(
                    lon=lon,
                    lat=lat,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color='rgb(0, 0, 0)',
                        line=dict(
                            width=3,
                            color='rgba(68, 68, 68, 0)'
                        )
                    )
                )
            )

            for i in range(self.size):
                fig.add_trace(
                    go.Scattergeo(
                        lon=[self.df["longitude"][i], np.array(self.df.loc[i, f"O{o+1}_point"])[0]],
                        lat=[self.df["latitude"][i], np.array(self.df.loc[i, f"O{o+1}_point"])[1]],
                        mode='lines',
                        hovertext=f'Feature: {self.feature}\t\t'
                                  f'Distance: {round(self.df["O1_dist"][i], 2)} km'
                                  f'<br>{self.df["clear_text"][i]}',
                        hoverinfo="name+text+lon+lat",
                        name="",
                        line=dict(
                            width=1,
                            color=self.palette[o+1]
                        ),
                        opacity=0.5
                    )
                )

        fig.update_layout(
            title_text=f'{self.prefix}<br>'
                       f'Error in distance between original and predicted geo locations of<br>'
                       f'{self.size} samples on {scope} scope, from model validated on {self.feature} feature',
            showlegend=False,
            geo=dict(
                scope=scope,
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
        )

        fig.update_geos(
            lataxis_showgrid=True, lonaxis_showgrid=True,
            visible=False, resolution=110,
            showcountries=True, countrycolor="Gray"
        )

        if save:
            # filename = f"results/img/{datetime.today().strftime('%Y-%m-%d')}_{self.feature}_map_{self.size}.png"
            # fig.write_image(filename)
            # print(f"VAL\tSAVE\tMap plot of {self.size} samples is drawn to file: {filename}")

            filename = f"results/map-html/intermap_{self.prefix}_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.html"
            fig.write_html(filename)
            print(f"VAL\tSAVE\tMap plot of {self.size} samples is drawn to file: {filename}")
        if not self.cluster:
            fig.show()

    # def gaus_compare(self, index=42, save=True):
    #     def get_gmm(outcomes, means, covs, weights, cov_type):
    #         sigmas = {
    #             "spherical": (outcomes,),
    #             "diag": (outcomes, 2),
    #             "full": (outcomes, 2, 2),
    #             "tied": (2, 2)
    #         }
    #         cov_type = "spherical" if cov_type == "spher" else cov_type
    #         if cov_type == "spherical":
    #             sigma = covs[:, 0, 0].reshape(sigmas[cov_type])
    #         elif cov_type == "diag":
    #             sigma = covs[:, [0, 1], [0, 1]].reshape(sigmas[cov_type])
    #         elif cov_type == "tied":
    #             sigma = covs[0, :].reshape(sigmas[cov_type])
    #         else:
    #             sigma = covs.reshape(sigmas[cov_type])
    #         gmm = GaussianMixture(n_components=outcomes, covariance_type=cov_type)
    #         gmm.means_ = means
    #         gmm.weights_ = weights
    #         gmm.covariances_ = sigma
    #         precisions_cholesky = _compute_precision_cholesky(sigma, cov_type).reshape(sigmas[cov_type])
    #         gmm.precisions_cholesky_ = precisions_cholesky
    #         if cov_type == "full":
    #             gmm.precisions_ = np.empty(gmm.precisions_cholesky_.shape)
    #             for k, prec_chol in enumerate(precisions_cholesky):
    #                 gmm.precisions_[k] = np.dot(prec_chol, prec_chol.T)
    #         elif cov_type == "tied":
    #             gmm.precisions_ = np.dot(precisions_cholesky, precisions_cholesky.T)
    #         else:
    #             gmm.precisions_ = precisions_cholesky**2
    #         return gmm
    #
    #     def get_full_gmm(outcomes, means, covs, weights, cov_type):
    #         sigma = covs + 1e-6
    #         gmm = GaussianMixture(n_components=outcomes, covariance_type="full")
    #         gmm.means_ = means
    #         gmm.weights_ = weights
    #         gmm.covariances_ = sigma
    #         gmm.precisions_cholesky_ = _compute_precision_cholesky(sigma, cov_type)
    #         if cov_type == "full":
    #             gmm.precisions_ = np.empty(gmm.precisions_cholesky_.shape)
    #             for k, prec_chol in enumerate(gmm.precisions_cholesky_):
    #                 gmm.precisions_[k] = np.dot(prec_chol, prec_chol.T)
    #         elif cov_type == "tied":
    #             gmm.precisions_ = np.dot(gmm.precisions_cholesky_, gmm.precisions_cholesky_.T)
    #         else:
    #             gmm.precisions_ = gmm.precisions_cholesky_**2
    #         return gmm
    #
    #     means = self.manager.means[index, :].reshape(self.outcomes, 2)
    #     covs = self.manager.covs[index, :].reshape(self.outcomes, 2, 2)
    #
    #     if self.outcomes > 1:
    #         weights = self.manager.weights[index, :].reshape(-1)
    #
    #     true = self.true[index, :]
    #     loss_lh = self.df.loc[index, f'lh_loss']
    #
    #     xmin, xmax = -180, 180
    #     ymin, ymax = -90, 90
    #     xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    #     XX = np.array([xx.ravel(), yy.ravel()]).T
    #
    #     #print(means, covs, true)
    #
    #     fig = plt.figure(figsize=(20, 20))
    #     #fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    #     #plt.title(f'GMMs with {self.outcomes} peaks')
    #
    #     gaus_model = ["pytorch", "gmm_full", "gmm_true"]
    #     cov_type = self.cov
    #
    #     nrows = len(gaus_model)
    #     ncols = 1
    #
    #     n = 1
    #     for gmm_func in gaus_model:
    #         print(gmm_func)
    #
    #         ax = plt.subplot(nrows, ncols, n)
    #         n += 1
    #
    #         ax.set_xlim(xmin, xmax)
    #         ax.set_ylim(ymin, ymax)
    #
    #         if gmm_func == "pytorch":
    #             gmm = get_gm_family(means, covs, weights)
    #             Z = gmm.log_prob(torch.from_numpy(XX)).numpy()
    #             gmm_lh = gmm.log_prob(torch.from_numpy(true)).numpy()
    #         elif gmm_func == "gmm_full":
    #             gmm = get_full_gmm(self.outcomes, means, covs, weights, self.cov)
    #             Z = gmm.score_samples(XX)
    #             gmm_lh = gmm.score_samples(true.reshape(1, -1))[0]
    #         elif gmm_func == "gmm_true":
    #             gmm = get_gmm(self.outcomes, means, covs, weights, cov_type)
    #             Z = gmm.score_samples(XX)
    #             gmm_lh = gmm.score_samples(true.reshape(1, -1))[0]
    #
    #         zmin = np.min(Z)
    #         zmax = np.max(Z)
    #         print(f"Original\tMin: {zmin}\tMax: {zmax}")
    #         Z = Z.reshape(xx.shape)
    #
    #         contour = ax.contour(xx, yy, Z, levels=np.linspace(zmin, zmax, 500), cmap='RdYlGn_r', linewidths=0.5)
    #
    #         map = Basemap(ax=ax)
    #         map.drawcoastlines(linewidth=0.5, color="gray")
    #         map.drawcountries(linewidth=0.7, color="gray")
    #         map.drawparallels(np.arange(ymin, ymax, 30.))
    #         map.drawmeridians(np.arange(xmin, xmax, 30.))
    #
    #         map.scatter(means[:, 0], means[:, 1], s=1, c="black", zorder=10)
    #
    #         map.scatter(true[0], true[1], color="red", s=5, zorder=11)
    #         plt.axvline(x=true[0], color="red", linestyle='--', lw=1)
    #         plt.axhline(y=true[1], color="red", linestyle='--', lw=1)
    #
    #         ax.set_yticks(range(-90, 90, 30))
    #         ax.set_xticks(range(-180, 180, 30))
    #         ax.title.set_text(gmm_func)
    #
    #     fig.text(0.5, 0.9, 'GMM', ha='center', va='center')
    #     fig.text(0.5, 0.04, 'Longitude', ha='center', va='center')
    #     fig.text(0.06, 0.5, 'Latitude', ha='center', va='center', rotation='vertical')
    #
    #     if save:
    #         filename = f"results/img/{self.prefix}_gaussian_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.png"
    #         plt.savefig(filename)
    #         print(f"VAL\tSAVE\tGaussian model of {self.size} samples is drawn to file: {filename}")
    #     if not self.cluster:
    #         plt.show()

    # def points_on_map(self, filename=None, original=True, pred=True, ua=False):
    #     fig, ax = plt.subplots(figsize=(30, 20))
    #     world = gpd.read_file(gpd.datasets.get_path(self.world_model))
    #     world.plot(ax=ax, color='white', edgecolor='gray', zorder=1)
    #
    #     title = ""
    #     location = "world map"
    #
    #     if original:
    #         geometry_original = [Point(xy) for xy in zip(self.df[self.x0], self.df[self.y0])]
    #         gdf_original = GeoDataFrame(self.df, geometry=geometry_original)
    #         gdf_original.plot(ax=ax, marker='o', color="black", markersize=3, zorder=2)
    #         title += "original"
    #
    #     if pred:
    #         geometry_predicted = [Point(xy) for xy in zip(self.df[self.x2], self.df[self.y2])]
    #         gdf_predicted = GeoDataFrame(self.df, geometry=geometry_predicted)
    #         for ctype, data in gdf_predicted.groupby('error'):
    #             color = self.palette[ctype]
    #             zorder = self.zorder[ctype] + 7
    #             data.plot(ax=ax,
    #                       color=color,
    #                       label=ctype,
    #                       zorder=zorder,
    #                       marker='o',
    #                       markersize=3)
    #         ax.legend()
    #         if len(title) > 0:
    #             title += " and predicted"
    #         else:
    #             title += "predicted"
    #
    #     if ua:
    #         ax.set_xlim(22.0, 41.0)
    #         ax.set_ylim(44.0, 53.0)
    #         location = "map of Ukraine"
    #
    #     ax.set(title=f'Points of {title} geo locations on the {location}')
    #
    #     if filename:
    #         plt.savefig(filename)
    #         print(f"VAL\tSAVE\tPlot of points of {self.size} samples is drawn to file: {filename}")
    #     if not self.cluster:
    #         plt.show()
    #
    # def lines_on_map(self, filename=None, ua=False):
    #     fig, ax = plt.subplots(figsize=(30, 20))
    #     world = gpd.read_file(gpd.datasets.get_path(self.world_model))
    #     world.plot(ax=ax, color='white', edgecolor='gray', zorder=1)
    #
    #     location = "world map"
    #
    #     geometry_lines = [LineString([[x1, y1], [x2, y2]]) for x1, y1, x2, y2 in
    #                       zip(self.df[self.x0], self.df[self.y0], self.df[self.x2], self.df[self.y2])]
    #     gdf_lines = GeoDataFrame(self.df, geometry=geometry_lines)
    #     for ctype, data in gdf_lines.groupby('error'):
    #         color = self.palette[ctype]
    #         zorder = self.zorder[ctype] + 2
    #         data.plot(ax=ax,
    #                   color=color,
    #                   label=ctype,
    #                   zorder=zorder,
    #                   linewidth=0.2,
    #                   legend=True)
    #
    #     ax.legend()
    #
    #     geometry_original = [Point(xy) for xy in zip(self.df[self.x0], self.df[self.y0])]
    #     gdf_original = GeoDataFrame(self.df, geometry=geometry_original)
    #     gdf_original.plot(ax=ax, marker='o', color="black", markersize=3, zorder=2)
    #
    #     geometry_predicted = [Point(xy) for xy in zip(self.df[self.x2], self.df[self.y2])]
    #     gdf_predicted = GeoDataFrame(self.df, geometry=geometry_predicted)
    #     for ctype, data in gdf_predicted.groupby('error'):
    #         color = self.palette[ctype]
    #         zorder = self.zorder[ctype] + 7
    #         data.plot(ax=ax,
    #                   color=color,
    #                   label=ctype,
    #                   zorder=zorder,
    #                   marker='o',
    #                   markersize=3)
    #
    #     if ua:
    #         ax.set_xlim(22.0, 41.0)
    #         ax.set_ylim(44.0, 53.0)
    #         location = "map of Ukraine"
    #
    #     ax.set(title=f'Error in distance between original and predicted geo locations on the {location}')
    #
    #     if filename:
    #         plt.savefig(filename)
    #         print(f"VAL\tSAVE\tPlot of points of {self.size} samples is drawn to file: {filename}")
    #     if not self.cluster:
    #         plt.show()


