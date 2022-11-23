import pandas as pd
import numpy as np
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
from scipy.ndimage.filters import maximum_filter
from scipy.special import softmax
import imageio
import os
import shutil
import moviepy.editor as mp
from moviepy.editor import *

from utils.regressor import *
from utils.benchmarks import *
from utils.result_manager import *

# visualization of evaluation results


def plot_gmm(samples, outcomes, means, covs, weights, cluster, title, filename, save=True):
    def map_grid(peaks, step=200):
        xmin, xmax = -180, 180
        ymin, ymax = -90, 90
        x = np.linspace(xmin, xmax, step)
        y = np.linspace(ymin, ymax, step)
        x = np.concatenate((x, peaks[:, 0]), axis=0)
        x = np.sort(x)
        y = np.concatenate((y, peaks[:, 1]), axis=0)
        y = np.sort(y)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    palette = {1: 'darkgreen',
                2: 'goldenrod',
                3: 'darkorange',
                4: 'crimson',
                5: 'darkred'}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))

    xmin, xmax = -180, 180
    ymin, ymax = -90, 90
    step_big = 45.
    ticks_big_x, ticks_big_y = range(int(xmin), int(xmax), int(step_big)), \
                               range(int(ymin), int(ymax), int(step_big))
    tick_labels_big_x, tick_labels_big_y = [str(x) for x in ticks_big_x], \
                                           [str(y) for y in ticks_big_y]

    xxb, yyb = map_grid(means[:, 0], 50) if samples > 1 else map_grid(means.reshape(-1, 2), 400)
    XX_big = np.array([xxb.ravel(), yyb.ravel()]).T

    if samples > 1:
        Z_big, Z_big_exp = np.zeros_like(xxb).flatten(), np.zeros_like(xxb).flatten()
        for i in range(samples):
            weights = weights[i, :] if outcomes > 1 else None
            gmm = get_gm_family(outcomes, means[i, :], covs[i, :], weights)
            Z_big += gmm.log_prob(torch.from_numpy(XX_big)).numpy()
            Z_big_exp += np.exp(gmm.log_prob(torch.from_numpy(XX_big)).numpy())

        Z = Z_big_exp.reshape(xxb.shape)
        local_max_indexes = np.where(1 == (Z == maximum_filter(Z, mode="nearest", size=(10, 10))))
        ind = np.ravel_multi_index(local_max_indexes, Z.shape)

        max_Z = Z_big_exp[ind]
        max_XX = XX_big[ind]
        max_XX_uni, ind_uni = np.unique(max_XX, return_index=True, axis=0)
        max_Z_uni = max_Z[ind_uni]
        print(f"Found {max_XX.shape[0]} local maximums - {max_XX_uni.shape[0]} unique peaks")

        top = 5 if len(ind_uni) > 5 else len(ind_uni)
        ind_top_5 = (-max_Z_uni).argsort()[:top]
        peaks = max_XX_uni[ind_top_5]
        p_weights = softmax(max_Z_uni[ind_top_5])

        ind = np.argwhere(np.round(p_weights * 100, 2) > 0)
        print(f"Regressing to top {top} - {len(ind)} significant peaks")
        significant = peaks[ind].reshape(len(ind), 2)
        sig_weights = p_weights[ind].flatten()

    else:
        gmm = get_gm_family(outcomes, means, covs, weights)
        Z_big = gmm.log_prob(torch.from_numpy(XX_big)).numpy()

        if outcomes > 1:
            ind = np.argwhere(np.round(weights * 100, 2) > 0)
            significant = means[ind].reshape(-1, 2)
            sig_weights = weights[ind].flatten()
        else:
            significant = means.reshape(-1, 2)
            sig_weights = np.ones(1)

    for i in range(len(sig_weights)):
        weight = np.round(sig_weights[i] * 100, 2)
        point = f"lon: {'  lat: '.join(map(str, significant[i])) }"
        if weight > 0:
            print(f"\tOut {i+1}\t{weight}%\t-\t{point}")

    margin_lon, margin_lat = 10, 5
    min_lon, max_lon = min(significant[:, 0]) - margin_lon, max(significant[:, 0]) + margin_lon
    min_lat, max_lat = min(significant[:, 1]) - margin_lat, max(significant[:, 1]) + margin_lat
    step_zoom = 15.0
    ticks_zoom_x, ticks_zoom_y = range(int(min_lon), int(max_lon), int(step_zoom)), \
                                 range(int(min_lat), int(max_lat), int(step_zoom))
    tick_labels_zoom_x, tick_labels_zoom_y = [str(x) for x in ticks_zoom_x], \
                                             [str(y) for y in ticks_zoom_y]
    xxz, yyz = np.mgrid[min_lon:max_lon:400j, min_lat:max_lat:400j]
    XX_zoom = np.array([xxz.ravel(), yyz.ravel()]).T

    if samples > 1:
        Z_zoom = np.zeros_like(xxz).flatten()
        for i in range(samples):
            weights = weights[i, :] if outcomes > 1 else None
            gmm = get_gm_family(outcomes, means[i, :], covs[i, :], weights)
            Z_zoom += np.exp(gmm.log_prob(torch.from_numpy(XX_zoom)).numpy())
    else:
        gmm = get_gm_family(outcomes, means, covs, weights)
        Z_zoom = np.exp(gmm.log_prob(torch.from_numpy(XX_zoom)).numpy())

    Z_big, Z_zoom = Z_big.reshape(xxb.shape), Z_zoom.reshape(xxz.shape)
    zbmin, zzmin = np.min(Z_big), np.min(Z_zoom)
    zbmax, zzmax = np.max(Z_big), np.max(Z_zoom)

    ax_big = ax[0]

    ax_big.set_xlim(xmin, xmax)
    ax_big.set_ylim(ymin, ymax)
    ax_big.set_title(f'Log-Likelihood score of {samples} GMM{"s" if samples > 2 else "" }', y=1.0, pad=24)

    map_big = Basemap(ax=ax_big, projection='mill', resolution='l')
    map_big.drawcoastlines(linewidth=0.5, color="black", zorder=2)
    map_big.drawcountries(linewidth=0.7, color="black", zorder=3)
    map_big.drawparallels(np.arange(ymin, ymax, step_big), labels=tick_labels_big_y)
    map_big.drawmeridians(np.arange(xmin, xmax, step_big), labels=tick_labels_big_x)
    map_big.drawmapboundary(fill_color='lightgrey', zorder=0)
    map_big.fillcontinents(color='white', lake_color='lightgrey', zorder=1)

    contour_big = map_big.contourf(xxb, yyb, Z_big, levels=np.linspace(zbmin, zbmax, 250),
                                   cmap='Spectral_r', alpha=0.7, zorder=9, latlon=True)
    plt.colorbar(contour_big, ax=ax_big, orientation="horizontal", pad=0.2)

    ax_zoom = ax[1]
    ax_zoom.set_xlim(min_lon, max_lon)
    ax_zoom.set_ylim(min_lat, max_lat)
    ax_zoom.set_title('Max Probability Density Function region', y=1.0, pad=24)

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

    Z_zoom = np.ma.array(Z_zoom, mask=Z_zoom < 1e-5)
    contour_zoom = map_zoom.contourf(xxz, yyz, Z_zoom, levels=np.linspace(zzmin, zzmax, 250),
                                     cmap='Spectral_r', alpha=0.7, zorder=9, latlon=True)
    plt.colorbar(contour_zoom, ax=ax_zoom, orientation="horizontal", pad=0.2)

    for i in range(len(sig_weights)):
        color = palette[i+1] if i < 5 else palette[5]
        point = f"lon: {'  lat: '.join(map(str, significant[i])) }"
        label = point + f" - {round(sig_weights[i] * 100, 2)}%",
        map_zoom.scatter(significant[i, 0], significant[i, 1], latlon=True,
                         label=label, color=color,
                         s=10, zorder=9999)
        map_zoom.scatter(significant[i, 0], significant[i, 1], latlon=True,
                         color=color, alpha=0.2,
                         s=100, zorder=9999)

    plt.legend(loc='upper center', title="Predicted outcomes", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
    plt.suptitle(title)

    if save:
        plt.savefig(filename, dpi=300)
        print(f"VAL\tSAVE\tGaussian model is drawn to file: {filename}")
    if not cluster:
        plt.show()


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

        # palette for sorted by weight outcomes
        self.palette = {1: 'darkgreen',
                        2: 'goldenrod',
                        3: 'darkorange',
                        4: 'crimson',
                        5: 'darkred'}

    # multiple tweets GMM NLLH contour on the map
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

            weights = self.manager.weights[i, :] if self.outcomes > 1 else None
            gmm = get_gm_family(self.outcomes, self.manager.means[i, :], self.manager.covs[i, :], weights)

            Z = gmm.log_prob(torch.from_numpy(XX)).numpy()
            gmm_lh = gmm.log_prob(torch.from_numpy(self.true[i, :])).numpy()

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

    def summarize_prediction(self, user_index=0, samples=100, save=True):
        user = self.df['USER-ONLY'].unique()[user_index]
        ids = self.df.index[self.df['USER-ONLY'] == user].tolist()[0:samples]
        size = len(ids)
        means, covs, weights = self.manager.means[ids], self.manager.covs[ids], self.manager.weights[ids]

        title = f'{self.prefix}\nsummary of {size} tweet GMMs with {self.outcomes} means' \
                f'\nUser: {user}'
        if save:
            filename = f"results/img/gmm_user_summary_S{size}_{self.prefix}_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.png"
        else:
            filename = None

        print(f"User:\t{user}")
        print(f"Estimating user geolocation from {size} tweet GMM predictions")
        plot_gmm(size, self.outcomes, means, covs, weights, self.cluster, title, filename, save)

    # single tweet GMM NLLH contour on the map
    def gaus_map(self, index=42, save=True):
        title = f'{self.prefix}\nplots of tweet GMM with {self.outcomes} means - sample {index}' \
                f'\nText: {self.df.loc[index, self.feature]}'
        if save:
            filename = f"results/img/gaussian_sample_map_ID-{index}_{self.prefix}_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.png"
        else:
            filename = None

        means, covs, weights = self.manager.means[index], self.manager.covs[index], self.manager.weights[index]

        plot_gmm(1, self.outcomes, means, covs, weights, self.cluster, title, filename, True)

    # single text results visualization on the map
    def text_map_result(self, index=0, save=True):
        if self.prob:
            title = f'{self.prefix}\nplots of GMM with {self.outcomes} means'
            if self.manager.text:
                title += f"\nText: {self.manager.text}\n"

            if save:
                filename = f"results/img/text_map_{self.prefix}_{datetime.today().strftime('%Y-%m-%d')}.png"
            else:
                filename = None

            means, covs = self.manager.means[0], self.manager.covs[0]
            weights = self.manager.weights[0] if self.outcomes > 1 else None
            plot_gmm(1, self.outcomes, means, covs, weights, self.cluster, title, filename, True)
        else:

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))

            xmin, xmax = -180, 180
            ymin, ymax = -90, 90
            step_big = 45.
            ticks_big_x, ticks_big_y = range(int(xmin), int(xmax), int(step_big)), \
                                       range(int(ymin), int(ymax), int(step_big))
            tick_labels_big_x, tick_labels_big_y = [str(x) for x in ticks_big_x], \
                                                   [str(y) for y in ticks_big_y]

            if self.outcomes > 1:
                ind = np.argwhere(np.round(self.manager.weights[index, :] * 100, 2) > 0)
                significant = self.manager.means[index, ind].reshape(-1, 2)
                sig_weights = self.manager.weights[index, ind].flatten()
            else:
                significant = self.manager.means.reshape(-1, 2)
                sig_weights = np.ones(1)

            for i in range(len(sig_weights)):
                weight = np.round(sig_weights[i] * 100, 2)
                point = f"lon: {'  lat: '.join(map(str, significant[i])) }"
                if weight > 0:
                    print(f"\tOut {i+1}\t{weight}%\t-\t{point}")

            margin_lon, margin_lat = 10, 5
            min_lon, max_lon = min(significant[:, 0]) - margin_lon, max(significant[:, 0]) + margin_lon
            min_lat, max_lat = min(significant[:, 1]) - margin_lat, max(significant[:, 1]) + margin_lat
            step_zoom = 5.0
            ticks_zoom_x, ticks_zoom_y = range(int(min_lon), int(max_lon), int(step_zoom)), \
                                         range(int(min_lat), int(max_lat), int(step_zoom))
            tick_labels_zoom_x, tick_labels_zoom_y = [str(x) for x in ticks_zoom_x], \
                                                     [str(y) for y in ticks_zoom_y]

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

            if self.outcomes > 1:
                for i in range(self.outcomes):
                    color = self.palette[i+1] if i < 5 else self.palette[5]
                    map_big.scatter(self.manager.means[index, i, 0],
                                    self.manager.means[index, i, 1],
                                    color=color,
                                    s=10 * self.manager.weights[index, i],
                                    latlon=True,
                                    zorder=9999)
                    map_big.scatter(self.manager.means[index, i, 0],
                                    self.manager.means[index, i, 1],
                                    color=color,
                                    alpha=0.2,
                                    s=max(100, 1000 * self.manager.weights[index, i]),
                                    latlon=True,
                                    zorder=9999)
            else:
                map_big.scatter(self.manager.means[index, 0],
                                self.manager.means[index, 1],
                                latlon=True,
                                s=10, c="black", zorder=9999)
                map_big.scatter(self.manager.means[index, 0],
                                self.manager.means[index, 1],
                                latlon=True,
                                s=100, c="black", alpha=0.2, zorder=9999)

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

            if self.outcomes > 1:
                for i in range(self.outcomes):
                    if np.round(self.manager.weights[index, i] * 100, 2) > 0:
                        color = self.palette[i+1] if i < 5 else self.palette[5]
                        map_zoom.scatter(self.manager.means[index, i, 0],
                                         self.manager.means[index, i, 1],
                                         latlon=True,
                                         label=f"Out {i+1}: { ', '.join(map(str, self.manager.means[index, i]))} - {round(self.manager.weights[index, i] * 100, 2)}%",
                                         color=color,
                                         s=max(1, 10 * self.manager.weights[index, i]),
                                         zorder=9999)
                        map_zoom.scatter(self.manager.means[index, i, 0],
                                         self.manager.means[index, i, 1],
                                         latlon=True,
                                         color=color,
                                         s=max(100, 1000 * self.manager.weights[index, i]),
                                         alpha=0.2,
                                         zorder=9999)
            else:
                map_zoom.scatter(self.manager.means[index, 0],
                                 self.manager.means[index, 1],
                                 latlon=True,
                                 s=10, c="black", zorder=9999)
                map_zoom.scatter(self.manager.means[index, 0],
                                 self.manager.means[index, 1],
                                 latlon=True,
                                 label=f"Out 1: { ', '.join(map(str, self.manager.means))}",
                                 s=100, c="black", alpha=0.2, zorder=9999)

            title = f'{self.prefix}\nscatter plots of {self.outcomes} points'
            if self.manager.text:
                title += f"\nText: {self.manager.text}"

            plt.legend(loc='upper center', title="Predicted outcomes", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
            plt.suptitle(title)

            if save:
                filename = f"results/img/text_map_{self.prefix}_{datetime.today().strftime('%Y-%m-%d')}.png"
                plt.savefig(filename, dpi=300)
                print(f"VAL\tSAVE\tScatter plot of text prediction is drawn to file: {filename}")
            if not self.cluster:
                plt.show()

    # Densities on log scale per outcome
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

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', title="Median per outcome", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
        plt.title(f'{self.prefix}\nDensity Plot for Distance error')
        plt.xlabel('Error (km)')
        plt.ylabel('Density')

        if save:
            filename = f"results/img/density_{self.prefix}_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.png"
            plt.savefig(filename, dpi=300)
            print(f"VAL\tSAVE\tDistance density of {self.size} samples is drawn to file: {filename}")
        if not self.cluster:
            plt.show()

    # Cumulative Distribution per outcome
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

        plt.title(f'{self.prefix}\nCumulative distribution for distance error with {threshold} km threshold')
        plt.xlabel('log Distance error (km)')
        plt.ylabel('Proportion')

        if save:
            filename = f"results/img/cum_dist_{self.prefix}_N{self.size}_{datetime.today().strftime('%Y-%m-%d')}.png"
            plt.savefig(filename, dpi=300)
            print(
                f"VAL\tSAVE\tError distance cumulative distribution of {self.size} samples is drawn to file: {filename}")
        if not self.cluster:
            plt.show()

    # distance error lines on the map
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
