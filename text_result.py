import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import math
import torch
import torch.distributions as dist
import numpy as np

import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig, BertTokenizer


# general model wrapper
# linear regression fork for features and preset outputs
class BERTregModel():
    def __init__(self, n_outcomes=5, covariance="spher", base_model_name=None, hub_model=None):
        self.n_outcomes = n_outcomes
        self.cov = covariance
        self.features = ["NON-GEO", "GEO-ONLY"]

        print(f"MODEL\tInitializing BERT Regression model for {self.n_outcomes} outcome(s)")
        # features
        print(f"MODEL\tText features:\t{' + '.join(self.features)}")
        # longitude, latitude for n outcomes
        self.coord_output = self.n_outcomes * 2
        print(f"MODEL\tCoordinates:\t{self.coord_output}")
        # weights of gaussians
        self.weights_output = self.n_outcomes
        print(f"MODEL\tWeights:\t{self.weights_output}")

        if self.cov is None:
            self.cov_output = 0
            print(f"MODEL\tNon-probabilistic model has been chosen")
        else:
            self.cov_output = self.n_outcomes
            print(f"MODEL\tCovariances:\t{self.cov_output}\tmatrix type:\t{self.cov}")

        self.original_model = "bert-base-multilingual-cased" if base_model_name is None else base_model_name
        print(f"MODEL\tOriginal model to load:\t{self.original_model}")

        self.key_output = self.coord_output + self.weights_output + self.cov_output
        self.minor_output = 2
        self.minor_output += 1 if self.cov_output > 0 else 0

        self.feature_outputs = {}
        for f in range(len(self.features)):
            if f == 0:
                output = self.key_output
                print(f"MODEL\tKey feature \t{self.features[f]} outputs:\t{output}")
            else:
                output = self.minor_output
                print(f"MODEL\tMinor feature\t{self.features[f]} outputs:\t{output}")
            self.feature_outputs[self.features[f]] = output

        self.model = GeoBertModel(BertConfig.from_pretrained(self.original_model), self.feature_outputs)
        if hub_model:
            print(f"LOAD\tLoading HF model from {hub_model}")
            self.model = self.model.from_pretrained(hub_model, self.feature_outputs)


# HF model wrapper layer
class GeoBertModel(BertPreTrainedModel):
    def __init__(self, config, feature_outputs):
        super().__init__(config)
        self.bert = BertModel(config)
        self.feature_outputs = feature_outputs

        self.key_regressor = nn.Linear(config.hidden_size, list(self.feature_outputs.values())[0])
        if len(self.feature_outputs) > 1:
            self.minor_regressor = nn.Linear(config.hidden_size, list(self.feature_outputs.values())[1])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, feature_name=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        pooler_output = outputs[1]
        if feature_name is None or feature_name == list(self.feature_outputs.keys())[0]:
            custom_output = self.key_regressor(pooler_output)
        else:
            custom_output = self.minor_regressor(pooler_output)
        return custom_output


# result manager
class ResultManager():
    def __init__(self, text, feature, device, model, prefix=None):
        self.cluster = device.type == "cuda"
        self.feature = feature
        if prefix is None:
            prefix = feature
        self.prefix = prefix

        self.model = model

        self.outcomes = self.model.n_outcomes
        self.cov = self.model.cov

        self.outputs_map = {
            "coord": [0, self.model.coord_output],
            "weight": [self.model.coord_output,
                       self.model.coord_output + self.model.weights_output],
            "sigma": [self.model.coord_output + self.model.weights_output,
                      self.model.coord_output + self.model.weights_output + self.model.cov_output] if self.cov else None
        }

        self.prob = self.cov is not None

        self.text = text

        self.size = 1

        self.means = None
        self.weights = None
        self.covs = None


    # sort per-tweet outcomes by their weights
    def sort_outcomes(self):
        print(f"RESULT\tSorting all outputs for {self.outcomes} outcomes by probabilistic weights")

        sort_indexes = np.argsort(self.weights[0])
        index = sort_indexes[::-1]
        self.means[0] = self.means[0, index]
        self.weights[0] = self.weights[0, index]
        if self.prob:
            self.covs[0] = self.covs[0, index]


    # raw outputs to params
    def raw_to_params(self, predicted):
        P = predicted[0, self.outputs_map["coord"][0]:self.outputs_map["coord"][1]]
        means = P.reshape(self.size, self.outcomes, 2)
        self.means = means.cpu().numpy() if self.cluster else means.numpy()

        softmax = nn.Softmax(dim=0)
        W = predicted[0, self.outputs_map["weight"][0]:self.outputs_map["weight"][1]]
        weights = softmax(W).reshape(self.size, self.outcomes)
        self.weights = weights.cpu().numpy() if self.cluster else weights.numpy()

        if self.prob:
            softplus = nn.Softplus()
            S = softplus(predicted[0, self.outputs_map["sigma"][0]:self.outputs_map["sigma"][1]]) + 1 / (2 * math.pi)
            sigma = torch.eye(2, device=predicted.device) * S.reshape(-1, 1)[:, None]
            covs = sigma.reshape([self.size, self.outcomes, 2, 2])
            self.covs = covs.cpu().numpy() if self.cluster else covs.numpy()

        self.sort_outcomes()


# GMM
def dist_gmm(outcomes, means, sigma, weights):
    means, sigma = means.reshape(outcomes, 2), sigma.reshape(outcomes, 2, 2)
    gaussian = dist.MultivariateNormal(torch.from_numpy(means), torch.from_numpy(sigma))
    gmm_weights = dist.Categorical(torch.from_numpy(weights.reshape(-1)))
    gmm = dist.MixtureSameFamily(gmm_weights, gaussian)
    return gmm


# generating map grid with intergrid peaks
def map_grid(peaks, step=10):
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


# visualization of results
class ResultVisuals():
    def __init__(self, manager):
        self.manager = manager
        self.cluster = self.manager.cluster
        self.feature = self.manager.feature
        self.prefix = self.manager.prefix

        self.size = self.manager.size

        self.prob = self.manager.prob
        self.outcomes = self.manager.outcomes
        self.cov = self.manager.cov

        # palette for sorted by weight outcomes
        self.palette = {1: 'darkgreen',
                        2: 'goldenrod',
                        3: 'darkorange',
                        4: 'crimson',
                        5: 'darkred'}


    # scatter plots
    def plot_scatter(self, means, weights, title):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))

        xmin, xmax = -180, 180
        ymin, ymax = -90, 90
        step_big = 45.
        ticks_big_x, ticks_big_y = range(int(xmin), int(xmax), int(step_big)), \
            range(int(ymin), int(ymax), int(step_big))
        tick_labels_big_x, tick_labels_big_y = [str(x) for x in ticks_big_x], \
            [str(y) for y in ticks_big_y]

        ind = np.argwhere(np.round(weights * 100, 2) > 0)
        significant = means[ind].reshape(-1, 2)

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

        for i in range(self.outcomes):
            color = self.palette[i + 1] if i < 5 else self.palette[5]
            map_big.scatter(means[i, 0],
                            means[i, 1],
                            color=color,
                            s=10 * weights[i],
                            latlon=True,
                            zorder=9999)
            map_big.scatter(means[i, 0],
                            means[i, 1],
                            color=color,
                            alpha=0.2,
                            s=max(100, 1000 * weights[i]),
                            latlon=True,
                            zorder=9999)

        ax_zoom = ax[1]
        ax_zoom.set_xlim(min_lon, max_lon)
        ax_zoom.set_ylim(min_lat, max_lat)

        map_zoom = Basemap(ax=ax_zoom, projection='mill',
                           llcrnrlat=min_lat,
                           llcrnrlon=min_lon,
                           urcrnrlat=max_lat,
                           urcrnrlon=max_lon, resolution='h')
        map_zoom.drawcoastlines(linewidth=0.5, color="black", zorder=2)
        map_zoom.drawcountries(linewidth=0.7, color="black", zorder=3)
        map_zoom.drawmapboundary(fill_color='lightgrey', zorder=0)
        map_zoom.drawparallels(np.arange(min_lat, max_lat, step_zoom), labels=tick_labels_zoom_y)
        map_zoom.drawmeridians(np.arange(min_lon, max_lon, step_zoom), labels=tick_labels_zoom_x)
        map_zoom.fillcontinents(color='white', lake_color='lightgrey', zorder=1)

        for i in range(self.outcomes):
            if np.round(weights[i] * 100, 2) > 0:
                color = self.palette[i + 1] if i < 5 else self.palette[5]
                map_zoom.scatter(means[i, 0],
                                 means[i, 1],
                                 latlon=True,
                                 label=f"Out {i + 1}: {', '.join(map(str, means[i]))} - {round(weights[i] * 100, 2)}%",
                                 color=color,
                                 s=max(1, 10 * weights[i]),
                                 zorder=9999)
                map_zoom.scatter(means[i, 0],
                                 means[i, 1],
                                 latlon=True,
                                 color=color,
                                 s=max(100, 1000 * weights[i]),
                                 alpha=0.2,
                                 zorder=9999)

        plt.legend(loc='upper center', title="Predicted outcomes", bbox_to_anchor=(0.5, -0.1), fancybox=True,
               shadow=True)
        plt.suptitle(title)

        plt.show()


    # GMM plots
    def plot_gmm(self, means, covs, weights, title):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))

        xmin, xmax = -180, 180
        ymin, ymax = -90, 90
        step_big = 45.
        ticks_big_x, ticks_big_y = range(int(xmin), int(xmax), int(step_big)), \
                                   range(int(ymin), int(ymax), int(step_big))
        tick_labels_big_x, tick_labels_big_y = [str(x) for x in ticks_big_x], \
                                               [str(y) for y in ticks_big_y]

        total_peaks = means.reshape(-1, 2)
        peaks_unique, ind = np.unique(np.round(total_peaks, 4), return_index=True, axis=0)
        intergrid_peaks = total_peaks[ind]
        gmm = dist_gmm(self.outcomes, means, covs, weights)

        # BIG MAP
        grid_step = 400
        xxb, yyb = map_grid(intergrid_peaks, grid_step)
        XX_big = np.array([xxb.ravel(), yyb.ravel()]).T

        Z_big = gmm.log_prob(torch.from_numpy(XX_big)).numpy()

        ind = np.argwhere(np.round(weights * 100, 2) > 0)
        significant = means[ind].reshape(-1, 2)
        sig_weights = weights[ind].flatten()

        margin_lon, margin_lat = 10, 5
        min_lon, max_lon = min(significant[:, 0]) - margin_lon, max(significant[:, 0]) + margin_lon
        min_lat, max_lat = min(significant[:, 1]) - margin_lat, max(significant[:, 1]) + margin_lat

        # ZOOM MAP
        step_zoom = 15.0
        ticks_zoom_x, ticks_zoom_y = range(int(min_lon), int(max_lon), int(step_zoom)), \
                                     range(int(min_lat), int(max_lat), int(step_zoom))
        tick_labels_zoom_x, tick_labels_zoom_y = [str(x) for x in ticks_zoom_x], \
                                                 [str(y) for y in ticks_zoom_y]
        xxz, yyz = np.mgrid[min_lon:max_lon:400j, min_lat:max_lat:400j]
        XX_zoom = np.array([xxz.ravel(), yyz.ravel()]).T

        Z_zoom = np.exp(gmm.log_prob(torch.from_numpy(XX_zoom)).numpy())

        # MAPS
        Z_big, Z_zoom = Z_big.reshape(xxb.shape), Z_zoom.reshape(xxz.shape)
        zbmin, zzmin = np.min(Z_big), np.min(Z_zoom)
        zbmax, zzmax = np.max(Z_big), np.max(Z_zoom)

        # BIF MAP
        ax_big = ax[0]

        ax_big.set_xlim(xmin, xmax)
        ax_big.set_ylim(ymin, ymax)
        ax_big.set_title(f'Log-Likelihood score of GMM', y=1.0, pad=24)

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

        # ZOOM MAP
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

        Z_zoom = np.ma.array(Z_zoom, mask=Z_zoom < 1e-8)
        contour_zoom = map_zoom.contourf(xxz, yyz, Z_zoom, levels=np.linspace(zzmin, zzmax, 250),
                                         cmap='Spectral_r', alpha=0.7, zorder=9, latlon=True)
        plt.colorbar(contour_zoom, ax=ax_zoom, orientation="horizontal", pad=0.2)

        for i in range(len(sig_weights)):
            color = self.palette[i+1] if i < 5 else self.palette[5]
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

        plt.show()

    # single text results visualization on the map
    def text_map_result(self):
        means = self.manager.means[0]
        weights = self.manager.weights[0]
        if self.prob:
            title = f'{self.prefix}\nplots of GMM with {self.outcomes} means'
            if self.manager.text:
                title += f"\nText: {self.manager.text}\n"

            covs = self.manager.covs[0]
            self.plot_gmm(means, covs, weights, title)

        else:
            title = f'{self.prefix}\nscatter plots of {self.outcomes} points'
            if self.manager.text:
                title += f"\nText: {self.manager.text}\n"
            self.plot_scatter(means, weights, title)


def main():
    hub_model = 'k4tel/geo-bert-multilingual'
    base_model = "bert-base-multilingual-cased"

    model_wrapper = BERTregModel(5, "spher", base_model, hub_model)
    tokenizer = BertTokenizer.from_pretrained(hub_model)

    text = "CIA and FBI can track anyone, and you willingly give the data away"

    result = ResultManager(text, "NON-GEO", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                           model_wrapper, hub_model)
    model = model_wrapper.model

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"RESULT\tPost-processing raw model outputs: {outputs}")
    result.raw_to_params(outputs)

    ind = np.argwhere(np.round(result.weights[0, :] * 100, 2) > 0)
    significant = result.means[0, ind].reshape(-1, 2)
    weights = result.weights[0, ind].flatten()

    sig_weights = np.round(weights * 100, 2)
    sig_weights = sig_weights[sig_weights > 0]

    print(f"RESULT\t{len(sig_weights)} significant prediction outcome(s):")

    for i in range(len(sig_weights)):
        point = f"lon: {'  lat: '.join(map(str, significant[i]))}"
        print(f"\tOut {i + 1}\t{sig_weights[i]}%\t-\t{point}")

    # visual = ResultVisuals(result)
    # visual.text_map_result()


if __name__ == "__main__":
    main()