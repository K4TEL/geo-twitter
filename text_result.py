import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import math
import torch
import torch.distributions as dist

import numpy as np
import pandas as pd

import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig, BertTokenizer

import string
import re

from transformers import Pipeline, pipeline
from transformers.pipelines import PIPELINE_REGISTRY


# general model wrapper
# linear regression fork for features and preset outputs
class BERTregModel():
    def __init__(self, n_outcomes=5, covariance="spher", base_model_name=None):
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

        self.outputs_map = {
            "coord": [0, self.coord_output],
            "weight": [self.coord_output,
                       self.coord_output + self.weights_output],
            "sigma": [self.coord_output + self.weights_output,
                      self.coord_output + self.weights_output + self.cov_output] if self.cov else None
        }

        self.model = GeoBertModel(BertConfig.from_pretrained(self.original_model), self.feature_outputs)
        self.tokenizer = BertTokenizer.from_pretrained(self.original_model)


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


# torch.nn.Softplus() in munpy
def softplus(x, threshold=20):
    x_clipped = np.clip(x, -threshold, threshold)
    result = np.log(1 + np.exp(x_clipped))
    result[x > threshold] = x[x > threshold]
    return result + (1 / (2 * np.pi))


# torch.nn.Softmax() in munpy
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum


# filtering
def filter_text(text):
    print(f"TEXT\tFiltering text: {text}")
    pattern = r'http\S+'
    text = re.sub(pattern, '', text)
    text = "".join([i for i in text if i not in string.punctuation])
    return text


# custom HF pipeline
class GeoRegressorPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "filter" in kwargs:
            preprocess_kwargs["filter"] = kwargs["filter"]

        postprocess_kwargs = {}
        if "outputs_map" in kwargs:
            postprocess_kwargs["outputs_map"] = kwargs["outputs_map"]

        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, text, filter=True):
        text = filter_text(text) if filter else text
        return self.tokenizer(text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, outputs_map=None):
        print(f"RESULT\tPost-processing raw model outputs: {model_outputs}")
        model_outputs = model_outputs.numpy() if self.device == "cpu" else model_outputs.cpu().numpy()

        outcomes = outputs_map["weight"][1] - outputs_map["weight"][0]

        P = model_outputs[0, outputs_map["coord"][0]:outputs_map["coord"][1]]
        means = P.reshape([outcomes, 2])

        W = model_outputs[0, outputs_map["weight"][0]:outputs_map["weight"][1]]
        weights = softmax(W, axis=0).reshape(outcomes)

        if outputs_map["sigma"]:
            S = model_outputs[0, outputs_map["sigma"][0]:outputs_map["sigma"][1]]
            S = softplus(S)
            sigma = np.eye(2).reshape(1, 2, 2) * S.reshape(-1, 1, 1)
            covs = sigma.reshape([outcomes, 2, 2])

        print(f"RESULT\tSorting all outputs for {outcomes} outcomes by probabilistic weights")
        sort_indexes = np.argsort(weights)
        index = sort_indexes[::-1]

        means, weights = means[index], weights[index]
        if outputs_map["sigma"]:
            covs = covs[index]

        result = []
        for i in range(5):
            result.append({"point": means[i],
                           "weight": weights[i],
                           "cov": covs[i] if outputs_map["sigma"] else None})

        return result


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
        self.outputs_map = self.model.outputs_map

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

    # raw model outputs to params
    def raw_to_params(self, predicted):
        print(f"RESULT\tPost-processing raw model outputs: {predicted}")
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

    # pipeline result to params
    def pipeline_to_params(self, result_dict):
        self.means = np.zeros((1, self.outcomes, 2), float)
        self.weights = np.zeros((1, self.outcomes), float)
        self.covs = np.zeros((1, self.outcomes, 2, 2), float) if self.prob else None

        for i in range(self.outcomes):
            self.means[0, i, :] = result_dict[i]["point"]
            self.weights[0, i] = result_dict[i]["weight"]
            if self.prob:
                self.covs[0, i, :] = result_dict[i]["cov"]


# GMM
def dist_gmm(outcomes, means, covs, weights):
    means, covs = means.reshape(outcomes, 2), covs.reshape(outcomes, 2, 2)
    gaussian = dist.MultivariateNormal(torch.from_numpy(means), torch.from_numpy(covs))
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


# load model or model pipeline
def load_model(base_model, hub_model, use_pipeline=True):
    model_wrapper = BERTregModel(5, "spher", base_model)
    model_wrapper.tokenizer = BertTokenizer.from_pretrained(hub_model)
    model_wrapper.prefix = hub_model

    if use_pipeline:
        PIPELINE_REGISTRY.register_pipeline(
            "geo-regressor",
            pipeline_class=GeoRegressorPipeline,
            pt_model=GeoBertModel,
            default={"pt": (hub_model, "main")},
            type="text",
        )

        print(f"LOAD\tLoading HF model from {hub_model}")
        model_wrapper.pipeline = pipeline("geo-regressor",
                                          model=hub_model,
                                          tokenizer=model_wrapper.tokenizer,
                                          device="cuda" if torch.cuda.is_available() else "cpu",
                                          outputs_map=model_wrapper.outputs_map,
                                          filter=filter,
                                          model_kwargs={"feature_outputs": model_wrapper.feature_outputs})
    else:
        print(f"LOAD\tLoading HF model from {hub_model}")
        model_wrapper.model = model_wrapper.model.from_pretrained(hub_model, model_wrapper.feature_outputs)
    return model_wrapper


# get prediction result for a single text
def text_prediction(model_wrapper, text, use_pipeline=True, filter=True):
    result = ResultManager(text, "NON-GEO", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                           model_wrapper, model_wrapper.prefix)
    if use_pipeline:
        outputs = model_wrapper.pipeline(text, filter=filter)
        result.pipeline_to_params(outputs)
    else:
        text = filter_text(text) if filter else text
        inputs = model_wrapper.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model_wrapper.model(**inputs)
        result.raw_to_params(outputs)

    return result


# get the most probable location predicted for a union of columns in pandas df
def df_prediction(model_wrapper, df, target_columns_list, use_pipeline=True, filter=True):
    input_col = "raw_input"

    def row_predict(row):
        result = text_prediction(model_wrapper, row[input_col], use_pipeline, filter)
        return tuple(result.means[0, 1])

    df[input_col] = df[target_columns_list].apply(lambda x: ' '.join(x), axis=1)
    df["location"] = df.apply(row_predict, axis=1)
    df.drop(input_col, axis=1, inplace=True)

    return df


def main():
    hub_model = 'k4tel/geo-bert-multilingual'
    base_model = "bert-base-multilingual-cased"
    use_pipeline = True

    model_wrapper = load_model(base_model, hub_model, use_pipeline)

    text = "CIA and FBI can track anyone, and you willingly give the data away"
    filter = True

    result = text_prediction(model_wrapper, text, use_pipeline, filter)

    ind = np.argwhere(np.round(result.weights[0, :] * 100, 2) > 0)
    significant = result.means[0, ind].reshape(-1, 2)
    weights = result.weights[0, ind].flatten()

    sig_weights = weights[weights > 0]

    print(f"RESULT\t{len(sig_weights)} significant prediction outcome(s):")
    for i in range(len(sig_weights)):
        point = f"lon: {'  lat: '.join(map(str, significant[i]))}"
        weight = str(np.round(sig_weights[i] * 100, 2))
        print(f"\tOut {i + 1}\t{weight}%\t-\t{point}")

    # visual = ResultVisuals(result)
    # visual.text_map_result()


if __name__ == "__main__":
    main()