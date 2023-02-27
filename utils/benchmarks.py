import math
import torch
import torch.distributions as dist
import numpy as np
from utils.regressor import *

# model benchmarks for loss and metrics


# R2 score
def r2_score(X, Y):
    labels_mean = torch.mean(Y)
    ss_tot = torch.sum((Y - labels_mean) ** 2)
    ss_res = torch.sum((Y - X) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


# spher cov from raw outputs
def spher_sigma(X, outcomes, outputs_map, lower_limit=0):
    softplus = nn.Softplus()
    S = softplus(X[:, outputs_map["sigma"][0]:outputs_map["sigma"][1]]) + lower_limit
    return S.reshape([X.size(dim=0), outcomes])


# GM/GMM from raw outputs
def GaussianModel(X, outcomes, outputs_map, prob_domain, cov):
    softplus = nn.Softplus()
    batch = X.size(dim=0)

    means = X[:, outputs_map["coord"][0]:outputs_map["coord"][1]]
    if outcomes > 1:
        means = means.reshape([batch, outcomes, 2])

    sigma_lower_limit = 1 / (2 * math.pi) if prob_domain == "pos" else 0
    positive_sigma = spher_sigma(X, outcomes, outputs_map, sigma_lower_limit)

    sigma = None
    tril = None

    if cov == "spher":
        sigma = torch.eye(2, device=X.device) * positive_sigma.reshape(-1, 1)[:, None]
    elif cov == "diag":
        sigma = torch.eye(2, device=X.device) * positive_sigma.reshape(-1, 2)[:, None]
    else:
        tril_indices = torch.tril_indices(row=2, col=2, offset=0, device=X.device)
        if cov == "tied" or outcomes == 1:
            tril = torch.zeros((2, 2), device=X.device).repeat(batch, 1).reshape([batch, 2, 2])
            tril[:, tril_indices[0], tril_indices[1]] = positive_sigma.reshape([batch, 3])
        else:
            tril = torch.zeros((2, 2), device=X.device).repeat(batch, outcomes).reshape([batch, outcomes, 2, 2])
            tril[:, :, tril_indices[0], tril_indices[1]] = positive_sigma.reshape([batch, outcomes, 3])

    if sigma is not None:
        if outcomes > 1:
            sigma = sigma.reshape([batch, outcomes, 2, 2])
        gaussian = dist.MultivariateNormal(means, sigma)
    else:
        if outcomes > 1 and cov == "tied":
            tril = tril.reshape(batch, -1).repeat(1, outcomes).reshape([batch, outcomes, 2, 2])
        gaussian = dist.MultivariateNormal(means, scale_tril=tril)

    return gaussian


# GMM weights from raw outputs
def GaussianWeights(X, outcomes, outputs_map):
    softmax = nn.Softmax(dim=1)
    weights = X[:, outputs_map["weight"][0]:outputs_map["weight"][1]].reshape([X.size(dim=0), outcomes]) if outputs_map["weight"] else torch.ones((X.size(dim=0), outcomes), device=X.device)
    gmm_weights = dist.Categorical(softmax(weights))
    return gmm_weights


# spatial weights from raw outputs
def weights(X, outcomes, outputs_map):
    softmax = nn.Softmax(dim=1)
    W = X[:, outputs_map["weight"][0]:outputs_map["weight"][1]].reshape([X.size(dim=0), outcomes]) if outputs_map["weight"] else torch.ones((X.size(dim=0), outcomes), device=X.device)
    return softmax(W)


# Distance^2 to genuine truth from raw outputs
def dist2(X, y, outcomes, outputs_map):
    def coord_se(X, y, outcomes):
        error_by_coord = nn.MSELoss(reduction='none')
        Y = y.repeat(1, outcomes)
        E = error_by_coord(X, Y)
        return E

    E = coord_se(X[:, outputs_map["coord"][0]:outputs_map["coord"][1]], y, outcomes)
    D = torch.zeros((X.size(dim=0), 0), device=X.device)
    for i in range(outputs_map["coord"][0], outputs_map["coord"][1], 2):
        D = torch.cat((D, torch.sum(E[:, i:i+2], dim=1, keepdim=True)), 1)
    return D


# Negative Log Likelihood fit for genuine truth from raw outputs
def lh_loss(X, y, outcomes, outputs_map, prob_domain):
    gaussian = GaussianModel(X, outcomes, outputs_map, prob_domain, "spher")
    if outcomes > 1:
        gmm_weights = GaussianWeights(X, outcomes, outputs_map)
        gmm = dist.MixtureSameFamily(gmm_weights, gaussian)
        L = -gmm.log_prob(y)
    else:
        L = -gaussian.log_prob(y)
    return L


# weighted D2 loss from raw outputs
def d_loss(X, y, outcomes, outputs_map):
    D = dist2(X, y, outcomes, outputs_map)
    if outcomes > 1:
        W = weights(X, outcomes, outputs_map)
        L = torch.sum(D * W, dim=1)
    else:
        L = D
    return L


class ModelBenchmark():
    def __init__(self, model, distance=True, loss_prob="pos", mf_loss="mean", total_loss="type"):
        self.model = model
        self.dist = distance
        self.prob_domain = loss_prob
        self.mf_handle = mf_loss
        self.total_loss_crit = total_loss

        self.outcomes = self.model.n_outcomes
        self.cov = self.model.cov
        self.weighted = self.model.weighted
        self.features = self.model.features

        self.outputs_map = {
            "coord": [0, self.model.coord_output],
            "weight": [self.model.coord_output, self.model.coord_output + self.model.weights_output] if self.weighted else None,
            "sigma": [self.model.coord_output + self.model.weights_output, self.model.coord_output + self.model.weights_output + self.model.cov_output] if self.cov else None
        }

        self.single_outputs_map = {
            "coord": [0, 2],
            "weight": None,
            "sigma": [2, 3] if self.cov else None
        }

        self.loss_type = 1 if self.outputs_map["sigma"] else 0

        print(f"TRAIN\tLOSS\tKey Feature - {'sum of spat and prob' if self.total_loss_crit == 'sum' else 'by model type'}:\n"
              f"\tGeospatial accuracy:\t{'weighted ' if self.weighted else ''}{'distance' if self.dist else 'coord'} error^2 for {self.outcomes} outcome(s)")

        if self.outputs_map["sigma"] is not None:
            print(f"\tProbability accuracy:\t {'limited' if self.prob_domain == 'pos' else 'unlimited'} -LLH for PDF of {'weighted ' if self.weighted else ''}"
                  f"{'GM' if self.outcomes == 1 else 'GMM'} with {self.cov} covariance matrix ")

        if len(self.features) > 1:
            print(f"TRAIN\tLOSS\tMinor Features - {self.mf_handle} of:\n\tGeospatial accuracy:\tsingle {'distance' if self.dist else 'coord'} error^2")
            if self.outputs_map["sigma"]:
                print(f"\tProbability accuracy:\t {'limited' if self.prob_domain == 'pos' else 'unlimited'} -LLH for PDF of single GM with spher covariance matrix ")

    def minor_feature_loss(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()

        if X.dim() == 1:
            X = X.reshape(1, -1)

        spat_loss = d_loss(X, y, 1, self.single_outputs_map).mean()
        prob_loss = torch.zeros_like(spat_loss, device=X.device)
        if self.outputs_map["sigma"]:
            prob_loss = lh_loss(X, y, 1, self.single_outputs_map, self.prob_domain).mean()
        return spat_loss, prob_loss

    def key_feature_loss(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()

        if X.dim() == 1:
            X = X.reshape(1, -1)

        spat_loss = d_loss(X, y, self.outcomes, self.outputs_map).mean()
        prob_loss = torch.zeros_like(spat_loss, device=X.device)
        if self.outputs_map["sigma"]:
            prob_loss = lh_loss(X, y, self.outcomes, self.outputs_map, self.prob_domain).mean()
        return spat_loss, prob_loss

    def total_batch_loss(self, batch_loss):
        all_features_loss = torch.mean(batch_loss, dim=0) if self.mf_handle == "mean" else torch.sum(batch_loss, dim=0)
        if self.total_loss_crit == "sum":
            total_loss = torch.sum(all_features_loss, dim=0)
        elif self.total_loss_crit == "mean":
            total_loss = torch.mean(all_features_loss, dim=0)
        elif self.total_loss_crit == "type":
            total_loss = all_features_loss[1 if self.outputs_map["sigma"] else 0]
        return total_loss

    # pytorch GM/GMM from raw outputs
    def prob_models(self, outputs):
        X = outputs.squeeze().float()

        if X.dim() == 1:
            X = X.reshape(1, -1)

        gaussian = GaussianModel(X, self.outcomes, self.outputs_map, self.prob_domain, self.cov)
        if self.outcomes > 1:
            gmm_weights = GaussianWeights(X, self.outcomes, self.outputs_map)
            return dist.MixtureSameFamily(gmm_weights, gaussian)
        else:
            return gaussian

    # spat and prob loss from raw outputs
    def result_metrics(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()

        if X.dim() == 1:
            X = X.reshape(1, -1)

        spat_loss = d_loss(X, y, self.outcomes, self.outputs_map).reshape(-1, 1)
        prob_loss = torch.zeros_like(spat_loss, device=X.device)
        if self.outputs_map["sigma"]:
            prob_loss = lh_loss(X, y, 1, self.single_outputs_map, self.prob_domain).reshape(-1, 1)

        return spat_loss, prob_loss

    def r2(self, outputs, labels):
        if outputs.dim() == 1:
            outputs = outputs.reshape(1, -1)

        Y = labels.repeat(1, self.outcomes) if self.outcomes > 1 else labels
        X = outputs[:, self.outputs_map["coord"][0]:self.outputs_map["coord"][1]]
        r2 = r2_score(X, Y)
        return r2

    # tensorboard metrics logging
    def log(self, writer, step, lr, train_metric, cur_batch, val_metric=None):
        def total_loss_log(metric, metric_type="total"):
            if metric_type == "val":
                atf_loss = metric[:, 0]
                folder = f"mean_val"
            else:
                atf_loss = np.mean(metric[:, :, 0], axis=0) if self.mf_handle == "mean" else np.sum(metric[:, :, 0], axis=0)
                folder = f"current_step" if metric_type != "total" else f"mean_train"

            if self.total_loss_crit == "sum":
                total_loss = np.sum(atf_loss, axis=0)
            elif self.total_loss_crit == "mean":
                total_loss = np.mean(atf_loss, axis=0)
            elif self.total_loss_crit == "type":
                total_loss = atf_loss[1 if self.outputs_map["sigma"] else 0]

            log = f"\tTotal loss of all features:\t{total_loss}"
            print(log)
            writer.add_scalar(f"{folder}/total_loss", total_loss, step)

            if metric_type == "total":
                self.mean_epoch_train_loss = total_loss

        def spat_loss_log(metric, metric_type="total"):
            if metric_type == "val":
                spat_loss, r2 = metric[0, 0], metric[0, 1]
                folder, log = f"mean_val", f"\tGeospatial {spatial} loss:\t{spat_loss}\tCoord R2:\t{r2}"
            else:
                spat_loss, r2 = np.mean(metric[:, 0, 0], axis=0) if self.mf_handle == "mean" else np.sum(metric[:, 0, 0], axis=0), None
                folder, log = f"current_step", f"\tGeospatial {self.mf_handle} {spatial} loss:\t{spat_loss}"
                if metric_type == "total":
                    r2 = metric[0, 0, 1]
                    folder = f"mean_train"
                    log += f"\tCoord R2:\t{r2}"

                if metric.shape[0] > 1:
                    key_spat, minor_spat = metric[0, 0, 0], np.mean(metric[1:, 0, 0], axis=0)
                    log += f"\n\t\tKey:\t{key_spat}\tMinor:\t{minor_spat}"
                    writer.add_scalar(f"{folder}/spat_key", key_spat, step)
                    writer.add_scalar(f"{folder}/spat_minor", minor_spat, step)

            print(log)
            writer.add_scalar(f"{folder}/loss_spat", spat_loss, step)
            if r2:
                writer.add_scalar(f"{folder}/r2", r2, step)

        def prob_loss_log(metric, metric_type="total"):
            if metric_type == "val":
                prob_loss, pdf = metric[1, 0], metric[1, 1]
                folder, log = f"mean_val", f"\tProbabilistic {self.mf_handle} -LLH loss:\t{prob_loss}\tPDF:\t{pdf}"
            else:
                prob_loss, pdf = np.mean(metric[:, 1, 0], axis=0) if self.mf_handle == "mean" else np.sum(metric[:, 1, 0], axis=0), None
                folder, log = f"current_step", f"\tProbabilistic {self.mf_handle} -LLH loss:\t{prob_loss}"
                if metric_type == "total":
                    pdf = metric[0, 1, 1]
                    folder = f"mean_train"
                    log += f"\tPDF:\t{pdf}"

                if metric.shape[0] > 1:
                    key_prob, minor_prob = metric[0, 1, 0], np.mean(metric[1:, 1, 0], axis=0)
                    log += f"\n\t\tKey:\t{key_prob}\tMinor:\t{minor_prob}"
                    writer.add_scalar(f"{folder}/prob_key", key_prob, step)
                    writer.add_scalar(f"{folder}/prob_minor", minor_prob, step)

            print(log)
            writer.add_scalar(f"{folder}/loss_prob", prob_loss, step)
            if pdf:
                writer.add_scalar(f"{folder}/pdf", pdf, step)

        spatial = 'D^2' if self.dist else 'Coord'
        processed_metric = train_metric[0:cur_batch, :]
        mean_metric = np.mean(processed_metric, axis=0)
        current_batch_metric = processed_metric[-1, :]
        # print(current_batch_metric)

        print(f"LOG\tCurrent step: {step}\tLR:\t{lr}")
        total_loss_log(current_batch_metric, "current")
        spat_loss_log(current_batch_metric, "current")
        if self.outputs_map["sigma"]:
            prob_loss_log(current_batch_metric, "current")

        print(f"LOG\tTRAIN\tMean metrics:")
        total_loss_log(mean_metric, "total")
        spat_loss_log(mean_metric, "total")
        if self.outputs_map["sigma"]:
            prob_loss_log(mean_metric, "total")

        if val_metric is not None:
            mean_val_metric = np.mean(val_metric, axis=0)

            print(f"LOG\tVAL\tMean metrics:")
            total_loss_log(mean_val_metric, "val")
            spat_loss_log(mean_val_metric, "val")
            if self.outputs_map["sigma"]:
                prob_loss_log(mean_val_metric, "val")

        writer.flush()
