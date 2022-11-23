import pandas as pd
import numpy as np
from geopy import distance
import geopy.distance
from datetime import datetime
import scipy.sparse as sparse
import haversine as hs

from utils.regressor import *
from utils.benchmarks import *

# evaluation results manager


# GM/GMM
def GaussianModel(means, sigma):
    return dist.MultivariateNormal(torch.from_numpy(means), torch.from_numpy(sigma))


# GMM weights
def GaussianWeights(weights):
    return dist.Categorical(torch.from_numpy(weights))


# GMM complete
def get_gm_family(outcomes, means, sigma, weights):
    means, sigma = means.reshape(outcomes, 2), sigma.reshape(outcomes, 2, 2)
    gaussian = GaussianModel(means, sigma)
    if weights is not None:
        gmm_weights = GaussianWeights(weights.reshape(-1))
        gm = dist.MixtureSameFamily(gmm_weights, gaussian)
    else:
        gm = gaussian
    return gm


# Haversine distance to the genuine truth in km for PRA metrics (outliers set to map borders)
def metric_distance(true, points, size):
    D = np.array([], dtype=float)
    longs = points[:, 0]
    lats = points[:, 1]
    longs[longs > 180] = 180
    longs[longs < -180] = -180
    lats[lats > 90] = 90
    lats[lats < -90] = -90
    for i in range(size):
        d = geopy.distance.distance((true[i, 1], true[i, 0]),
                                    (lats[i], longs[i])).km
        D = np.append(D, d)
    return D


# spatial metrics: Average/Median Distance Error (AED, MED), MSE, MAE, Acc@161
def geospatial_performance(true, dists, means, weights, outcomes, size, best, threshold=100):
    print(f"Calculating spatial metrics for {size} samples")
    if best and outcomes > 1:
        dists, means = dists[:, 0], means[:, 0]
    else:
        if outcomes > 1:
            dists = np.sum(dists * weights, axis=1)
            mse, mae = np.zeros((size, outcomes)), np.zeros((size, outcomes))

    aed, med = np.mean(dists), np.median(dists)
    acc = (dists < threshold).sum() / size * 100
    acc161 = (dists < 161).sum() / size * 100

    if best or outcomes == 1:
        mse = np.mean((true - means)**2)
        mae = np.sum(np.abs(true - means), axis=1).mean()
    elif not best and outcomes > 1:
        for k in range(outcomes):
            mse[:, k] = np.mean(np.power(true - means[:, k], 2), axis=1)
            mae[:, k] = np.mean(np.abs(true - means[:, k]), axis=1)

        mse = np.sum(mse * weights, axis=1).mean()
        mae = np.sum(mae * weights, axis=1).mean()

    return (aed, med, acc, acc161, mse, mae)


# probabilistic metrics: Average/Median Comprehensive Accuracy Error (ACAE, MCAE), Average/Median 95% Prediction Region Area (APRA, MPRA), 95%Coverage (COV)
def probabilistic_performance(trues, covs, means, weights, outcomes, size, best, n=100):
    print(f"Calculating probabilistic metrics for {size} samples")
    if best and outcomes > 1:
        covs, means = covs[:, 0], means[:, 0]

    cae, pra, cov = np.zeros((size, outcomes)), \
                    np.zeros((size, outcomes)), \
                    np.zeros((size, outcomes))

    crit_chi = 5.991  # 0.95 2

    rng = np.random.default_rng()
    if not best and outcomes > 1:
        for i in range(size):
            for k in range(outcomes):
                mean, covar, true, weight = means[i, k], covs[i, k], trues[i], weights[i, k]
                gaus_sample = rng.multivariate_normal(mean, covar, n)
                rep_true = np.repeat(true.reshape(1, 2), n, axis=0).reshape(n, 2)
                gaus_dist = metric_distance(rep_true, gaus_sample, n)

                cae[i, k] = np.mean(gaus_dist, axis=0) * weight

                sigma = np.sqrt(covar[0, 0])
                error = np.sqrt(np.sum((rep_true - mean)**2))
                pra[i, k] = np.pi * sigma * crit_chi * weight
                cov[i, k] = 1 if error/sigma <= crit_chi else 0

        pra, cae = np.sum(pra, axis=1), np.sum(cae, axis=1)

    else:
        for i in range(size):
            mean, covar, true = means[i], covs[i], trues[i]

            gaus_sample = rng.multivariate_normal(mean, covar, n)
            rep_true = np.repeat(true.reshape(1, 2), n, axis=0).reshape(n, 2)
            gaus_dist = metric_distance(rep_true, gaus_sample, n)

            cae[i] = np.mean(gaus_dist, axis=0)

            sigma = covar[0, 0]
            error = np.sum((rep_true - mean)**2)

            pra[i] = np.pi * sigma * crit_chi
            cov[i] = 1 if error/sigma <= crit_chi else 0

    acae, mcae, apra, mpra, cov = np.mean(cae), np.median(cae), \
                                  np.mean(pra), np.median(pra), \
                                  cov.mean()

    return (acae, mcae, apra, mpra, cov)


class ResultManager():
    def __init__(self, val_df, text, feature, device, model_benchmark, scaled, by_user=False, prefix=None):
        self.cluster = device.type == "cuda"
        self.feature = feature
        if prefix is None:
            prefix = feature
        self.prefix = prefix

        self.model_bm = model_benchmark

        self.scaled = scaled
        self.by_user = by_user

        self.outcomes = self.model_bm.outcomes
        self.cov = self.model_bm.cov
        self.weighted = self.model_bm.weighted
        self.dist = self.model_bm.dist

        self.outputs_map = self.model_bm.outputs_map

        self.prob = self.cov is not None

        self.covariances = {'spher': 1,
                            'diag': 2,
                            'tied': 3,
                            'full': 3}

        self.pred_columns = {}
        for i in range(self.outcomes):
            self.pred_columns[f"O{i+1}_point"] = "str"
            self.pred_columns[f"O{i+1}_dist"] = "float"
            if self.outcomes > 1:
                    self.pred_columns[f"O{i+1}_weight"] = "float"
            if self.prob:
                self.pred_columns[f"O{i+1}_sigma"] = "str"

        self.text = text
        if self.text is None:
            print(f"RESULT\tInitializing dataframe with {len(self.pred_columns)} columns for {self.outcomes} outcome(s)")
            print(f"column tag:\ttype")
            for key, value in self.pred_columns.items():
                print(f"{key}:\t{value}")

        if val_df is not None:
            val_df.reset_index(drop=True, inplace=True)
            self.df = val_df
            self.true = val_df[["lon", "lat"]].to_numpy()
            self.size = len(self.df.index)
            if self.by_user:
                self.users = len(self.df['USER-ONLY'].unique())
        else:
            self.df = pd.DataFrame()
            self.size = 1
            self.true = np.array([])
            self.dists = None

    # load evaluation results df to class entities
    def load_df(self, filename, sorting=True):
        print(f"VAL\tLOAD\tLoading dataset from {filename}")
        self.df = pd.read_json(path_or_buf=filename, lines=True)
        self.size = len(self.df.index)
        if self.by_user:
            self.users = len(self.df['USER-ONLY'].unique())
        self.true = self.df[["lon", "lat"]].to_numpy()

        if self.outcomes > 1:
            means = np.empty((0, self.outcomes, 2), float)
            dists = np.empty((0, self.outcomes))
            weights = np.empty((0, self.outcomes))
            if self.prob:
                covs = np.empty((0, self.outcomes, 2, 2), float)
        else:
            means = np.empty((0, 2), float)
            dists = np.empty((0, 1), float)
            if self.prob:
                covs = np.empty((0, 2, 2), float)

        for i in range(self.size):
            if self.outcomes > 1:
                mean_row = np.empty((0, 2), float)
                dists_row = np.empty((0, 1), float)
                weight_row = np.empty((0, 1), float)
                if self.prob:
                    cov_row = np.empty((0, 2, 2), float)

                for o in range(self.outcomes):
                    mean_row = np.append(mean_row, np.array(self.df.loc[i, f"O{o+1}_point"]).reshape(-1, 2), axis=0)
                    dists_row = np.append(dists_row, np.array(self.df.loc[i, f"O{o+1}_dist"]).reshape(-1, 1), axis=0)
                    weight_row = np.append(weight_row, self.df.loc[i, f"O{o+1}_weight"].reshape(-1, 1), axis=0)
                    if self.prob:
                        cov_row = np.append(cov_row, np.array(self.df.loc[i, f"O{o+1}_sigma"]).reshape(-1, 2, 2), axis=0)

                means = np.append(means, mean_row.reshape(-1, self.outcomes, 2), axis=0)
                dists = np.append(dists, dists_row.reshape(-1, self.outcomes), axis=0)
                weights = np.append(weights, weight_row.reshape(-1, self.outcomes), axis=0)
                if self.prob:
                    covs = np.append(covs, cov_row.reshape(-1, self.outcomes, 2, 2), axis=0)
            else:
                means = np.append(means, np.array(self.df.loc[i, "O1_point"]).reshape(-1, 2), axis=0)
                dists = np.append(dists, np.array(self.df.loc[i, "O1_dist"]).reshape(-1, 1), axis=0)
                if self.prob:
                    covs = np.append(covs, np.array(self.df.loc[i, "O1_sigma"]).reshape(-1, 2, 2), axis=0)

        self.means = means
        self.dists = dists
        self.weights = weights if self.outcomes > 1 else None
        self.covs = covs if self.prob else None

        print(f"VAL\tLOAD\tDataset of {self.size} samples is loaded")
        if sorting and self.outcomes > 1:
            self.sort_outcomes()

        if self.outcomes == 1:
            self.df["O1_point"] = sparse.coo_matrix(self.means, shape=(self.size, 2)).toarray().tolist()
            self.df["O1_dist"] = self.dists
            if self.prob:
                self.df["O1_sigma"] = sparse.coo_matrix(self.covs.reshape((self.size, 4)), shape=(self.size, 4)).toarray().tolist()
        else:
            for i in range(self.outcomes):
                self.df[f"O{i+1}_point"] = sparse.coo_matrix(self.means[:, i, :], shape=(self.size, 2)).toarray().tolist()
                self.df[f"O{i+1}_dist"] = self.dists[:, i]
                if self.prob:
                    self.df[f"O{i+1}_sigma"] = sparse.coo_matrix(self.covs[:, i, :].reshape((self.size, 4)), shape=(self.size, 4)).toarray().tolist()
                    self.df[f"O{i+1}_weight"] = self.weights[:, i]

    def save_df(self, prefix=None, filename=None):
        if prefix is None:
            prefix = self.prefix
        if filename is None:
            filename = f"results/val-data/{prefix}_predicted_N{self.size}_VF-{self.feature}_{datetime.today().strftime('%Y-%m-%d')}.jsonl"

        with open(filename, "w") as f:
            self.df.to_json(f, orient='records', lines=True)
        print(f"VAL\tSAVE\tPredicted data of {self.size} samples is written to file: {filename}")

    # add loss metrics to df
    def metrics(self, val_metric):
        if self.prob:
            metrics_df = pd.DataFrame(
                {f'{"dist" if self.dist else "coord"}_loss': val_metric[:, 0, 0].reshape(-1),
                 f'lh_loss': val_metric[:, 1, 0].reshape(-1),
                 'pdf': val_metric[:, 1, 1].reshape(-1)})
        else:
            metrics_df = pd.DataFrame({f'{"dist" if self.dist else "coord"}_loss': val_metric[:, 0, 0].reshape(-1)})

        print(f"RESULT\tAdding metrics column(s) {', '.join(str(col) for col in metrics_df.columns.values.tolist())} to dataframe")
        self.df = pd.concat([self.df, metrics_df], axis=1)

    # results spatial metrics
    def spatial_metric(self, threshold=100, best=True):
        if self.by_user:
            users = self.df['USER-ONLY'].unique()
            metric = np.zeros(6, dtype=float)
            for user in users:
                ids = self.df.index[self.df['USER-ONLY'] == user].tolist()
                user_metric = geospatial_performance(self.true[ids],
                                                     self.dists[ids],
                                                     self.means[ids],
                                                     self.weights[ids],
                                                     self.outcomes,
                                                     len(ids),
                                                     best, threshold)

                metric += np.array(user_metric)

            metric = metric / len(users)
            aed, med, acc, acc161, mse, mae = metric

        else:
            aed, med, acc, acc161, mse, mae = geospatial_performance(self.true,
                                                                     self.dists,
                                                                     self.means,
                                                                     self.weights,
                                                                     self.outcomes,
                                                                     self.size,
                                                                     best, threshold)

        loss_dist = np.array(self.df["dist_loss"])
        loss_dist = loss_dist * 10000 if self.scaled else loss_dist

        print(f"Spatial metrics {'best outcome' if best else ''}{'weighted outcomes' if self.outcomes >1 and not best else ''}:"
              f"\n\tAED: {round(aed, 2)} km\t- avg error\n\tMED: {round(med, 2)} km\t- median error"
              f"\n\tMSE: {round(mse, 2)}\t- mean error^2 (degrees)\n\tMAE: {round(mae, 2)}\t- mean abs error (degrees)"
              f"\n\tLoss D^2:\t{round(loss_dist.mean(), 2)}\t- avg loss degrees")
        print(f"\tAccuracy (<100km): {round(acc, 2)}%\t- below threshold"
              f"\n\tAccuracy (<161km): {round(acc161, 2)}%\t- below threshold")

        return aed, med, mse, mae, acc, acc161

    # results probabilistic metrics
    def prob_metric(self, best=True, n=100):
        if self.by_user:
            users = self.df['USER-ONLY'].unique()
            metric = np.zeros(5, dtype=float)
            for user in users:
                ids = self.df.index[self.df['USER-ONLY'] == user].tolist()
                user_metric = probabilistic_performance(self.true[ids],
                                                         self.covs[ids],
                                                         self.means[ids],
                                                         self.weights[ids],
                                                         self.outcomes,
                                                         len(ids),
                                                         best, n)

                metric += np.array(user_metric)

            metric = metric / len(users)
            acae, mcae, apra, mpra, cov = metric

        else:
            acae, mcae, apra, mpra, cov = probabilistic_performance(self.true,
                                                 self.covs,
                                                 self.means,
                                                 self.weights,
                                                 self.outcomes,
                                                 self.size,
                                                 best, n)

        loss_llh = np.array(self.df["lh_loss"]).mean()
        pdf = np.array(self.df["pdf"]).mean()

        print(f"GMM metrics {'best outcome' if best else ''}{'weighted outcomes' if self.outcomes > 1 and not best else ''}:"
              f"\n\tACAE: {round(acae, 2)} km\t- avg GMM sample error\n\tMCAE: {round(mcae, 2)} km\t- median GMM sample error"
              f"\n\tAPRA (0.95): {round(apra, 2)} km2\t- avg GMM area\n\tMPRA (0.95): {round(mpra, 2)} km2\t- median GMM area"
              f"\n\tNLLH loss:\t{round(loss_llh, 2)}\t- avg GMM neg log-likelihood loss"
              f"\n\tPDF:\t{round(pdf*100, 2)}%\t- avg GMM fit likelihood"
              f"\n\tCOV (0.95): {round(cov*100, 2)}%\t- GMM coverage")

        return acae, mcae, apra, mpra, cov

    # all results metrics
    def result_metrics(self, best=True, threshold=100):
        print(f"Calculating spatial {'and probabilistic ' if self.prob else ''}metrics "
              f"for {self.size} result samples {'per user' if self.by_user else 'per tweet'}")

        aed, med, mse, mae, acc, acc161 = self.spatial_metric(threshold, best)
        spat_metric = [["Average SAE", aed],
                      ["Median SAE", med],
                      ["MSE", mse],
                      ["MAE", mae],
                      [f"Acc@{threshold}", acc],
                      ["Acc@161", acc161]]

        if self.prob:
            acae, mcae, apra, mpra, cov = self.prob_metric(best)
            prob_metric = [["Average CAE", acae],
                           ["Median CAE", mcae],
                           ["Average 95% PRA", apra],
                           ["Median 95% PRA", mpra],
                           ["PRA COVerage", cov]]
        else:
            prob_metric = []

        out = "BEST" if best else f"ALL {self.outcomes}"
        return [["Outcome", out]] + spat_metric + prob_metric

    # full performance measure of the model
    def performance(self, save=True):
        best_metric = self.result_metrics(True)
        self.performance_df = pd.DataFrame(best_metric, columns=["metric", "value"])

        if self.outcomes > 1:
            all_metric = self.result_metrics(False)
            self.performance_df = self.performance_df.append(pd.DataFrame(all_metric, columns=["metric", "value"]), ignore_index=True)

        self.performance_df['metric'] = self.performance_df['metric'].apply(lambda x: "{:<20}".format(x))

        if save:
            filename = f"results/metric/{self.prefix}_metric_N{self.size}_VF-{self.feature}_{datetime.today().strftime('%Y-%m-%d')}.txt"

            with open(filename, "w") as f:
                self.performance_df.to_csv(f, header=False, index=False, sep="\t", mode="a")
            print(f"VAL\tSAVE\tPerformance metrics of {self.size} samples are written to file: {filename}")

    # sort per-tweet outcomes by their weights
    def sort_outcomes(self):
        if self.dists is None and self.true.size > 0:
            self.dists = self.distances(self.means)

        print(f"RESULT\tSorting all outputs for {self.outcomes} outcomes by probabilistic weights")

        if self.text:
            sort_indexes = np.argsort(self.weights[0])
            index = sort_indexes[::-1]
            self.means[0] = self.means[0, index]
            if self.outcomes > 1:
                self.weights[0] = self.weights[0, index]
            if self.prob:
                self.covs[0] = self.covs[0, index]

        else:
            sort_indexes = np.argsort(self.weights, axis=1)

            for i in range(self.size):
                index = sort_indexes[i][::-1]
                self.means[i, :] = self.means[i, index]
                self.weights[i, :] = self.weights[i, index]
                if self.dists is not None:
                    self.dists[i, :] = self.dists[i, index]
                if self.prob:
                    self.covs[i, :] = self.covs[i, index]

    # add to df plain spat outputs
    def coord_outputs(self, predicted):
        self.means = np.multiply(predicted[:, self.outputs_map["coord"][0]:self.outputs_map["coord"][1]], 100) if self.scaled else predicted[:, self.outputs_map["coord"][0]:self.outputs_map["coord"][1]]
        self.means = self.means.reshape(self.size, 2) if self.outcomes == 1 else self.means.reshape(self.size, self.outcomes, 2)

        if self.outputs_map["weight"]:
            self.weights = weights(torch.from_numpy(predicted), self.outcomes, self.outputs_map).numpy()

        if self.true.size > 0:
            self.dists = self.distances(self.means)

        if self.outcomes > 1:
            #print(f"RESULT\tSorting all outputs for {self.outcomes} outcomes by distances error")
            self.sort_outcomes()

        if self.df.size > 0:
            spat_df = pd.DataFrame({column: pd.Series(dtype=type) for column, type in self.pred_columns.items()})

            if self.outcomes == 1:
                spat_df["O1_point"] = sparse.coo_matrix(self.means, shape=(self.size, 2)).toarray().tolist()
                spat_df["O1_dist"] = self.dists
            else:
                for i in range(self.outcomes):
                    spat_df[f"O{i+1}_weight"] = self.weights[:, i]
                    spat_df[f"O{i+1}_point"] = sparse.coo_matrix(self.means[:, i, :], shape=(self.size, 2)).toarray().tolist()
                    spat_df[f"O{i+1}_dist"] = self.dists[:, i]

            print(f"RESULT\tSetting spatial output columns {', '.join(str(col) for col in spat_df.columns.values.tolist())} in dataframe")
            self.df = pd.concat([self.df, spat_df], axis=1)

    # add to df prob outputs (reading GM/GMM models)
    def soft_outputs(self, pm):
        self.prob_models = pm

        if self.outcomes > 1:
            means = np.empty((0, self.outcomes, 2), float)
            covs = np.empty((0, self.outcomes, 2, 2), float)
            weights = np.empty((0, self.outcomes))
        else:
            means = np.empty((0, 2), float)
            covs = np.empty((0, 2, 2), float)

        for models in self.prob_models:
            if self.outcomes > 1:
                if self.cluster:
                    means = np.append(means, models.component_distribution.loc.cpu().numpy(), axis=0)
                    covs = np.append(covs, models.component_distribution.covariance_matrix.cpu().numpy(), axis=0)
                    weights = np.append(weights, models.mixture_distribution.probs.cpu().numpy(), axis=0)
                else:
                    means = np.append(means, models.component_distribution.loc.numpy(), axis=0)
                    covs = np.append(covs, models.component_distribution.covariance_matrix.numpy(), axis=0)
                    weights = np.append(weights, models.mixture_distribution.probs.numpy(), axis=0)
            else:
                if self.cluster:
                    means = np.append(means, models.loc.cpu().numpy().reshape(-1, 2), axis=0)
                    covs = np.append(covs, models.covariance_matrix.cpu().numpy().reshape(-1, 2, 2), axis=0)
                else:
                    means = np.append(means, models.loc.numpy().reshape(-1, 2), axis=0)
                    covs = np.append(covs, models.covariance_matrix.numpy().reshape(-1, 2, 2), axis=0)

        self.means = np.multiply(means, 100) if self.scaled else means
        self.covs = np.multiply(covs, 100) if self.scaled else covs

        if self.true.size > 0:
            self.dists = self.distances(self.means)

        if self.outcomes > 1:
            self.weights = weights
            self.sort_outcomes()

        if self.df.size > 0:
            prob_models_df = pd.DataFrame({column: pd.Series(dtype=type) for column, type in self.pred_columns.items()})

            if self.outcomes == 1:
                prob_models_df["O1_point"] = sparse.coo_matrix(self.means, shape=(self.size, 2)).toarray().tolist()
                prob_models_df["O1_sigma"] = sparse.coo_matrix(self.covs.reshape((self.size, 4)), shape=(self.size, 4)).toarray().tolist()
                prob_models_df["O1_dist"] = self.dists
            else:
                for i in range(self.outcomes):
                    prob_models_df[f"O{i+1}_point"] = sparse.coo_matrix(self.means[:, i, :], shape=(self.size, 2)).toarray().tolist()
                    prob_models_df[f"O{i+1}_sigma"] = sparse.coo_matrix(self.covs[:, i, :].reshape((self.size, 4)), shape=(self.size, 4)).toarray().tolist()
                    prob_models_df[f"O{i+1}_weight"] = self.weights[:, i]
                    prob_models_df[f"O{i+1}_dist"] = self.dists[:, i]

            print(f"RESULT\tSetting spatial and probabilistic output columns {', '.join(str(col) for col in prob_models_df.columns.values.tolist())} in dataframe")
            self.df = pd.concat([self.df, prob_models_df], axis=1)

    # Haversine distances to the genuine truth in km for coords
    def distances(self, points):
        print(f"RESULT\tCalculating distances of {self.size} samples with {self.outcomes} outcome(s)")
        if self.outcomes == 1:
            D = np.array([], dtype=float)
            for i in range(self.size):
                d = geopy.distance.distance((self.true[i, 1], self.true[i, 0]),
                                            (points[i, 1], points[i, 0])).km
                D = np.append(D, d)
        else:
            D = np.empty((0, self.outcomes), dtype=float)
            for i in range(self.size):
                row = np.array([])
                for j in range(self.outcomes):
                    d = geopy.distance.distance((self.true[i, 1], self.true[i, 0]),
                                                (points[i, j, 1], points[i, j, 0])).km
                    row = np.append(row, d)
                D = np.append(D, row.reshape(1, self.outcomes), axis=0)
        return D

