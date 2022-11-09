import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
import torch.distributions as dist

from utils.regressor import *

def coord_se_matrix(X, y, outcomes):
    error_by_coord = nn.MSELoss(reduction='none')
    Y = y.repeat(1, outcomes)
    E = error_by_coord(X, Y)
    return E


def distance2_matrix(X, y, outcomes, outputs_map):
    E = coord_se_matrix(X[:, outputs_map["coord"][0]:outputs_map["coord"][1]], y, outcomes)
    D = torch.zeros((X.size(dim=0), 0), device=X.device)
    for i in range(outputs_map["coord"][0], outputs_map["coord"][1], 2):
        D = torch.cat((D, torch.sum(E[:, i:i+2], dim=1, keepdim=True)), 1)
    return D

def weights(X, outcomes, outputs_map):
    softmax = nn.Softmax(dim=1)
    W = X[:, outputs_map["weight"][0]:outputs_map["weight"][1]].reshape([X.size(dim=0), outcomes]) if outputs_map["weight"] else torch.ones((X.size(dim=0), outcomes), device=X.device)
    return softmax(W)

def spatial_loss(X, y, outcomes, outputs_map):
    if dist:
        E = distance2_matrix(X, y, outcomes, outputs_map)
    else:
        E = coord_se_matrix(X, y, outcomes)
        M = torch.zeros((X.size(dim=0), 0), device=X.device)
        for i in range(outputs_map["coord"][0], outputs_map["coord"][1], 2):
            M = torch.cat((M, E[:, i:i+2].mean(dim=1).reshape(X.size(dim=0), 1)), 1)
        E = M

    if outcomes > 1:
        W = weights(X, outcomes, outputs_map)
        L = torch.sum(E * W, 1, keepdim=True)
    else:
        L = E.mean(dim=1)
    return L

def log_likelihood_loss(X, y, outcomes, outputs_map, cov="spher"):
    gaussian = GaussianModel(X, outcomes, outputs_map, cov)
    if outcomes > 1:
        gmm_weights = GaussianWeights(X, outcomes, outputs_map)
        gmm = dist.MixtureSameFamily(gmm_weights, gaussian)
        lh = -gmm.log_prob(y)
    else:
        lh = -gaussian.log_prob(y)
    return lh

def coord_se_matrix(X, y, outcomes):
    error_by_coord = nn.MSELoss(reduction='none')
    Y = y.repeat(1, outcomes)
    E = error_by_coord(X, Y)
    return E

def distance2_matrix(X, y, batch, outcomes, coord_slice):
    E = coord_se_matrix(X, y, outcomes)
    D = torch.zeros((batch, 0), device=X.device)
    for i in range(0, coord_slice, 2):
        D = torch.cat((D, torch.sum(E[:, i:i+2], dim=1, keepdim=True)), 1)
    return D

def GaussianModel(X, batch, outcomes, cov, coord_slice, cov_slice):
    softplus = nn.Softplus()

    means = X[:, 0:coord_slice]
    if outcomes > 1:
        means = means.reshape([batch, outcomes, 2])

    positive_sigma = softplus(X[:, coord_slice:cov_slice])

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
            tril = tril[: None]
        else:
            tril = torch.zeros((2, 2), device=X.device).repeat(batch, outcomes).reshape([batch, outcomes, 2, 2])
            tril[:, :, tril_indices[0], tril_indices[1]] = positive_sigma.reshape([batch, outcomes, 3])

    if sigma is not None:
        if outcomes > 1:
            sigma = sigma.reshape([batch, outcomes, 2, 2])
        gaussian = dist.MultivariateNormal(means, sigma)
    else:
        gaussian = dist.MultivariateNormal(means, scale_tril=tril[:, None]) if cov == "tied" else dist.MultivariateNormal(means, scale_tril=tril)

    return gaussian

def GaussianWeights(X, batch, outcomes, weighted, cov_slice):
    softmax = nn.Softmax(dim=1)
    weights = X[:, cov_slice:].reshape([batch, outcomes]) if weighted else torch.ones((batch, outcomes), device=X.device)
    gmm_weights = dist.Categorical(softmax(weights))
    return gmm_weights

class Loss():
    def __init__(self, batch_size, model, distance=True, out="min"):
        self.batch = batch_size
        self.model = model
        self.dist = distance

        self.outcomes = self.model.n_outcomes
        self.cov = self.model.cov
        self.weighted = self.model.weighted

        self.out_column = "first" if self.outcomes == 1 else out

        # self.mean_coord = {'first': self.mc_first,
        #                     'min': self.mc_min,
        #                     'all': self.mc_all}
        #
        # self.mean_dist = {'first': self.md_first,
        #                    'min': self.md_min,
        #                    'all': self.md_all}
        #
        # self.log_lh_single = {'spher': self.lls_first,
        #                       'diag': self.lld_first,
        #                       'full': self.llf_first,
        #                       'tied': self.llf_first}
        #
        # self.log_lh_equal = {'spher': self.lls_equal,
        #                       'diag': self.lld_equal,
        #                       'full': self.llf_equal,
        #                       'tied': self.llt_equal}
        #
        # self.log_lh_mult = {'spher': self.lls_mult,
        #                       'diag': self.lld_mult,
        #                       'full': self.llf_mult,
        #                       'tied': self.llt_mult}

        self.cov_outputs = {'spher': self.outcomes,
                            'diag': self.outcomes * 2,
                            'tied': 3,
                            'full': self.outcomes * 3}

        self.coord_slice = self.outcomes*2

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.mse = nn.MSELoss()

        print(f"TRAIN\tLOSS\tDistance accuracy:\t {self.out_column} {'distance' if self.dist else 'coord'} error^2 for BATCHxOUTCOMES(s):\t {self.batch}x{self.outcomes}")
        self.spatial_loss = self.spatial_loss
        self.prob_loss = None

        if self.cov is not None:
            self.n_simga = self.cov_outputs[self.cov]
            self.cov_slice = self.coord_slice + self.n_simga
            print(f"TRAIN\tLOSS\tProbability accuracy:\t -log-likelihood for PDF of {'weighted ' if self.weighted else ''}{'GM' if self.outcomes == 1 else 'GMM'} with {self.cov} covariance matrix ")
            # if self.outcomes == 1:
            #     self.prob_loss = self.log_lh_single[self.cov]
            # else:
            #     self.prob_loss = self.log_lh_mult[self.cov] if self.weighted else self.log_lh_equal[self.cov]

            self.prob_loss = self.log_likelihood_loss


    def log_likelihood_loss(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        gaussian = GaussianModel(X, self.batch, self.outcomes, self.cov, self.coord_slice, self.cov_slice)
        if self.outcomes > 1:
            gmm_weights = GaussianWeights(X, self.batch, self.outcomes, self.weighted, self.cov_slice)
            gmm = dist.MixtureSameFamily(gmm_weights, gaussian)
            lh = -gmm.log_prob(y)
        else:
            lh = -gaussian.log_prob(y)

        return lh.mean()

    def spatial_loss(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()

        out = self.outcomes
        if self.out_column == "first":
            X = X[:, 0:2]
            out = 1

        E = distance2_matrix(X, y, self.batch, out, self.coord_slice) if self.dist else coord_se_matrix(X, y, out)

        if self.out_column == "min":
            if not self.dist:
                M = torch.zeros((self.batch, 0), device=X.device)
                for i in range(0, self.coord_slice, 2):
                    M = torch.cat((M, E[:, i:i+2].mean(dim=1).reshape(self.batch, 1)), 1)
                E = M
            L = torch.min(E, 1, keepdim=True).values
        else:
            L = E.mean(dim=1)
        return L.mean()

    # coord

    def mc_first(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        return self.mse(X[:, 0:2], y)

    def mc_min(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        loss_per_outcome = []
        for i in range(0, X.size(dim=1), 2):
            loss_per_outcome.append(self.mse(X[:, i:i+2], y))
        return min(loss_per_outcome)

    def mc_all(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        Y = y.repeat(1, X.size(dim=1)//2)
        return self.mse(X, Y)

    # dist

    def md_first(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        error_by_coord = nn.MSELoss(reduction='none')
        return torch.sum(error_by_coord(X[:, 0:2], y), dim=1, keepdim=True).mean()

    def md_min(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        return torch.min(distance2_matrix(X, y), 1, keepdim=True).values.mean()

    def md_all(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        return distance2_matrix(X, y).mean()

    # prob single

    def lls_first(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        sigma = torch.eye(2, device=X.device) * self.softplus(X[:, 2]).reshape(X.size(dim=0), 1)[:, None]
        return -dist.MultivariateNormal(X[:, 0:2], sigma).log_prob(y).mean()

    def lld_first(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        sigma = torch.eye(2, device=X.device) * self.softplus(X[:, 2:4]).reshape(X.size(dim=0), 2)[:, None]
        return -dist.MultivariateNormal(X[:, 0:2], sigma).log_prob(y).mean()

    def llf_first(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        tril = torch.zeros((2, 2), device=X.device).repeat(X.size(dim=0), 1).reshape([X.size(dim=0), 2, 2])
        tril_indices = torch.tril_indices(row=2, col=2, offset=0, device=X.device)
        tril[:, tril_indices[0], tril_indices[1]] = self.softplus(X[:, 2:5])[: None]
        return -dist.MultivariateNormal(X[:, 0:2], scale_tril=tril).log_prob(y).mean()

    # prob equal weights

    def lls_equal(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        mix_weights = dist.Categorical(torch.ones((X.size(dim=0), self.outcomes), device=X.device))

        means = X[:, 0:self.outcomes*2].reshape([X.size(dim=0), self.outcomes, 2])
        sigma = torch.eye(2, device=X.device) * self.softplus(X[:, self.outcomes*2:]).reshape(-1, 1)[:, None]
        mix_normals = dist.MultivariateNormal(means, sigma.reshape([X.size(dim=0), self.outcomes, 2, 2]))
        return -dist.MixtureSameFamily(mix_weights, mix_normals).log_prob(y).mean()

    def lld_equal(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        mix_weights = dist.Categorical(torch.ones((X.size(dim=0), self.outcomes), device=X.device))

        means = X[:, 0:self.outcomes*2].reshape([X.size(dim=0), self.outcomes, 2])
        sigma = torch.eye(2, device=X.device) * self.softplus(X[:, self.outcomes*2:]).reshape(-1, 2)[:, None]
        mix_normals = dist.MultivariateNormal(means, sigma.reshape([X.size(dim=0), self.outcomes, 2, 2]))
        return -dist.MixtureSameFamily(mix_weights, mix_normals).log_prob(y).mean()

    def llf_equal(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        mix_weights = dist.Categorical(torch.ones((X.size(dim=0), self.outcomes), device=X.device))

        tril = torch.zeros((2, 2), device=X.device).repeat(X.size(dim=0), self.outcomes).reshape([X.size(dim=0), self.outcomes, 2, 2])
        tril_indices = torch.tril_indices(row=2, col=2, offset=0, device=X.device)
        tril[:, :, tril_indices[0], tril_indices[1]] = self.softplus(X[:, self.outcomes*2:]).reshape([X.size(dim=0), self.outcomes, 3])

        means = X[:, 0:self.outcomes*2].reshape([X.size(dim=0), self.outcomes, 2])
        mix_normals = dist.MultivariateNormal(means, scale_tril=tril)
        return -dist.MixtureSameFamily(mix_weights, mix_normals).log_prob(y).mean()

    def llt_equal(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        mix_weights = dist.Categorical(torch.ones((X.size(dim=0), self.outcomes), device=X.device))

        tril = torch.zeros((2, 2), device=X.device).repeat(X.size(dim=0), 1).reshape([X.size(dim=0), 2, 2])
        tril_indices = torch.tril_indices(row=2, col=2, offset=0, device=X.device)
        tril[:, tril_indices[0], tril_indices[1]] = self.softplus(X[:, self.outcomes*2:]).reshape([X.size(dim=0), 3])

        means = X[:, 0:self.outcomes*2].reshape([X.size(dim=0), self.outcomes, 2])
        mix_normals = dist.MultivariateNormal(means, scale_tril=tril[:, None])
        return -dist.MixtureSameFamily(mix_weights, mix_normals).log_prob(y).mean()

    # prob different weights

    def lls_mult(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        mix_weights = dist.Categorical(self.softmax(X[:, self.outcomes*2+self.cov_outputs[self.cov]:].reshape([X.size(dim=0), self.outcomes])))

        means = X[:, 0:self.outcomes*2].reshape([X.size(dim=0), self.outcomes, 2])
        sigma = torch.eye(2, device=X.device) * self.softplus(X[:, self.outcomes*2:self.outcomes*2+self.cov_outputs[self.cov]]).reshape(-1, 1)[:, None]
        mix_normals = dist.MultivariateNormal(means, sigma.reshape([X.size(dim=0), self.outcomes, 2, 2]))
        return -dist.MixtureSameFamily(mix_weights, mix_normals).log_prob(y).mean()

    def lld_mult(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        mix_weights = dist.Categorical(self.softmax(X[:, self.cov_slice:].reshape([X.size(dim=0), self.outcomes])))

        means = X[:, 0:self.coord_slice].reshape([X.size(dim=0), self.outcomes, 2])
        sigma = torch.eye(2, device=X.device) * self.softplus(X[:, self.coord_slice:self.cov_slice]).reshape(-1, 2)[:, None]
        mix_normals = dist.MultivariateNormal(means, sigma.reshape([X.size(dim=0), self.outcomes, 2, 2]))
        return -dist.MixtureSameFamily(mix_weights, mix_normals).log_prob(y).mean()

    def llf_mult(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        mix_weights = dist.Categorical(self.softmax(X[:, self.cov_slice:].reshape([X.size(dim=0), self.outcomes])))

        tril = torch.zeros((2, 2), device=X.device).repeat(X.size(dim=0), self.outcomes).reshape([X.size(dim=0), self.outcomes, 2, 2])
        tril_indices = torch.tril_indices(row=2, col=2, offset=0, device=X.device)
        tril[:, :, tril_indices[0], tril_indices[1]] = self.softplus(X[:, self.coord_slice:self.cov_slice]).reshape([X.size(dim=0), self.outcomes, 3])

        means = X[:, 0:self.coord_slice].reshape([X.size(dim=0), self.outcomes, 2])
        mix_normals = dist.MultivariateNormal(means, scale_tril=tril)
        return -dist.MixtureSameFamily(mix_weights, mix_normals).log_prob(y).mean()

    def llt_mult(self, outputs, labels):
        X = outputs.squeeze().float()
        y = labels.squeeze().float()
        mix_weights = dist.Categorical(self.softmax(X[:, self.cov_slice:].reshape([X.size(dim=0), self.outcomes])))

        tril = torch.zeros((2, 2), device=X.device).repeat(X.size(dim=0), 1).reshape([X.size(dim=0), 2, 2])
        tril_indices = torch.tril_indices(row=2, col=2, offset=0, device=X.device)
        tril[:, tril_indices[0], tril_indices[1]] = self.softplus(X[:, self.coord_slice:self.cov_slice]).reshape([X.size(dim=0), 3])

        means = X[:, 0:self.coord_slice].reshape([X.size(dim=0), self.outcomes, 2])
        mix_normals = dist.MultivariateNormal(means, scale_tril=tril[:, None])
        return -dist.MixtureSameFamily(mix_weights, mix_normals).log_prob(y).mean()

#
#  DO NOT USE \/
#

def first_out_distance_loss(outputs, labels):
    X = outputs.squeeze().float()
    y = labels.squeeze().float()
    pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
    return pdist(X[:, 0:2], y).mean()

def mult_out_weighted_coord_loss(outputs, labels):
    X = outputs.squeeze().float()
    y = labels.squeeze().float()

    Y = y.repeat(1, X.size(dim=1)//2)

    error_by_coord = nn.MSELoss(reduction='none')
    coef = 0.1  # x2 per coord, 1-x2 per output

    E = error_by_coord(X, Y)
    for i in range(2, X.size(dim=1), 2):
        E[:, 0:2] = E[:, 0:2] + torch.mul(E[:, i:i+2], max(0.0, 1 - coef*i))

    total = torch.zeros(1, dtype=torch.float, requires_grad=True, device=X.device)
    return total + E[:, 0:2].mean()

def single_loss(outputs, labels):
    x = outputs.squeeze().float()
    y = labels.squeeze().float()

    total = torch.zeros(1, dtype=torch.float, requires_grad=True, device=x.device)

    pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)

    # mloss = nn.MSELoss()
    # loss = nn.MSELoss(reduction='none')

    # l = loss(x, y)
    # p = pdist(x, y)
    # print("x\n", x)
    # print("y\n", y)
    # print("p\n", p)
    # print("l\n", l)

    return total + pdist(torch.tensor(x[:, 0:2]), y).mean()

def mult_coef_loss(outputs, labels):
    X = outputs.squeeze().float()
    y = labels.squeeze().float()

    Y = y.repeat(1, X.size(dim=1)//2)

    error_by_coord = nn.MSELoss(reduction='none')
    E = error_by_coord(X, Y)

    for i in range(X.size(dim=1)//2):
        E[:, 2*i:2*i+2] = torch.mul(torch.tensor(E[:, 2*i:2*i+2]), 1 - 0.2*i)

    E_out_sum = torch.zeros((X.size(dim=0), 2), device=X.device)
    for i in range(0, X.size(dim=1), 2):
        E_out_sum = E_out_sum + E[:, i:i+2]

    return E_out_sum.mean()

def pair_dist(X, y):
    pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
    D = torch.zeros((X.size(dim=0), 0), device=X.device)
    for i in range(0, X.size(dim=1), 2):
        D = torch.cat((D, pdist(torch.tensor(X[:, i:i+2]), y)), 1)
    return D

def mult_loss(outputs, labels):
    X = outputs.squeeze().float()
    y = labels.squeeze().float()

    total = torch.zeros(1, dtype=torch.float, requires_grad=True, device=X.device)
    total = total + torch.min(pair_dist(X, y), 1, keepdim=True).values.mean()

    # softmin = nn.Softmin(dim=1)
    #
    # total = torch.zeros(1, dtype=torch.float, requires_grad=True, device=X.device)
    # total = total + torch.max(softmin(pair_dist(X, y)), 1, keepdim=True).values.mean()

    # pdist = torch.nn.PairwiseDistance(p=2)
    # X_error = []
    # for i, x in enumerate(X.detach().numpy()):
    #     print(i)
    #     print(x)
    #     x_dist = []
    #     for j in range(0, len(x), 2):
    #         pair = torch.Tensor(x[j:j+2])
    #         print(pair)
    #         print(y[i])
    #         p = pdist(pair, y[i])
    #         print(p)
    #         x_dist.append(p)
    #     print(x_dist)
    #     best = torch.Tensor(x_dist).min()
    #     print(best)
    #     X_error.append(best)
    # print(X_error)
    # total = total + torch.Tensor(X_error).mean()
    # print(total)

    # X_dist = pair_dist(X, y)
    # #softmin = nn.Softmin(dim=1)
    # # sm = softmin(X_dist)
    # # print(sm)
    # m = torch.min(X_dist, 1, keepdim=True)
    # print(m)
    # res = m.values.mean()
    # print(res)
    return total

def mult_out_min_distance_loss(outputs, labels):
    X = outputs.squeeze().float()
    y = labels.squeeze().float()
    total = torch.zeros(1, dtype=torch.float, requires_grad=True, device=X.device)
    return total + torch.min(pair_dist(X, y), 1, keepdim=True).values.mean()

def mult_out_softmin_distance_loss(outputs, labels):
    X = outputs.squeeze().float()
    y = labels.squeeze().float()
    #softmin = nn.Softmin(dim=1)
    softmax = nn.Softmax(dim=1)
    total = torch.zeros(1, dtype=torch.float, requires_grad=True, device=X.device)
    return total + torch.min(softmax(pair_dist(X, y)), 1, keepdim=True).values.mean()
