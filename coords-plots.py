import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import scipy.stats as st
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from sklearn.preprocessing import MinMaxScaler


def load_jsonl(filename):
    filename = f"datasets/{filename}"
    print(f"DATASET\tLOAD\tLoading dataset from {filename}")
    data = pd.read_json(path_or_buf=filename, lines=True)
    print(f"DATASET\tLOAD\tDataset of {len(data.index)} coords is loaded")
    return data


def save_df(data, filename):
    with open(filename, "w") as f:
        data.to_json(f, orient='records', lines=True)
    print(f"VAL\tSAVE\tEstimated data of {len(data.index)} coords is written to file: {filename}")

# coords = load_jsonl(filename)
#
# #print(coords)
# test = coords[::100]
# print(test)
#
# dens_u = sm.nonparametric.KDEMultivariate(data=[coords["longitude"], coords["latitude"]], var_type='cc')
# print(dens_u)
# print(dens_u.bw)
#
# test["density"] = dens_u.pdf(test)
# save_df(test, "datasets/test.jsonl")
# #print(dens_test)
#
# x = test["longitude"]
# y = test["latitude"]
# #plt.pcolormesh([x, y], dens_test, shading='auto')
# #plt.show()


# Kernel Density Estimation surface on map
def kde(coords):
    x, y = coords["longitude"], coords["latitude"]

    xmin, xmax = -180, 180
    ymin, ymax = -90, 90

    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    print(f"KDE for {len(xx)**2} points is calculating")

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    # f = np.reshape(kernel(positions).T, xx.shape)
    # kde = pd.DataFrame(f)
    # save_df(kde, f"datasets/{kde_results}")
    f = load_jsonl(kde_results)
    print(f)

    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes(projection='3d')

    ncolors = 256
    color_array = plt.get_cmap('rainbow')(range(ncolors))
    color_array[:,-1] = np.linspace(0.2,1.0,ncolors)
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    plt.register_cmap(cmap=map_object)

    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='rainbow_alpha', edgecolor='none')
    ax.contour(xx, yy, f, zdir='z', offset=0, cmap=cm.coolwarm)

    map = Basemap(fix_aspect=False)
    ax.add_collection3d(map.drawcoastlines(linewidth=0.25))
    ax.add_collection3d(map.drawcountries(linewidth=0.35))

    ax.set_yticks(range(-90, 90, 30))
    ax.set_xticks(range(-180, 180, 30))

    ax.set_box_aspect((4, 3, 1))
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE for 200x200 points estimated from worldwide tweets 2022')
    fig.colorbar(surf, shrink=0.5, aspect=10, location='left') # add color bar indicating the PDF
    ax.view_init(20, -60)

    pic = f"results/img/kde_test_world.png"

    #fig.tight_layout()
    plt.savefig(pic, dpi=600)
    print(f"VAL\tSAVE\tPlot of {len(f.index)} samples is drawn to file: {pic}")
    plt.show()


# GMM clustering of point on the map
def gmm(coords, peaks, seed):
    X = coords.to_numpy(dtype=float)
    xmin, xmax = -180., 180.
    ymin, ymax = -90., 90.
    print(X)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    map = Basemap()
    map.drawcoastlines(linewidth=0.5, color="black")
    map.drawcountries(linewidth=0.7, color="black")
    map.drawparallels(np.arange(ymin, ymax, 30.))
    map.drawmeridians(np.arange(xmin, xmax, 30.))
    map.drawmapboundary(fill_color='azure')
    map.fillcontinents(color='white', lake_color='azure')

    print(f"Calculating clusters of {X.shape[0]} points from GMM with {peaks} means and random seed {seed}")
    gmm = GaussianMixture(n_components=peaks, covariance_type='full', random_state=seed).fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    size = probs.max(1)**2

    map.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap="turbo", zorder=5)
    #plt.scatter(X[:, 0], X[:, 1], s=4, c="black")

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        U, s, Vt = np.linalg.svd(covar)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(pos, nsig * width, nsig * height, angle,
                                 alpha=w * w_factor, color="black"))

    plt.title(f'Scatter plot of coordinated')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


# BIC and AIC criterion for GMMs of different peaks number
def gmm_crit(coords, start, end, step, seed):
    X = coords.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(20, 10))

    models = []
    n_components = []
    for n in range(start, end, step):
        print(f"Calculating model with {n} means")
        n_components.append(n)
        models.append(GaussianMixture(n_components=n,
                                      covariance_type='full',
                                      verbose=1,
                                      random_state=seed).fit(X))

    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    #plt.show()

    f = f"results/img/gmm.png"
    plt.savefig(f)


# write GMM to jsonl file
def save_gmm(gmm, filename):
    gmm_df = pd.DataFrame(columns=["weights", "means", "covariances", "precisions", "precisions_cholesky"])
    gmm_df[["means", "covariances", "precisions", "precisions_cholesky"]] = gmm_df[["means", "covariances", "precisions", "precisions_cholesky"]].astype(object)
    gmm_df["weights"] = gmm.weights_

    weights = gmm.weights_
    print('2 dec - Estimated number of clusters: ' + str((np.round(weights, 2) > 0).sum()))
    print('3 dec - Estimated number of clusters: ' + str((np.round(weights, 3) > 0).sum()))
    print('4 dec - Estimated number of clusters: ' + str((np.round(weights, 4) > 0).sum()))
    print('5 dec - Estimated number of clusters: ' + str((np.round(weights, 5) > 0).sum()))

    for i in range(len(gmm.covariances_)):
        gmm_df.at[i, "means"] = np.array(gmm.means_[i])
        gmm_df.at[i, "covariances"] = np.array(gmm.covariances_[i])
        gmm_df.at[i, "precisions"] = np.array(gmm.precisions_[i])
        gmm_df.at[i, "precisions_cholesky"] = np.array(gmm.precisions_cholesky_[i])
    #print(gmm_df)

    with open(filename, "w") as f:
        gmm_df.to_json(f, orient='records', lines=True)
    print(f"PARAM\tSAVE\tParameters for GMM with {len(gmm_df.index)} means are written to file: {filename}")


# read GMM from jsonl file
def load_gmm(filename):
    data = pd.read_json(path_or_buf=filename, lines=True)
    #print(data)
    print(f"PARAM\tLOAD\tParameters for GMM with {len(data.index)} means are loaded")
    means, covs, prec, prec_chol = [], [], [], []
    for i in range(len(data.index)):
        means.append(np.array(data.at[i, "means"]))
        covs.append(np.array(data.at[i, "covariances"]))
        prec.append(np.array(data.at[i, "precisions"]))
        prec_chol.append(np.array(data.at[i, "precisions_cholesky"]))

    print(f"MODEL\tINIT\tInitialization of GMM with {len(data.index)} means")
    gmm = GaussianMixture(n_components=len(data.index), covariance_type='full', max_iter=1, verbose=0, random_state=seed)
    gmm.weights_ = data["weights"].values

    weights = gmm.weights_
    print('2 dec - Estimated number of clusters: ' + str((np.round(weights, 2) > 0).sum()))
    print('3 dec - Estimated number of clusters: ' + str((np.round(weights, 3) > 0).sum()))
    print('4 dec - Estimated number of clusters: ' + str((np.round(weights, 4) > 0).sum()))
    print('5 dec - Estimated number of clusters: ' + str((np.round(weights, 5) > 0).sum()))

    gmm.means_ = np.array(means)
    gmm.covariances_ = np.array(covs)
    gmm.precisions_ = np.array(prec)
    gmm.precisions_cholesky_ = np.array(prec_chol)

    print(f"MODEL\tSET\tParameters of GMM with {len(data.index)} means are set")
    return gmm


# GMM clustering of point on the map
def plot_gmm(gmm, coords):
    X = coords.to_numpy(dtype=float)
    xmin, xmax = -180., 180.
    ymin, ymax = -90., 90.
    #print(X)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    map = Basemap()
    map.drawcoastlines(linewidth=0.5, color="black")
    map.drawcountries(linewidth=0.7, color="black")
    map.drawparallels(np.arange(ymin, ymax, 30.))
    map.drawmeridians(np.arange(xmin, xmax, 30.))
    map.drawmapboundary(fill_color='azure')
    map.fillcontinents(color='white', lake_color='azure')

    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    size = probs.max(1)**2

    map.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap="turbo", zorder=5)

    #w_factor = 0.9 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        U, s, Vt = np.linalg.svd(covar)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(pos, nsig * width, nsig * height, angle,
                                 color="black", alpha=0.05))

    plt.title(f'Scatter plot of coordinates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    f = f"results/img/gmm_plot.png"
    plt.savefig(f)


# fitting GMM to coords data
def calc_gmm(coords, peaks, seed, iter, gmm_filename):
    X = coords.to_numpy(dtype=float)
    print(f"Calculating GMM with {peaks} means for max {iter} iterations")
    gmm = GaussianMixture(n_components=peaks,
                       covariance_type='full',
                       verbose=1,
                       n_init=1,
                       max_iter=iter,
                       random_state=seed).fit(X)
    save_gmm(gmm, gmm_filename)


# fitting Bayesian GMM to coords data
def calc_bgmm(coords, peaks, seed, iter, bgmm_filename):
    X = coords.to_numpy(dtype=float)
    print(f"Calculating BGMM with {peaks} means for max {iter} iterations")
    bgmm = BayesianGaussianMixture(n_components=peaks,
                               covariance_type='full',
                               verbose=1,
                               n_init=1,
                               max_iter=iter,
                               random_state=seed).fit(X)
    save_gmm(bgmm, bgmm_filename)


# generate grid by number of steps
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
    #xx, yy = np.meshgrid(peaks[:, 0], peaks[:, 1])
    return xx, yy


# GMM likelihood score surface plot on the map (shifted to min as 0)
def gmm_likelihood(gmm):
    xx, yy = map_grid(gmm.means_, 100)
    XX = np.array([xx.ravel(), yy.ravel()]).T

    print(f"Calculating scores from GMM for {len(XX)} points")
    Z = gmm.score_samples(XX)
    zmin = np.min(Z)
    zmax = np.max(Z)
    print(f"Original\tMin: {zmin}\tMax: {zmax}")
    Z = Z - np.min(Z)
    zmin = np.min(Z)
    zmax = np.max(Z)
    print(f"Adjusted\tMin: {zmin}\tMax: {zmax}")
    #Z = np.exp(Z)
    Z = Z.reshape(xx.shape)
    #print(Z)

    # P = gmm.predict_proba(XX)
    # S = P.max(1)
    # S = S.reshape(xx.shape)
    # print(S)
    # scaler = MinMaxScaler((0, 100))
    # Z = scaler.fit_transform(Z)
    # print(Z)

    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes(projection='3d')

    ncolors = 256
    color_array = plt.get_cmap('rainbow')(range(ncolors))
    color_array[:,-1] = np.linspace(0.2,1.0,ncolors)
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    plt.register_cmap(cmap=map_object)

    surf = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap='rainbow_alpha', edgecolor='none')
    contour = ax.contour(xx, yy, Z, levels=np.linspace(zmin, zmax, 500), zdir='z', offset=0, cmap='rainbow_alpha')

    map = Basemap(fix_aspect=False)
    ax.add_collection3d(map.drawcoastlines(linewidth=0.25))
    ax.add_collection3d(map.drawcountries(linewidth=0.35))

    ax.set_yticks(range(-90, 90, 30))
    ax.set_xticks(range(-180, 180, 30))
    ax.set_zticks(range(int(zmin), int(zmax), 5))

    ax.set_box_aspect((4, 3, 1))
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('Log-likelihood')
    ax.set_title(f'Surface plot of likelihood from GMM with {len(gmm.weights_)} peaks for {len(XX)} points estimated from worldwide tweets 2022')
    fig.colorbar(surf, shrink=0.5, aspect=10, location='left') # add color bar indicating the likelihood
    ax.view_init(30, -60)

    pic = f"results/img/gmm_likelihood_world.png"

    #fig.tight_layout()
    plt.savefig(pic, dpi=600)
    print(f"PLOT\tSAVE\tPlot of GMM likelihood for {len(XX)} points is drawn to file: {pic}")
    plt.show()


# GMM PDF surface plot on the map
def gmm_density(gmm):
    xmin, xmax = -180, 180
    ymin, ymax = -90, 90
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    XX = np.array([xx.ravel(), yy.ravel()]).T

    print(f"Calculating scores from GMM for {len(XX)} points")
    Z = gmm.score_samples(XX)
    zmin = np.min(Z)
    zmax = np.max(Z)
    print(f"Original\tMin: {zmin}\tMax: {zmax}")
    Z = np.exp(Z)
    zmin = np.min(Z)
    zmax = np.max(Z)
    print(f"Probability\tMin: {zmin}\tMax: {zmax}")
    Z = Z.reshape(xx.shape)
    #print(Z)

    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes(projection='3d')

    ncolors = 256
    color_array = plt.get_cmap('rainbow')(range(ncolors))
    color_array[:,-1] = np.linspace(0.2,1.0,ncolors)
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    plt.register_cmap(cmap=map_object)

    surf = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap='rainbow_alpha', edgecolor='none')
    contour = ax.contour(xx, yy, Z, levels=np.linspace(zmin, zmax, 500), zdir='z', offset=0, cmap='rainbow_alpha')

    map = Basemap(fix_aspect=False)
    ax.add_collection3d(map.drawcoastlines(linewidth=0.25))
    ax.add_collection3d(map.drawcountries(linewidth=0.35))

    ax.set_yticks(range(-90, 90, 30))
    ax.set_xticks(range(-180, 180, 30))
    #ax.set_zticks(range(int(zmin), int(zmax)))

    ax.set_box_aspect((4, 3, 1))
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('Probability')
    ax.set_title(f'Surface plot of probability from GMM with {len(gmm.weights_)} peaks for {len(XX)} points estimated from worldwide tweets 2022')
    fig.colorbar(surf, shrink=0.5, aspect=10, location='left') # add color bar indicating the probability
    ax.view_init(10, -60)

    pic = f"results/img/gmm_probability_world.png"

    #fig.tight_layout()
    plt.savefig(pic, dpi=600)
    print(f"PLOT\tSAVE\tPlot of GMM probability for {len(XX)} points is drawn to file: {pic}")
    plt.show()


# GMM PDF contour plot on the map
def gmm_contour(gmm):
    xmin, xmax = -180, 180
    ymin, ymax = -90, 90
    xx, yy = map_grid(gmm.means_, 200)
    XX = np.array([xx.ravel(), yy.ravel()]).T

    print(f"Calculating scores from GMM for {len(XX)} points")
    Z = gmm.score_samples(XX)
    zmin = np.min(Z)
    zmax = np.max(Z)
    print(f"Original\tMin: {zmin}\tMax: {zmax}")
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    contour = ax.contour(xx, yy, Z, levels=np.linspace(zmin, zmax, 300), cmap='RdYlGn_r', linewidths=0.5)

    map = Basemap()
    map.drawcoastlines(linewidth=0.5, color="black")
    map.drawcountries(linewidth=0.7, color="black")
    map.drawparallels(np.arange(ymin, ymax, 30.))
    map.drawmeridians(np.arange(xmin, xmax, 30.))
    map.drawmapboundary(fill_color='azure')
    map.fillcontinents(color='white', lake_color='azure')

    peaks = gmm.means_
    map.scatter(peaks[:, 0], peaks[:, 1], c=gmm.weights_, cmap="RdYlGn_r", s=0.7, zorder=5)

    plt.title(f'Contour plot of likelihood from GMM with {len(gmm.weights_)} peaks for {len(XX)} points estimated from worldwide tweets 2022')
    fig.colorbar(contour, shrink=0.5, aspect=10, location='left') # add color bar indicating the likelihood
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    pic = f"results/img/gmm_contour_plot.png"
    plt.savefig(pic, dpi=600)
    print(f"PLOT\tSAVE\tPlot of GMM probability for {len(XX)} points is drawn to file: {pic}")

    plt.show()


filename = "twitter-2020-02-28.txt"
world = "map-world/world-twitter-2022-coords.jsonl"
kde_results = "kde_world_2022.jsonl"

coords = load_jsonl(filename)
print(coords)

peaks = 1000
seed = 42
iter = 1000

gmm_filename = f"datasets/gmm-p{peaks}-c{len(coords.index)}.jsonl"
bgmm_filename = f"datasets/bgmm-p{peaks}-c{len(coords.index)}.jsonl"
gmm_200 = "datasets/200-gmm.jsonl"
bgmm_cluser = "datasets/bgmm-p200-c12057022.jsonl"

#calc_gmm(coords, peaks, seed, iter, gmm_filename)
#calc_bgmm(coords, peaks, seed, iter, bgmm_filename)

# gmm = load_gmm(bgmm_cluser)
X = coords[["longitude", "latitude"]].to_numpy(dtype=float)
# Z = gmm.score_samples(X)
# print(Z.min(), Z.max(), Z.mean())
# gmm = load_gmm(gmm_200)
# Z = gmm.score_samples(X)
# print(Z.min(), Z.max(), Z.mean())
# gmm_likelihood(gmm)
# gmm_density(gmm)
# gmm_contour(gmm)



# test for differences in covariance
#
# from scipy.linalg import cholesky
#
# cov = "spherical"
# gmm = GaussianMixture(5, covariance_type=cov, max_iter=1).fit(X)
# print(cov)
# print(gmm.covariances_)
# print(gmm.precisions_)
# print(gmm.precisions_cholesky_)
# print(cholesky(gmm.covariances_))
#
# cov = "diag"
# gmm = GaussianMixture(5, covariance_type=cov, max_iter=1).fit(X)
# print(cov)
# print(gmm.covariances_)
# print(gmm.precisions_)
# print(gmm.precisions_cholesky_)
#
# cov = "full"
# gmm = GaussianMixture(5, covariance_type=cov, max_iter=1).fit(X)
# print(cov)
# print(gmm.covariances_)
# print(gmm.precisions_)
# print(gmm.precisions_cholesky_)
#
# cov = "tied"
# gmm = GaussianMixture(5, covariance_type=cov, max_iter=1).fit(X)
# print(cov)
# print(gmm.covariances_)
# print(gmm.precisions_)
# print(gmm.precisions_cholesky_)
