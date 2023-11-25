import os
from os.path import join
import pickle
from nonpoisson import catalogs, paths
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from matplotlib.lines import Line2D
import time
from functools import partial
from multiprocessing import Pool
import csep
from csep.core.regions import CartesianGrid2D
from csep.core.forecasts import GriddedForecast
from csep.core.catalogs import CSEPCatalog
from csep.core.poisson_evaluations import paired_t_test
from csep.utils.stats import poisson_joint_log_likelihood_ndarray
import datetime
from nonpoisson.paths import Gisborne, Christchurch, Wellington, Queenstown, Napier
import seaborn as sns
import cartopy
from csep.utils.plots import plot_comparison_test
import rioxarray as rxr

# style must be one of white, dark, whitegrid, darkgrid, ticks
sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})


def log_score(forecast, obs_catalog):
    forecast_data = forecast.data
    observed_data = obs_catalog.spatial_magnitude_counts()
    expected_forecast_count = np.sum(forecast_data)
    log_bin_expectations = np.log(forecast_data.ravel())

    target_idx = np.nonzero(observed_data.ravel())
    observed_data_nonzero = observed_data.ravel()[target_idx]
    target_event_forecast = log_bin_expectations[
                                target_idx] * observed_data_nonzero
    obs_ll = poisson_joint_log_likelihood_ndarray(target_event_forecast,
                                                  observed_data_nonzero,
                                                  expected_forecast_count)
    return obs_ll


def dist_array(geom):
    """
    Calculate geodesic distance, using small_angle hyphotesis, between a point
    and an array of points. Uses precalculated deg2rad(x) and cos(x), sin(x)
    for parallelization efficiency.

    **** I suggest that this formula should be changed, as it gives prec error!
    Input:
        - geom[0] (dtype=list/array, shape=(2,)):
                    Single point to evaluate distances (in radians)
        - geom[1] (dtype=list of 3 array), shape=(3, (n,3)):
                    List of 3 arrays of the grid geom:
                        geom[1][0] (dtype=array, shape(n,): lon      (in rads.)
                        geom[1][1] (dtype=array, shape(n,): cos(lat) (in rads.)
                        geom[1][2] (dtype=array, shape(n,): sin(lat) (in rads.)
    """

    dtheta = (np.sin(geom[0][1]) * geom[1][:, 2] +
              np.cos(geom[0][1]) * geom[1][:, 1] *
              np.cos(geom[1][:, 0] - geom[0][0]))

    row = 6371. * np.arccos(np.round(dtheta, 14))

    return row


def truncated_gr(n, bval, mag_bin, learning_mmin, target_mmin, target_mmax):

    aval = np.log10(n) + bval * target_mmin
    mag_bins = np.arange(learning_mmin, target_mmax + mag_bin, mag_bin)
    mag_weights = (10 ** (aval - bval * (mag_bins - mag_bin / 2.)) - 10 ** (aval - bval * (mag_bins + mag_bin / 2.))) / \
                  (10 ** (aval - bval * (learning_mmin - mag_bin/2.)) - 10 ** (aval - bval * (target_mmax + mag_bin/2.)))
    return aval, mag_bins, mag_weights


def calibrate_models(catalog_fit, training_end=datetime.datetime(2010, 1, 1)):

    models = ['gaussian', 'powerlaw', 'adaptive']
    param_space = {'gaussian': np.linspace(8, 60, 20),
                   'powerlaw': np.linspace(5, 25, 20),
                   'adaptive': np.arange(1, 20)}

    forecasts = {model: [] for model in models}
    log_scores = {model: [] for model in models}

    for model in models:

        Exp = Experiment(catalog_fit, None, None)
        Exp.nz_grid()
        Exp.filter_catalogs()
        Exp.get_distances()

        indx_training = np.sum(Exp.training_cat.get_epoch_times() <
                               training_end.timestamp()*1000)
        indx_testing = np.sum(Exp.test_cat.get_epoch_times() <
                              training_end.timestamp()*1000)

        Exp.test_cat.catalog = Exp.test_cat.catalog[indx_testing:]
        N_total = Exp.test_cat.get_number_of_events()

        for param in param_space[model]:
            pdf = Exp.get_ssm(np.arange(0, indx_training),
                              model=model,
                              kernel_size=param)
            forecast = GriddedForecast(region=Exp.test_region,
                                       magnitudes=Exp.test_region.magnitudes,
                                       data=pdf.reshape(-1, 1) * N_total)
            forecasts[model].append(forecast)
            log_scores[model].append(log_score(forecast, Exp.test_cat))
        log_scores[model] = np.array(log_scores[model])

    return {'space': param_space, 'forecasts': forecasts, 'scores': log_scores}


def plot_fit_results(results_fit, cat,
                     figpath=None, figprefix='nz', clim=(-4, 0)):

    space = results_fit['space']
    scores = results_fit['scores']
    forecasts = results_fit['forecasts']

    figure, axs = plt.subplots(1, 3, figsize=(12, 5))

    # Gaussian SSM
    axs[0].axvline(space['gaussian'][scores['gaussian'].argmax()],
                   color='k',
                   linestyle='--')
    axs[0].plot(space['gaussian'], scores['gaussian'], 'o-')
    axs[0].plot(space['gaussian'][scores['gaussian'].argmax()],
                scores['gaussian'].max(), "^",
                label=r'Optimal $\sigma$:'
                      f" {space['gaussian'][scores['gaussian'].argmax()]:.1f}",
                markersize=13)
    axs[0].set_xlabel(r'Gaussian smoothing $\sigma$ [km]')
    axs[0].set_ylabel('Log-Likelihood')
    axs[0].set_title('Gaussian')
    axs[0].legend()

    # Power-Law SSM
    axs[1].axvline(space['powerlaw'][scores['powerlaw'].argmax()],
                   color='k',
                   linestyle='--')
    axs[1].plot(space['powerlaw'], scores['powerlaw'], 'o-')
    axs[1].plot(space['powerlaw'][scores['powerlaw'].argmax()],
                scores['powerlaw'].max(), "^",
                label=r'Optimal $d$:'
                      f" {space['powerlaw'][scores['powerlaw'].argmax()]:.1f}",
                markersize=13)

    axs[1].set_xlabel(r'Smoothing distance $d$ [km]')
    axs[1].set_ylabel('Log-Likelihood')
    axs[1].set_title('Power-law')
    axs[1].legend()

    # Adaptive SSM
    axs[2].axvline(space['adaptive'][scores['adaptive'].argmax()],
                   color='k',
                   linestyle='--')
    axs[2].plot(space['adaptive'], scores['adaptive'], 'o-')
    axs[2].plot(space['adaptive'][scores['adaptive'].argmax()],
                scores['adaptive'].max(), "^",
                label=r'Optimal $N$:'
                      f" {space['adaptive'][scores['adaptive'].argmax()]:.1f}",
                markersize=13)

    axs[2].set_xlabel(r'Nearest Neighbor $N$')
    axs[2].set_ylabel('Log-Likelihood')
    axs[2].set_title('Adaptive')
    axs[2].legend()

    figure.suptitle(f'SSM parameter fit - {cat.name}')
    plt.tight_layout()
    if figpath:
        plt.savefig(join(figpath, f'{figprefix}_params_fit.png'), dpi=600)
    plt.show()

    plot_args = {'basemap': None,
                 'clim': clim,
                 'cmap': 'rainbow'}
    forecasts['gaussian'][np.argmax(scores['gaussian'])].plot(
        plot_args={**plot_args,
                   'title': 'Gaussian SSM - optimal'})
    plt.savefig(paths.get('temporal', 'fig', f'{figprefix}_gaussian'), dpi=600)
    plt.show()

    forecasts['powerlaw'][np.argmax(scores['powerlaw'])].plot(
        plot_args={**plot_args,
                   'title': 'Power-law SSM  - optimal'})
    plt.savefig(paths.get('temporal', 'fig', f'{figprefix}_powerlaw'), dpi=600)
    plt.show()

    forecasts['adaptive'][np.argmax(scores['adaptive'])].plot(
        plot_args={**plot_args,
                   'title': 'Adaptive SSM  - optimal'})
    plt.savefig(paths.get('temporal', 'fig', f'{figprefix}_adaptive'), dpi=600)
    plt.show()


def get_global_model(catalog, model='adaptive', param=7):
    Exp = Experiment(catalog, None, None)
    Exp.nz_grid()
    Exp.filter_catalogs()
    Exp.get_distances()

    pdf = Exp.get_ssm(range(catalog.get_number_of_events()),
                      model=model,
                      kernel_size=param)
    forecast = GriddedForecast(region=Exp.test_region,
                               magnitudes=Exp.test_region.magnitudes,
                               data=pdf.reshape(-1, 1))
    return forecast


class Experiment:

    def __init__(self, catalogue, n_test, point):

        self.point = point
        self.catalogue = catalogue
        self.test_n = n_test

        self.training_grid = None
        self.training_region = None
        self.training_cat = None
        self.test_grid = None
        self.test_region = None
        self.test_cat = None

        self.uniform_forecast = None
        self.dists = None
        self.dists_cat = None

    def nz_grid(self, magnitudes=np.array([5.])):
        self.training_region = csep.regions.nz_csep_collection_region(
            magnitudes=magnitudes)
        self.test_region = csep.regions.nz_csep_region(
            magnitudes=magnitudes)

    def create_grids(self, dh=0.1,
                     magnitudes=np.array([5.]),
                     train_magnitudes=np.array([4.]),
                     r_train=200,
                     r_test=150,
                     bound_train_region=None,
                     bound_test_region=None):

        region_grid = self.catalogue.region.midpoints()
        extent = [np.min(region_grid[:, 0]), np.min(region_grid[:, 1]),
                  np.max(region_grid[:, 0]), np.max(region_grid[:, 1])]

        x, y = [coord.ravel() for coord in np.meshgrid(
                            np.arange(extent[0] + dh/2,
                                      extent[2] + dh/2, dh),
                            np.arange(extent[1] + dh/2,
                                      extent[3] + dh/2, dh))]
        grid_geom = np.vstack((np.deg2rad(x),
                               np.cos(np.deg2rad(y)),
                               np.sin(np.deg2rad(y)))).T
        dists = np.array(dist_array((np.deg2rad(self.point), grid_geom)))

        self.training_grid = np.vstack((
                            x[dists <= r_train] ,
                            y[dists <= r_train])).T
        self.test_grid = np.vstack((x[dists <= r_test],
                                    y[dists <= r_test])).T

        if bound_test_region:
            idxs = bound_test_region.get_masked(self.test_grid[:, 0],
                                          self.test_grid[:, 1])
            self.test_grid = self.test_grid[~idxs]
        if bound_train_region:
            idxs = bound_train_region.get_masked(self.training_grid[:, 0],
                                           self.training_grid[:, 1])
            self.training_grid = self.training_grid[~idxs]
        self.training_region = CartesianGrid2D.from_origins(
                    self.training_grid, dh=0.1, magnitudes=train_magnitudes)
        self.test_region = CartesianGrid2D.from_origins(
                    self.test_grid, dh=0.1, magnitudes=magnitudes)

    @classmethod
    def load(cls, filename=None):

        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def get_distances(self):

        """
        Fast paralellized function that calculates the distances between two
        set of points.

        Output
        --------
        self.dists[(attr, att2)] : array
            Distance matrix of shape n × m.

        """

        start = time.process_time()
        grid_points = np.deg2rad(self.test_region.midpoints())

        catalog_points = np.deg2rad(np.array(
            [[x, y] for x, y in zip(self.training_cat.data['longitude'],
                                    self.training_cat.data['latitude'])]))
        cat_loc_geom = np.vstack((catalog_points[:, 0],
                                  np.cos(catalog_points[:, 1]),
                                  np.sin(catalog_points[:, 1]))).T

        cat_grid_positions = [[p, cat_loc_geom] for p in grid_points]
        self.dists = np.array(list(map(dist_array, cat_grid_positions)))[:]

        cat_cat_positions = [[p, cat_loc_geom] for p in catalog_points]
        self.dists_cat = np.array(list(map(dist_array, cat_cat_positions)))

    def get_ssm(self,
                cat_inds,
                model='gaussian',
                kernel_size=50.,
                power=1.5,
                dist_cutoff=5):

        """
        Fixed-sized Kernel smoothing from Helmstetter et al, 2007
        Modified from func_kernelfix.m developed by Hiemer, S. in matlab

        Input

        - subcat (string): Name of the sub-catalog to calculate
        - KernelSize (float): Fix distance of the smoothing kernel
        - power (float): 1.0: Wang-Kernel, 1.5: Helmstetter-Kernel


        Outpu
        - pdfX
            Probability density for each grid cell, as the sum of every catalog
            event contribution.

        """

        pdfX = np.zeros(self.test_region.midpoints().shape[0])

        if model == 'powerlaw':
            for d in self.dists.T[cat_inds, :]:
                kernel_i = 1. / ((d ** 2 + kernel_size ** 2) ** power)
                kernel_i /= (np.sum(kernel_i))
                pdfX += kernel_i

        elif model == 'gaussian':
            for d in self.dists.T[cat_inds, :]:
                kernel_i = 1 / np.sqrt(2*np.pi) / kernel_size * \
                           np.exp(-d ** 2 / (2 * kernel_size ** 2))
                kernel_i /= (np.sum(kernel_i))
                pdfX += kernel_i

        elif model == 'adaptive':

            nearest_neigh = kernel_size
            # Get distances between all events
            d_cat2cat = self.dists_cat[cat_inds, :][:, cat_inds]
            kernel = np.sort(d_cat2cat).T[nearest_neigh, :]
            if dist_cutoff:
                kernel[kernel < 5] = 5
            for d, k in zip(self.dists.T[cat_inds, :], kernel):
                kernel_i = 1./((d**2 + k**2)**power)
                kernel_i /= (np.sum(kernel_i))
                pdfX += kernel_i

        pdfX /= np.sum(pdfX)

        return pdfX[:]

    def rates_from_model(self, model):

        """
        Fixed-sized Kernel smoothing from Helmstetter et al, 2007
        Modified from func_kernelfix.m developed by Hiemer, S. in matlab

        Input

        - subcat (string): Name of the sub-catalog to calculate
        - KernelSize (float): Fix distance of the smoothing kernel
        - power (float): 1.0: Wang-Kernel, 1.5: Helmstetter-Kernel


        Outpu
        - pdfX
            Probability density for each grid cell, as the sum of every catalog
            event contribution.

        """

        midpoints = model.region.midpoints()
        idxs_model = self.test_region.get_masked(
            midpoints[:, 0],
            midpoints[:, 1]
        )

        rates = model.data[~idxs_model]


        idxs_test = self.test_region.get_index_of(
            midpoints[~idxs_model, 0],
            midpoints[~idxs_model, 1])

        pdf = np.zeros((self.test_region.midpoints().shape[0], 1))
        pdf[idxs_test] = rates
        pdf /= np.sum(pdf)
        self.model_forecast = GriddedForecast(
                            region=self.test_region,
                            magnitudes=self.test_region.magnitudes,
                            data=pdf)

    def filter_catalogs(self):

        self.training_cat = copy.deepcopy(self.catalogue).filter_spatial(
            region=self.training_region, update_stats=True)
        # print(np.argwhere(np.diff(self.training_cat.get_epoch_times())==0))
        assert np.all(np.diff(self.training_cat.get_epoch_times()) > 0)
        self.test_cat = copy.deepcopy(self.catalogue).filter_spatial(
            region=self.test_region, update_stats=True)
        self.test_cat.filter([r'magnitude >= 5.0'])

        assert np.all(np.diff(self.test_cat.get_epoch_times()) > 0)

    def get_uniform_forecast(self):

        n_cells = self.test_region.midpoints().shape[0]
        data = np.ones((n_cells, 1)) / n_cells
        mags = self.test_region.magnitudes
        if mags.shape[0] > 1:
            _, _, gr = truncated_gr(1, 1, 0.1,
                                    mags.min(), mags.min(), mags.max())
            data = np.outer(data, gr)

        self.uniform_forecast = GriddedForecast(
                                region=self.test_region,
                                magnitudes=mags,
                                data=data)

    def create_forecast(self,
                        n_train,
                        max_iterations=10,
                        ssm='powerlaw',
                        kernel_size=50.0):

        n_cat = self.test_cat.get_number_of_events()
        train_idxs = []
        test_idxs = []

        train_times = self.training_cat.get_epoch_times()
        test_times = self.test_cat.get_epoch_times()

        # Fixed n_train
        if 2 > test_times.shape[0]:
            return [], [], []
        t_end_test = test_times[-1]
        ind_last_train = np.sum(train_times < t_end_test) - 1

        available_idxs = np.arange(ind_last_train - n_train + 1)
        if available_idxs.shape[0] == 0:
            return [], [], []

        for iter in range(max_iterations):
            index = random.choice(available_idxs)
            forecast_inds = [index + k for k in range(n_train)]
            t_train_f = train_times[forecast_inds[-1]]

            if t_train_f in test_times:
                idx_test_0 = np.sum(test_times < t_train_f) + 1
            else:
                idx_test_0 = np.sum(test_times < t_train_f)

            idx_test_f = min(idx_test_0 + self.test_n, n_cat)
            test_inds = np.arange(idx_test_0, idx_test_f)
            train_idxs.append(forecast_inds)
            test_idxs.append(test_inds)

        forecasts = []
        train_cats = []
        test_cats = []

        if ssm == 'global':
            for f_ind, t_ind in zip(train_idxs, test_idxs):
                train_cat = CSEPCatalog(data=self.training_cat.catalog[f_ind])
                test_cat = CSEPCatalog(data=self.test_cat.catalog[t_ind],
                                       region=self.uniform_forecast.region)

                forecasts.append(None)
                train_cats.append(train_cat)
                test_cats.append(test_cat)

        else:
            for f_ind, t_ind in zip(train_idxs, test_idxs):
                fit_f = []
                if ssm in ['gaussian', 'powerlaw', 'adaptive']:
                    if kernel_size == 0:
                        n_learn = np.round(len(f_ind)*0.75).astype(int)
                        if ssm in ['gaussian', 'powerlaw']:
                            params_search = [15, 22, 30, 39, 50, 75, 100, 125]
                        elif ssm == 'adaptive':
                            params_search = np.array([4, 6, 8, 10, 12, 15])
                            params_search = params_search[params_search < n_learn]

                        for param_i in params_search:

                            pdf_i = self.get_ssm(f_ind[:n_learn],
                                                 model=ssm, kernel_size=param_i)

                            data_i = pdf_i.reshape(-1, 1) * (len(f_ind) - n_learn)
                            forecast_i = GriddedForecast(
                                region=self.test_region,
                                magnitudes=self.training_region.magnitudes,
                                data=data_i)

                            cat_i = CSEPCatalog(
                                data=self.training_cat.catalog[f_ind[n_learn:]],
                                region=forecast_i.region)
                            cat_i.filter_spatial(self.test_region, in_place=True)
                            cat_i.filter(f'magnitude >= {np.median(cat_i.get_magnitudes())}')

                            fit_f.append(log_score(forecast_i, cat_i))

                        k_size = params_search[np.argmax(fit_f)]
                    else:
                        k_size = kernel_size
                    pdf = self.get_ssm(f_ind, model=ssm, kernel_size=k_size)
                    if self.test_region.magnitudes.shape[0] == 1:
                        data = pdf.reshape(-1, 1)
                    else:
                        mag = self.test_region.magnitudes
                        _, _, mag_weights = truncated_gr(
                            1, 1, 1, mag.min(), mag.min(), mag.max())

                        data = np.outer(pdf, mag_weights)

                    forecast = GriddedForecast(
                        region=self.test_region,
                        magnitudes=self.test_region.magnitudes,
                        data=data)

                elif ssm == 'model':

                    forecast = self.model_forecast


                train_cat = CSEPCatalog(data=self.training_cat.catalog[f_ind],
                                        region=self.training_region)
                test_cat = CSEPCatalog(data=self.test_cat.catalog[t_ind],
                                       region=forecast.region)
                plt.show()
                forecasts.append(forecast)
                train_cats.append(train_cat)
                test_cats.append(test_cat)

        return forecasts, train_cats, test_cats

    def evaluate_forecasts(self, forecasts, catalogs, alpha=0.2):

        tests = {'t_stats': [],
                 'ranks': []}

        assert len(forecasts) == len(catalogs)

        for f, c in zip(forecasts, catalogs):

            urz = self.uniform_forecast
            urz.scale(c.get_number_of_events())
            f.scale(c.get_number_of_events())

            t_test = paired_t_test(f, urz, c, alpha=alpha)
            dist = np.array(t_test.test_distribution)

            # if np.isnan(dist).any():
            #     continue

            if (dist > 0).all():  # Statistically better
                tests['ranks'].append(1)
            elif (dist > 0).any():   # Statistically indistinguishable
                tests['ranks'].append(0)
            else:  # Statistically worse
                tests['ranks'].append(-1)
            tests['t_stats'].append(t_test)

        return tests

    def save(self, filename=None):

        """
        Serializes Model_results object into a file
        :param filename: If None, save in results folder named with self.name
        """

        with open(filename, 'wb') as obj:
            pickle.dump(self, obj, protocol=None,
                        fix_imports=True, buffer_callback=None)


def run_singleloc(origin,
                  catalog=None,
                  model=None,
                  n_training=(100,),
                  max_iters=50,
                  n_max=10,
                  ssm_class='gaussian',
                  kernel_size=5,
                  r_train=150,
                  r_test=250,
                  alpha=0.2,
                  bound_test_region=None,
                  bound_train_region=None,
                  return_models=False):
    id_origin, origin,  = origin
    print(f'Running models centered at {id_origin}: {origin}')
    test_rank = {i: [] for i in n_training}
    prop_rank = {i: [] for i in n_training}
    models = {i: [] for i in n_training}

    A = Experiment(catalog, n_max, origin)
    A.create_grids(bound_train_region=bound_train_region,
                   bound_test_region=bound_test_region,
                   r_train=r_train, r_test=r_test)
    A.filter_catalogs()
    print(f'Train : {A.training_cat.get_number_of_events()}',
          f'Test : {A.test_cat.get_number_of_events()}')
    if A.training_cat.get_number_of_events() < min(n_training):
        return
    A.get_distances()
    A.get_uniform_forecast()
    if model:
        A.rates_from_model(model)

    for i in n_training:

        forecasts, train_cats, test_cats = A.create_forecast(
            n_train=i,
            ssm=ssm_class,
            kernel_size=kernel_size,
            max_iterations=max_iters)
        ttest_results = A.evaluate_forecasts(forecasts, test_cats, alpha=alpha)
        test_rank[i].extend(ttest_results['ranks'])

        unique, counts = np.unique(ttest_results['ranks'], return_counts=True)
        search = [-1, 0, 1]
        search_counts = []
        for counter, pref_model in enumerate(search):
            if pref_model in unique:
                search_counts.append(counts[np.argwhere(unique == pref_model)[0][0]])
            else:
                search_counts.append(0)
        if np.sum(search_counts):
            prop_rank[i] = np.array(search_counts)/np.sum(search_counts)

        else:
            prop_rank[i] = np.zeros(3)
        models[i] = [forecasts, train_cats, test_cats]

    if return_models:
        return prop_rank, models
    else:
        return prop_rank


def run_christchurch_adaptive(figpath=None):

    start = time.process_time()

    ####################################
    # Initialize Catalog and Region
    collection_shp = paths.region_nz_collection
    collection_region = csep.regions.nz_csep_collection_region()
    catalog = catalogs.cat_oq2csep(
        catalogs.filter_cat(
            catalogs.get_cat_nz(name='New Zealand, Non-declustered'),
            mws=(3.99, 10.0), depth=[40, -2],
            start_time=dt(1980, 1, 1),
            shapefile=collection_shp))
    catalog.filter_spatial(collection_region, in_place=True)

    ####################################
    # Set experiment parameters
    centers = [paths.Christchurch[0]]

    seed = 14     # seed for reproducibility
    ssm_class = 'adaptive'   # model class: 'gaussian', 'powerlaw', 'adaptive'
    ssm_param = 7   # gaussian std; powerlaw d; adaptive N; Set 0 for autofit
    nmax = 20           # Testing events
    rtrain = 200       # Collection radius
    rtest = 150        # Forecast radius
    max_cat_iters = 400     # iters for given N through cat of a subregion
    alpha = 0.4
    n_training = [25, 50, 100, 200, 280, 400]

    # Initialize result arrays
    np.random.seed(seed)
    random.seed(seed)

    prop_ranks = {i: np.zeros(3) for i in n_training}
    models = {i: [] for i in n_training}

    ssm_better = []
    ssm_worse = []
    ssm_undist = []

    ####################################
    # Prepare main run function with parameters
    main_func = partial(
        run_singleloc,
        catalog=catalog,
        n_max=nmax,
        r_test=rtest,
        r_train=rtrain,
        n_training=n_training,
        max_iters=max_cat_iters,
        alpha=alpha,
        ssm_class=ssm_class,
        kernel_size=ssm_param,
        bound_test_region=collection_region,
        bound_train_region=collection_region,
        return_models=True)

    ####################################
    # Run
    results = list(map(main_func, enumerate(centers)))

    ####################################
    # Re-order results

    for res in results:
        for i in n_training:
            if res[0][i].shape[0] == 3:
                prop_ranks[i] += res[0][i]
                models[i].extend(res[1][i])
    for n, rank in prop_ranks.items():
        ssm_worse.append(rank[0])
        ssm_undist.append(rank[1])
        ssm_better.append(rank[2])

    ####################################

    ssm_better = np.array(ssm_better)
    ssm_worse = np.array(ssm_worse)
    ssm_undist = np.array(ssm_undist)

    frac_better = ssm_better / (ssm_better + ssm_worse + ssm_undist)
    frac_worse = ssm_worse / (ssm_better + ssm_worse + ssm_undist)
    frac_undist = ssm_undist / (ssm_better + ssm_worse + ssm_undist)

    # Plot figures
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(n_training, np.zeros(len(n_training)),
                    frac_better, color='g', label='SSM is better', alpha=0.2)
    ax.plot(n_training, frac_better, 'g.--')

    ax.fill_between(n_training, frac_better, frac_better + frac_undist,
                    color='gray', label='Indistinguishable', alpha=0.2)
    ax.plot(n_training, 1 - frac_worse, 'r.--')
    ax.fill_between(n_training, frac_better + frac_undist,
                    frac_better + frac_undist + frac_worse,
                    color='r', label='URZ is better',
                    alpha=0.2)
    ax.set_title(f'Christchurch - Adaptive SSM', fontsize=16)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, max(n_training)])
    ax.set_xlabel('Number of training events $N_1$', fontsize=14)
    ax.set_ylabel('Fraction of forecasts', fontsize=14)

    plt.legend(loc=4, fontsize=12)
    if figpath:
        plt.savefig(figpath, dpi=600)
    print(f'ready. {time.process_time() - start:.1f}')
    plt.show()


def run_christchurch_gaussian(figpath=None):

    start = time.process_time()

    ####################################
    # Initialize Catalog and Region
    collection_shp = paths.region_nz_collection
    collection_region = csep.regions.nz_csep_collection_region()
    catalog = catalogs.cat_oq2csep(
        catalogs.filter_cat(
            catalogs.get_cat_nz(name='New Zealand, Non-declustered'),
            mws=(3.99, 10.0), depth=[40, -2],
            start_time=dt(1980, 1, 1),
            shapefile=collection_shp))
    catalog.filter_spatial(collection_region, in_place=True)

    ####################################
    # Set experiment parameters
    centers = [paths.Christchurch[0]]

    seed = 14     # seed for reproducibility
    ssm_class = 'gaussian'   # model class: 'gaussian', 'powerlaw', 'adaptive'
    ssm_param = 17   # gaussian std; powerlaw d; adaptive N; Set 0 for autofit
    nmax = 20           # Testing events
    rtrain = 200       # Collection radius
    rtest = 150        # Forecast radius
    max_cat_iters = 400     # iters for given N through cat of a subregion
    alpha = 0.4
    n_training = [25, 50, 100, 200, 280, 400]

    # Initialize result arrays
    np.random.seed(seed)
    random.seed(seed)

    prop_ranks = {i: np.zeros(3) for i in n_training}
    models = {i: [] for i in n_training}

    ssm_better = []
    ssm_worse = []
    ssm_undist = []

    ####################################
    # Prepare main run function with parameters
    main_func = partial(
        run_singleloc,
        catalog=catalog,
        n_max=nmax,
        r_test=rtest,
        r_train=rtrain,
        n_training=n_training,
        max_iters=max_cat_iters,
        alpha=alpha,
        ssm_class=ssm_class,
        kernel_size=ssm_param,
        bound_test_region=collection_region,
        bound_train_region=collection_region,
        return_models=True)

    ####################################
    # Run
    results = list(map(main_func, enumerate(centers)))

    ####################################
    # Re-order results

    for res in results:
        for i in n_training:
            if res[0][i].shape[0] == 3:
                prop_ranks[i] += res[0][i]
                models[i].extend(res[1][i])
    for n, rank in prop_ranks.items():
        ssm_worse.append(rank[0])
        ssm_undist.append(rank[1])
        ssm_better.append(rank[2])

    ####################################

    ssm_better = np.array(ssm_better)
    ssm_worse = np.array(ssm_worse)
    ssm_undist = np.array(ssm_undist)

    frac_better = ssm_better / (ssm_better + ssm_worse + ssm_undist)
    frac_worse = ssm_worse / (ssm_better + ssm_worse + ssm_undist)
    frac_undist = ssm_undist / (ssm_better + ssm_worse + ssm_undist)

    # Plot figures
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(n_training, np.zeros(len(n_training)),
                    frac_better, color='g', label='SSM is better', alpha=0.2)
    ax.plot(n_training, frac_better, 'g.--')

    ax.fill_between(n_training, frac_better, frac_better + frac_undist,
                    color='gray', label='Indistinguishable', alpha=0.2)
    ax.plot(n_training, 1 - frac_worse, 'r.--')
    ax.fill_between(n_training, frac_better + frac_undist,
                    frac_better + frac_undist + frac_worse,
                    color='r', label='URZ is better',
                    alpha=0.2)
    ax.set_title(f'Christchurch - Gaussian SSM', fontsize=16)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, max(n_training)])
    ax.set_xlabel('Number of training events $N_1$', fontsize=14)
    ax.set_ylabel('Fraction of forecasts', fontsize=14)

    plt.legend(loc=4, fontsize=12)
    if figpath:
        plt.savefig(figpath, dpi=600)
    print(f'ready. {time.process_time() - start:.1f}')
    plt.show()


def run_napier(figpath=None):

    start = time.process_time()

    # Initialize Catalog and Region
    ####################################
    collection_shp = paths.region_nz_collection
    collection_region = csep.regions.nz_csep_collection_region()
    catalog = catalogs.cat_oq2csep(
        catalogs.filter_cat(
            catalogs.get_cat_nz(name='New Zealand, Non-declustered'),
            mws=(3.99, 10.0), depth=[40, -2],
            start_time=dt(1980, 1, 1),
            shapefile=collection_shp))
    catalog.filter_spatial(collection_region, in_place=True)

    # Set experiment parameters
    ####################################
    centers = [paths.Napier[0]]

    seed = 14     # seed for reproducibility
    ssm_class = 'adaptive'   # model class: 'gaussian', 'powerlaw', 'adaptive'
    ssm_param = 0   # gaussian std; powerlaw d; adaptive N; Set 0 for autofit
    nmax = 5  # Testing events
    rtrain = 250       # Collection radius
    rtest = 150        # Forecast radius
    max_cat_iters = 500     # iters for given N through cat of a subregion
    alpha = 0.5
    n_training = [10, 25, 35, 50, 75, 100, 150, 200, 280, 400, 500]

    # Initialize result arrays
    np.random.seed(seed)
    random.seed(seed)

    prop_ranks = {i: np.zeros(3) for i in n_training}
    models = {i: [] for i in n_training}

    ssm_better = []
    ssm_worse = []
    ssm_undist = []

    ####################################
    # Prepare main run function with parameters
    main_func = partial(
            run_singleloc,
            catalog=catalog,
            n_max=nmax,
            r_test=rtest,
            r_train=rtrain,
            n_training=n_training,
            max_iters=max_cat_iters,
            alpha=alpha,
            ssm_class=ssm_class,
            kernel_size=ssm_param,
            bound_test_region=collection_region,
            bound_train_region=collection_region,
            return_models=True)

    ####################################
    # Run
    results = list(map(main_func, enumerate(centers)))

    ####################################
    # Re-order results

    for res in results:
        for i in n_training:
            if res[0][i].shape[0] == 3:
                prop_ranks[i] += res[0][i]
                models[i].extend(res[1][i])
    for n, rank in prop_ranks.items():
        ssm_worse.append(rank[0])
        ssm_undist.append(rank[1])
        ssm_better.append(rank[2])

    ####################################

    ssm_better = np.array(ssm_better)
    ssm_worse = np.array(ssm_worse)
    ssm_undist = np.array(ssm_undist)

    frac_better = ssm_better / (ssm_better + ssm_worse + ssm_undist)
    frac_worse = ssm_worse / (ssm_better + ssm_worse + ssm_undist)
    frac_undist = ssm_undist / (ssm_better + ssm_worse + ssm_undist)

    # Plot figures
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(n_training, np.zeros(len(n_training)),
                    frac_better, color='g', label='SSM is better', alpha=0.2)
    ax.plot(n_training, frac_better, 'g.--')

    ax.fill_between(n_training, frac_better, frac_better + frac_undist,
                    color='gray', label='Indistinguishable', alpha=0.2)
    ax.plot(n_training, 1 - frac_worse, 'r.--')
    ax.fill_between(n_training, frac_better + frac_undist,
                    frac_better + frac_undist + frac_worse,
                    color='r', label='URZ is better',
                    alpha=0.2)
    ax.set_title(f'Napier - Full catalog', fontsize=16)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, max(n_training)])
    ax.set_xlabel('Number of training events $N_1$', fontsize=14)
    ax.set_ylabel('Fraction of forecasts', fontsize=14)

    plt.legend(loc=4, fontsize=12)
    if figpath:
        plt.savefig(figpath, dpi=600)
    print(f'ready. {time.process_time() - start:.1f}')
    plt.show()


def run_napier_dc(figpath=None):

    start = time.process_time()

    # Initialize Catalog and Region
    ####################################
    collection_shp = paths.region_nz_collection
    collection_region = csep.regions.nz_csep_collection_region()
    catalog = catalogs.cat_oq2csep(
        catalogs.filter_cat(
            catalogs.get_cat_nz_dc(name='New Zealand, Non-declustered'),
            mws=(3.99, 10.0), depth=[40, -2],
            start_time=dt(1980, 1, 1),
            shapefile=collection_shp))
    catalog.filter_spatial(collection_region, in_place=True)

    # Set experiment parameters
    ####################################
    centers = [paths.Napier[0]]

    seed = 14     # seed for reproducibility
    ssm_class = 'adaptive'   # model class: 'gaussian', 'powerlaw', 'adaptive'
    ssm_param = 0   # gaussian std; powerlaw d; adaptive N; Set 0 for autofit
    nmax = 5  # Testing events
    rtrain = 250       # Collection radius
    rtest = 150        # Forecast radius
    max_cat_iters = 500     # iters for given N through cat of a subregion
    alpha = 0.5
    n_training = [10, 25, 35, 50, 75, 100, 150, 200, 300]

    # Initialize result arrays
    np.random.seed(seed)
    random.seed(seed)

    prop_ranks = {i: np.zeros(3) for i in n_training}
    models = {i: [] for i in n_training}

    ssm_better = []
    ssm_worse = []
    ssm_undist = []

    ####################################
    # Prepare main run function with parameters
    main_func = partial(
        run_singleloc,
        catalog=catalog,
        n_max=nmax,
        r_test=rtest,
        r_train=rtrain,
        n_training=n_training,
        max_iters=max_cat_iters,
        alpha=alpha,
        ssm_class=ssm_class,
        kernel_size=ssm_param,
        bound_test_region=collection_region,
        bound_train_region=collection_region,
        return_models=True)

    ####################################
    # Run
    results = list(map(main_func, enumerate(centers)))

    ####################################
    # Re-order results

    for res in results:
        for i in n_training:
            if res[0][i].shape[0] == 3:
                prop_ranks[i] += res[0][i]
                models[i].extend(res[1][i])
    for n, rank in prop_ranks.items():
        ssm_worse.append(rank[0])
        ssm_undist.append(rank[1])
        ssm_better.append(rank[2])

    ####################################

    ssm_better = np.array(ssm_better)
    ssm_worse = np.array(ssm_worse)
    ssm_undist = np.array(ssm_undist)

    frac_better = ssm_better / (ssm_better + ssm_worse + ssm_undist)
    frac_worse = ssm_worse / (ssm_better + ssm_worse + ssm_undist)
    frac_undist = ssm_undist / (ssm_better + ssm_worse + ssm_undist)

    # Plot figures
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(n_training, np.zeros(len(n_training)),
                    frac_better, color='g', label='SSM is better', alpha=0.2)
    ax.plot(n_training, frac_better, 'g.--')

    ax.fill_between(n_training, frac_better, frac_better + frac_undist,
                    color='gray', label='Indistinguishable', alpha=0.2)
    ax.plot(n_training, 1 - frac_worse, 'r.--')
    ax.fill_between(n_training, frac_better + frac_undist,
                    frac_better + frac_undist + frac_worse,
                    color='r', label='URZ is better',
                    alpha=0.2)
    ax.set_title(f'Napier - Declustered Catalog', fontsize=16)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, max(n_training)])
    ax.set_xlabel('Number of training events $N_1$', fontsize=14)
    ax.set_ylabel('Fraction of forecasts', fontsize=14)

    plt.legend(loc=4, fontsize=12)
    if figpath:
        plt.savefig(figpath, dpi=600)
    print(f'ready. {time.process_time() - start:.1f}')
    plt.show()


def run_example_ssm_napier(fig_path):

    start = time.process_time()

    ####################################
    # Initialize Catalog and Region
    collection_shp = paths.region_nz_collection
    collection_region = csep.regions.nz_csep_collection_region()
    catalog = catalogs.cat_oq2csep(
        catalogs.filter_cat(
            catalogs.get_cat_nz_dc(name='New Zealand, Non-declustered'),
            mws=(3.99, 10.0), depth=[40, -2],
            start_time=dt(1964, 1, 1),
            shapefile=collection_shp))
    catalog.filter_spatial(collection_region, in_place=True)

    ####################################
    # Set experiment parameters
    centers = [paths.Napier[0]]

    seed = 23     # seed for reproducibility
    ssm_class = 'adaptive'   # model class: 'gaussian', 'powerlaw', 'adaptive'
    ssm_param = 0   # gaussian std; powerlaw d; adaptive N; Set 0 for autofit
    nmax = 10   # Testing events
    rtrain = 200       # Collection radius
    rtest = 150        # Forecast radius
    max_cat_iters = 30     # iters for given N through cat of a subregion      #
    n_training = [50]

    ####################################
    # Initialize result arrays
    np.random.seed(seed)
    random.seed(seed)

    prop_ranks = {i: np.zeros(3) for i in n_training}
    models = {i: [] for i in n_training}

    ####################################
    # Prepare main run function with parameters
    main_func = partial(run_singleloc,
                        catalog=catalog,
                        n_max=nmax,
                        r_test=rtest,
                        r_train=rtrain,
                        n_training=n_training,
                        max_iters=max_cat_iters,
                        ssm_class=ssm_class,
                        # bound_test_region=collection_region,
                        bound_train_region=collection_region,
                        kernel_size=ssm_param,
                        return_models=True)

    ####################################
    # Run
    results = list(map(main_func, enumerate(centers)))

    ####################################
    # Re-order results
    for res in results:
        for i in n_training:
            if res[0][i].shape[0] == 3:
                prop_ranks[i] += res[0][i]
                models[i].extend(res[1][i])

    ####################################
    # Plot forecasts
    fig = plt.figure(figsize=(16, 8), layout='compressed')
    titles = ['SSM performs better than URZ',
              'URZ performs better than SSM',
              'Statistically indistinguishable']
    sns.reset_defaults()
    for n_fig, id_forecast in enumerate([3, 2, 15]):

        forecast_i = models[50][0][id_forecast]
        train_cat_i = models[50][1][id_forecast]
        test_cat_i = models[50][2][id_forecast]

        urz = GriddedForecast(region=forecast_i.region,
                              magnitudes=forecast_i.magnitudes,
                              data=np.ones(forecast_i.data.shape)/forecast_i.data.shape[0])
        urz.scale(test_cat_i.get_number_of_events())
        t_test = paired_t_test(forecast_i, urz, test_cat_i, alpha=0.2)
        dist = np.array(t_test.test_distribution)

        print(f'Forecast {id_forecast}. '
              f'Information Gain SSM from URZ conf. intervals {dist}')
        extent = np.array(forecast_i.region.get_bbox())
        extent = [extent[0] - 2.5, extent[1] + 2,
                  extent[2] - 2, extent[3] + 2]

        projection = cartopy.crs.epsg(2193)
        ax = fig.add_subplot(1, 3, n_fig + 1, projection=projection)
        raster_fn = paths.basemap_bluebrown_2193
        rds = rxr.open_rasterio(raster_fn, parse_coordinates=True)

        rds.plot.imshow(ax=ax, transform=projection, add_labels=False)
        ax = forecast_i.plot(ax=ax, plot_args={
            'basemap': None,
            'projection': projection,
            'grid': False,
            'cmap': 'rainbow',
            'include_cbar': True if n_fig == 0 else False,
            'clabel': r'       '
                      r'  $\log\mu/N_{test}$' if n_fig == 0 else False,
            'clabel_fontsize': 18,
            'coastline': False,
            'clim': [-2.5, -1.5]})
        ax.plot(*train_cat_i.region.tight_bbox().T,
                transform=cartopy.crs.PlateCarree(), color='steelblue',
                alpha=0.4, label='Collection regions')
        ax.plot(*csep.core.regions.nz_csep_collection_region().tight_bbox().T,
                transform=cartopy.crs.PlateCarree(), color='black',
                alpha=0.4)

        if n_fig == 0:
            cax = ax.get_figure().get_axes()[1]
            pos = cax.get_position().translated(-0.34, 0.15)
            cax.set_position(pos)

            legend_elements = [
                Line2D([0], [0], marker='s',  markeredgecolor='gray', lw=0,
                       markerfacecolor='none', label='NZ collection region',
                       markersize=16),
                Line2D([0], [0], marker='s', lw=0, markeredgecolor='steelblue',
                       markerfacecolor='none', label='SSM collection region',
                       markersize=16),
                Line2D([0], [0], marker='o',  markeredgecolor='black', lw=0,
                       markerfacecolor='lightgray', label='Training events',
                       markersize=16),
                Line2D([0], [0], marker='o',   markeredgecolor='black', lw=0,
                       markerfacecolor='red', label='Testing events',
                       markersize=16)]

            ax.legend(handles=legend_elements, loc=4, fontsize=14)

        ax = train_cat_i.plot(
            ax=ax,
            extent=extent,
            plot_args={
                'basemap': None,
                'markercolor': 'black',
                'grid': False,
                'legend': True if n_fig == 2 else False,
                'mag_ticks': [4.0, 5.0, 6.0] if n_fig == 2 else False,
                'mag_scale': 7,
                'markersize': 4,
                'legend_fontsize': 14,
                'legend_titlesize': 17,
                'coastline': False,})
        ax = test_cat_i.plot(ax=ax,
                             extent=extent,
                             plot_args={'basemap': None,
                                        'projection': projection,
                                        'markercolor': 'red',
                                        'mag_scale': 8,
                                        'alpha': 0.6,
                                        'grid': False,
                                        'markersize': 10,
                                        'legend': False,
                                        'coastline': False,
                                        })

        ax.set_title(titles[n_fig], fontsize=16)

    fig.savefig(fig_path, dpi=600)
    plt.show()
    sns.set_style("darkgrid",
                  {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})
    print(f'Ready: {time.process_time() - start: .2f}')


if __name__ == '__main__':

    # Calibrate global SSM models
    cat = catalogs.cat_oq2csep(
        catalogs.filter_cat(
            catalogs.get_cat_nz(name='New Zealand, non-declustered'),
            mws=(3.95, 10.0), depth=[40, -2],
            start_time=dt(1964, 1, 1),
            shapefile=paths.region_nz_collection))
    results_nondc = calibrate_models(cat)
    # results stored in 'nonpoisson_nz/results/temporal/fig'
    plot_fit_results(results_nondc, cat,
                     paths.get('temporal', 'fig'),
                     clim=[-4, 0])

    cat = catalogs.cat_oq2csep(
        catalogs.filter_cat(
            catalogs.get_cat_nz_dc(name='New Zealand, declustered'),
            mws=(3.95, 10.0), depth=[40, -2],
            start_time=dt(1964, 1, 1),
            shapefile=paths.region_nz_collection))
    results_dc = calibrate_models(cat)
    # results stored in 'nonpoisson_nz/results/temporal/fig'
    plot_fit_results(results_dc,  cat,
                     paths.get('temporal', 'fig'),
                     clim=[-4, -1],
                     figprefix='nzdc')

    # Run experiments and manuscript figures
    fig2_example_ssm = join(paths.ms1_figs['fig3'],
                            'example_ssm.png')
    run_example_ssm_napier(fig2_example_ssm)

    fig4a_napier_nondc = join(paths.ms1_figs['fig4'], 'napier_ssm_results.png')
    run_napier(fig4a_napier_nondc)

    fig4b_napier_dc = join(paths.ms1_figs['fig4'], 'napier_dc_ssm_results.png')
    run_napier_dc(fig4b_napier_dc)

    fig_christchurch = join(paths.ms1_figs['fig4'],
                            'christchurch_adaptiveresults.png')
    run_christchurch_adaptive(fig_christchurch)

    fig_christchurch = join(paths.ms1_figs['fig4'],
                            'christchurch_gaussian_results.png')
    run_christchurch_gaussian(fig_christchurch)


#
#
#
#     gauss_opt = [16.5, 18.9]
#     power_opt = [12.4, 12.4]
#     adapt_opt = [1, 7]
#
#     start = time.process_time()
#
#     # Initialize Catalog and Region
#     collection_shp = paths.region_nz_collection
#     collection_region = csep.regions.nz_csep_collection_region()
#     test_region = csep.regions.nz_csep_region()
#
#     catalog = catalogs.cat_oq2csep(
#         catalogs.filter_cat(
#             catalogs.get_cat_nz(name='New Zealand, Non-declustered'),
#             mws=(3.99, 10.0), depth=[40, -2],
#             start_time=dt(1964, 1, 1),
#             shapefile=collection_shp))
#     catalog.filter_spatial(collection_region, in_place=True)
#
#     # Set experiment parameters
#     nproc = 16  # cpu processes
#     seed = 23  # seed for reproducibility
#     ssm_class = 'gaussian'   # model class: 'gaussian', 'powerlaw', 'adaptive'
#     ssm_param = 17   # gaussian - std ;  powerlaw - dist ; adaptive - nearest N
#     ref_model = get_global_model(catalog, 'adaptive', 7)
#
#
#
#     nmax = 10   # Testing events
#     max_cat_iters = 200     # iters for given N through cat of a subregion
#     max_origin_iters = 300  # iters to create subregions from a region
#     # n_training = [20,  50,  100, 200]  # Number of training events
#     n_training = [20, 50, 85, 100,  200, 350, 500]  # Number of training events
#     # n_training = [ 200, 300]  # Number of training events
#
#     # max_cat_iters = 1     # iters for given N through cat of a subregion
#     # n_training = [150] # Number of training events
#     #
#     np.random.seed(seed)
#     random.seed(seed)
#
#     # Get randomn locations from the NZ region
#     # idxs = np.random.choice(range(test_region.num_nodes), max_origin_iters)
#     highseism = np.argwhere(ref_model.data < np.quantile(ref_model.data, 0.2))[:, 0]
#     idxs = np.random.choice(highseism, max_origin_iters)
#     centers = test_region.midpoints()[idxs]
#     print(f'Running {len(centers)} origins')
#     # centers = [paths.Wellington[0]]
#
#     # data = ref_model.data
#     # data[highseism] = 10
#     # ref_model.data = np.ones(ref_model.data.shape)
#     # ref_model.data[highseism] = 10
#     # fc = csep.core.forecasts.GriddedForecast(data=data,
#     #                                          region=test_region,
#     #                                          magnitudes=[5.0])
#
#
#     # Initialize result arrays
#     test_ranks = {i: [] for i in n_training}
#     prop_ranks = {i: np.zeros(3) for i in n_training}
#
#     test_fracs = {i: np.zeros(3) for i in n_training}
#
#     ssm_better = []
#     ssm_worse = []
#     ssm_undist = []
#
#     # Prepare main run function with parameters
#     main_func = partial(run_singleloc,
#                         catalog=catalog,
#                         model=ref_model,
#                         n_max=nmax,
#                         n_training=n_training,
#                         max_iters=max_cat_iters,
#                         ssm_class=ssm_class,
#                         kernel_size=ssm_param,
#                         plot=False)
#
#     # Run in parallel. Iterate through locations
#     if nproc > 0:
#         solver = Pool(processes=nproc)
#         results = solver.map(main_func, enumerate(centers))
#         solver.close()
#         solver.terminate()
#     # Run in serial
#     else:
#         results = list(map(main_func, centers))
#
# ####################################
#     # # Re-ordering of results
#     # for res in results:
#     #     if res:
#     #         for i in n_training:
#     #             test_ranks[i].extend(res[i])
#     # # Get Histograms
#     # for n, rank in test_ranks.items():
#     #     unique, counts = np.unique(rank, return_counts=True)
#     #     search = [-1, 0, 1]
#     #     search_counts = []
#     #     for i, j in enumerate(search):
#     #         if j in unique:
#     #             search_counts.append(counts[np.argwhere(unique == j)[0][0]])
#     #         else:
#     #             search_counts.append(0)
#     #
#     #     ssm_worse.append(search_counts[0])
#     #     ssm_undist.append(search_counts[1])
#     #     ssm_better.append(search_counts[2])
#
#
# ####################################
#     for res in results:
#
#         for i in n_training:
#             if res[i].shape[0] == 3:
#                 prop_ranks[i] += res[i]
#     for n, rank in prop_ranks.items():
#         ssm_worse.append(rank[0])
#         ssm_undist.append(rank[1])
#         ssm_better.append(rank[2])

####################################

    # ssm_better = np.array(ssm_better)
    # ssm_worse = np.array(ssm_worse)
    # ssm_undist = np.array(ssm_undist)
    #
    # frac_better = ssm_better / (ssm_better + ssm_worse + ssm_undist)
    # frac_worse = ssm_worse / (ssm_better + ssm_worse + ssm_undist)
    # frac_undist = ssm_undist / (ssm_better + ssm_worse + ssm_undist)
    #
    # # Plot figures
    # fig, ax = plt.subplots(1, 1)
    # ax.fill_between(n_training, np.zeros(len(n_training)),
    #                 frac_better, color='g',label='Better SSM', alpha=0.2)
    # ax.plot(n_training, frac_better, 'g.--')
    #
    # ax.fill_between(n_training, frac_better, frac_better + frac_undist,
    #                 color='gray', label='Undistinguishable', alpha=0.2)
    # ax.plot(n_training, 1- frac_worse, 'r.--')
    # ax.fill_between(n_training, frac_better + frac_undist,
    #                 frac_better + frac_undist + frac_worse,
    #                 color='r', label='Better URZ',
    #                 alpha=0.2)
    # ax.set_title(f'{ssm_class}-{ssm_param}-iters:{max_cat_iters}-'
    #              f'orig: {max_origin_iters}-nmax{nmax}-{catalog.name}')
    # now = datetime.datetime.now().time()
    # plt.savefig(f'test_{now.hour:02d}{now.minute:02d}{now.second:02d}.png')
    # print(f'ready. {time.process_time() - start:.1f}')
    #
    # plt.show()
    #
    #
    #



    # for i, j, k in zip(forecasts, test_cats, tests['ranks']):
    #     if k == -1:
    #         ax = i.plot(plot_args={'basemap': None, 'cmap': 'rainbow'})
    #         j.plot(ax=ax, plot_args={'basemap': None,
    #                                  'markercolor': 'black', 'markersize': 8},
    #                extent=i.region.get_bbox())
    #         plt.show()
    #
    # catalog = catalogs.cat_oq2csep(
    #     catalogs.filter_cat(
    #         catalogs.get_cat_ca(),
    #         mws=(3.95, 10.0), depth=[40, -2],
    #         start_time=dt(1964, 1, 1)))
    # region = csep.regions.california_relm_region()
    # centers = [paths.Wellington[0]]
    # ax = catalog.plot(plot_args={
    #     'markercolor': 'blue',
    #     'basemap': 'stock_img',
    #     'alpha': 0.1,
    #     'projection': cartopy.crs.Mercator(central_longitude=179)})
    # for i in centers:
    #     ax.plot(i[0], i[1], 'o', markersize=20, color='red',
    #             transform=cartopy.crs.PlateCarree())
    # plt.show()

# ax = A.uniform_forecast.plot()
# ax = catalog.plot(ax=ax, extent=collection_region.get_bbox(),
#                   plot_args={'basemap': None,
#                              'region_border': True})
# plt.show()
# ax = A.training_cat.plot(ax=ax, extent=collection_region.get_bbox(),
#                      plot_args={'markercolor': 'blue',
#                                 'markersize': 5,
#                                 'basemap': None,
#                                 'region_border': False})
# ax = A.test_cat.plot(ax=ax, extent=collection_region.get_bbox(),
#                      plot_args={'markercolor': 'black',
#                                 'alpha': 1,
#                                 'markersize': 8,
#                                 'basemap': None,
#                                 'region_border': False})
# ax.plot(*csep.core.regions.nz_csep_region().tight_bbox().T,
#         transform=cartopy.crs.PlateCarree())
# print(f'N_uni: {A.uniform_forecast.event_count}')
# plt.show()
# for k in forecasts:
#     ax = k.plot()
#     print(f'N_SSM: {k.event_count}')
#     ax = catalog.plot(ax=ax, extent=collection_region.get_bbox(),
#                       plot_args={'basemap': None, 'region_border': True})
#     ax.plot(*csep.core.regions.nz_csep_region().tight_bbox().T,
#             transform=cartopy.crs.PlateCarree())
#     plt.show()
