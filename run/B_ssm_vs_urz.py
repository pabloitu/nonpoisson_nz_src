import pickle
import time
import cartopy
from nonpoisson import catalogs, paths
from datetime import datetime as dt
import matplotlib.pyplot as plt
from openquake.hazardlib.geo.point import Point
import numpy as np
from openquake.hmtk.seismicity import catalogue, selector, utils
import copy
import random
import time
from multiprocessing import Pool
import cartopy.crs as ccrs
import csep
from csep.core.regions import CartesianGrid2D
from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import MarkedGriddedDataSet, GriddedForecast
from csep.core.poisson_evaluations import paired_t_test
from csep.utils.plots import plot_comparison_test
from csep.utils.stats import poisson_joint_log_likelihood_ndarray
import datetime
import itertools
import seaborn as sns


sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})



def log_score(forecast, catalog):
    forecast_data = forecast.data
    observed_data = catalog.spatial_magnitude_counts()
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
def dist_p2grid(geom):
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

class Forecast:

    def __init__(self, id, n_disc, cat, grid):
        self.id = id
        self.n_disc = n_disc
        self.training_cat = cat
        self.grid = grid
        self.kernel_d = 50

    def get_ssm(self):
        pass


class Experiment:

    def __init__(self, catalogue, n, point, forecast_radius, test_radius):

        self.n = n
        # assert self.n >= 10, 'Not enough number of earthquakes. Dont be cheap, and choose 10'
        self.point = point
        self.forecast_radius = forecast_radius
        self.test_radius = test_radius
        self.catalogue = catalogue
        self.test_n = 10

        self.training_grid = None
        self.training_region = None
        self.training_cat = None
        self.test_grid = None
        self.test_region = None
        self.test_cat = None

        self.dists = None
        self.dists_cat = None

    def nz_grid(self):
        self.training_region = csep.regions.nz_csep_collection_region(
            magnitudes=np.array([4.]))
        self.test_region = csep.regions.nz_csep_region(
            magnitudes=np.array([4.]))

    def create_grids(self, extent=[155, -55, 200, -15], dh=0.1):

        x, y = [coord.ravel() for coord in np.meshgrid(np.arange(extent[0] + dh/2, extent[2] + dh/2, 0.1),
                                                       np.arange(extent[1] + dh/2, extent[3] + dh/2, 0.1))]
        grid_geom = np.vstack((np.deg2rad(x), np.cos(np.deg2rad(y)), np.sin(np.deg2rad(y)))).T
        dists = np.array(dist_p2grid((np.deg2rad(self.point), grid_geom)))

        self.training_grid = np.vstack((x[dists <= self.forecast_radius] - dh / 2,
                                        y[dists <= self.forecast_radius] - dh / 2)).T
        self.test_grid = np.vstack((x[dists <= self.test_radius] - dh/2,
                                    y[dists <= self.test_radius] - dh/2)).T
        self.training_region = CartesianGrid2D.from_origins(self.training_grid, dh=0.1,
                                                            magnitudes=np.array([4.]))
        self.test_region = CartesianGrid2D.from_origins(self.test_grid, dh=0.1,
                                                        magnitudes=np.array([4.]))

    @classmethod
    def load(cls, filename=None):

        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def get_distances(self, nproc=8):

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
        catalog_points = np.deg2rad(np.array([[i, j]
                                              for i, j in zip(self.training_cat.data['longitude'],
                                                              self.training_cat.data['latitude'])]))

        print("Calculate distances")
        cat_loc_geom = np.vstack((catalog_points[:, 0],
                                  np.cos(catalog_points[:, 1]),
                                  np.sin(catalog_points[:, 1]))).T

        Input_list = [[i, cat_loc_geom] for i in grid_points]

        if nproc != 0:
            pool = Pool(nproc)
            A = np.array(pool.map(dist_p2grid, Input_list))
            pool.close()
            pool.join()
        else:
            A = np.array(list(map(dist_p2grid, Input_list)))

        self.dists = A[:]
        print("Processing time: %.1f seconds" % (time.process_time() - start))

    def get_distances_cat(self, nproc=8):

        """
        Fast paralellized function that calculates the distances between two
        set of points.

        Output
        --------
        self.dists[(attr, att2)] : array
            Distance matrix of shape n × m.

        """

        start = time.process_time()
        catalog_points = np.deg2rad(np.array([[i, j]
                                              for i, j in zip(self.training_cat.data['longitude'],
                                                              self.training_cat.data['latitude'])]))

        print("Calculate distances")
        cat_loc_geom = np.vstack((catalog_points[:, 0],
                                  np.cos(catalog_points[:, 1]),
                                  np.sin(catalog_points[:, 1]))).T

        Input_list = [[i, cat_loc_geom] for i in catalog_points]

        if nproc != 0:
            pool = Pool(nproc)
            A = np.array(pool.map(dist_p2grid, Input_list))
            pool.close()
            pool.join()
        else:
            A = np.array(list(map(dist_p2grid, Input_list)))

        self.dists_cat = A[:]
        print("Processing time: %.1f seconds" % (time.process_time() - start))

    def get_ssm(self, cat_inds, model='gaussian', KernelSize=50.,
                power=1.5,
                nearest=5,
                dist_cutoff=5):

        """
        Fixed-sized Kernel smoothing from Helmstetter et al, 2007
        Modified from func_kernelfix.m developed by Hiemer, S. in matlab

        Input

        - subcat (string): Name of the sub-catalog to calculate
        - KernelSize (float): Fix distance of the smoothing kernel
        - power (float): 1.0: Wang-Kernel, 1.5: Helmstetter-Kernel
        - mc_min (float): min magnitude used for completeness correction
        - mag_scaling (bool): Ponderates kernel by moment of event
        - area_norm (boolean): Flag to normalize each cell by its area
        - sum_norm (boolean): Flag to normalize every cell by the total sum of
                                of all cells
        - nproc (int): Number of processes for parallelization. If 0, no
                             parallelization scheme is used.
        - memclean (bool): Removes the cat2grid distance matrix

        Output
        - pdfx_catalog_fk[subcat] (dtype=array, shape=(m,)):
            Probability density for each grid cell, as the sum of every catalog
            event contribution.

        """

        pdfX = np.zeros(self.test_region.midpoints().shape[0])

        if model == 'power':
            for i in self.dists.T[cat_inds, :]:
                kernel_i = 1. / ((i ** 2 + KernelSize ** 2) ** power)
                kernel_i /= (np.sum(kernel_i))
                pdfX += kernel_i

        elif model == 'gaussian':
            # KernelSize = KernelSize/2.  # Radius
            for i in self.dists.T[cat_inds, :]:
                kernel_i = 1/np.sqrt(2*np.pi)/KernelSize*np.exp(-i**2/(2*KernelSize**2))
                kernel_i /= (np.sum(kernel_i))
                pdfX += kernel_i

        elif model == 'adapt':

            N = nearest
            # Calculate distances between all events
            d_cat2cat = self.dists_cat[cat_inds,:][:, cat_inds]
            kernel_size = np.sort(d_cat2cat).T[N, :]
            if dist_cutoff:
                kernel_size[kernel_size < 5] = 5

            for i, j in zip(self.dists.T[cat_inds, :],
                               kernel_size):
                kernel_i = 1./((i**2 + j**2)**power)
                kernel_i /= (np.sum(kernel_i))
                pdfX += kernel_i

        pdfX /= np.sum(pdfX)

        return pdfX[:]

    def filter_catalogs(self):
        self.training_cat = self.catalogue.filter_spatial(
            region=self.training_region,
            update_stats=True,
            in_place=False)
        assert np.all(np.diff(self.training_cat.get_epoch_times()) > 0)
        self.test_cat = self.catalogue.filter_spatial(region=self.test_region,
                                                      update_stats=True,
                                                      in_place=False)
        self.test_cat.filter([r'magnitude >= 5.0'])
        assert np.all(np.diff(self.test_cat.get_epoch_times()) > 0)

    def get_uniform_forecast(self):

        data = np.zeros((self.test_region.midpoints().shape[0],
                         self.test_region.magnitudes.shape[0]))
        data[:, 0] = self.test_n / self.test_region.midpoints().shape[0]   ## Spatial pdf set only in first magnitude bin
        self.uniform_forecast = GriddedForecast(region=self.test_region,
                                                magnitudes=self.test_region.magnitudes,
                                                data=data)

    def create_forecast(self, n, max_iters=10,
                        ssm='power',
                        show_plots=False,
                        KernelSize=50.0,
                        nearest=5):

        n_cat = self.training_cat.get_number_of_events()
        training_inds = []
        test_inds = []

        first_testevent_time = self.test_cat.get_epoch_times()[-self.test_n]

        for i in range(max_iters):
            max_id = np.sum(self.training_cat.get_epoch_times() < first_testevent_time)
            index = random.choice(range(max_id - n))
            forecast_inds = [index + j for j in range(n)]
            last_time = self.training_cat.get_epoch_times()[forecast_inds[-1]]
            first_t_ind = np.min(np.argwhere(self.test_cat.get_epoch_times() > last_time))
            test_ind = np.arange(first_t_ind, first_t_ind + 10, 1)
            training_inds.append(forecast_inds)
            test_inds.append(test_ind)

        while len(training_inds) < max_iters:
            index = random.choice(range(1, n_cat - n))
            forecast_inds = [index + j for j in range(n)]
            last_time = self.training_cat.get_epoch_times()[forecast_inds[-1]]
            if np.sum(self.test_cat.get_epoch_times() > last_time) >= self.test_n:
                first_t_ind = np.min(np.argwhere(self.test_cat.get_epoch_times() > last_time))
                test_ind = np.arange(first_t_ind, first_t_ind + 10, 1)
                training_inds.append(forecast_inds)
                test_inds.append(test_ind)

        cat_test_i = copy.deepcopy(self.test_cat)
        cat_training_i = copy.deepcopy(self.training_cat)
        ssm_better = []


        for n, (f_ind, t_ind) in enumerate(zip(training_inds, test_inds)):
            start = time.process_time()
            pdf = self.get_ssm(f_ind, model=ssm, KernelSize=KernelSize)
            forecast = GriddedForecast(region=self.test_region,
                                       magnitudes=self.test_region.magnitudes,
                                       data=pdf.reshape(-1, 1) * self.test_n)

            cat_test_i.catalog = self.test_cat.catalog[t_ind]
            cat_training_i.catalog = self.training_cat.catalog[f_ind]


            if show_plots:
                ax = forecast.plot()
                ax = cat_training_i.plot(ax=ax, plot_args={'markercolor': 'blue',
                                                            'basemap':'stock_img'})
                ax = cat_test_i.plot(ax=ax, plot_args={'markercolor': 'red', 'markersize':5,
                                                       'basemap': 'stock_img'})

                plt.show()

            t_test = paired_t_test(forecast,
                                   self.uniform_forecast,
                                   cat_test_i,
                                   alpha=0.4)
            dist = np.array(t_test.test_distribution)
            if (dist > 0).all():    # Statistically better
                ssm_better.append(1)
            elif (dist < 0).all():  # Statistically worse
                ssm_better.append(-1)
            else:  # Statistically indistinguishable
                ssm_better.append(0)

        return ssm_better

    def save(self, filename=None):
        """
        Serializes Model_results object into a file
        :param filename: If None, save in results folder named with self.name
        """

        with open(filename, 'wb') as obj:
            pickle.dump(self, obj, protocol=None,
                        fix_imports=True, buffer_callback=None)


def calibrate_gaussian(training_end=datetime.datetime(2010, 1, 1)):
    cat = catalogs.filter_cat(catalogs.get_cat_nz(),
                                 mws=(3.95, 10.0), depth=[40, -2],
                                 start_time=dt(1964, 1, 1),
                                 shapefile=paths.region_nz_collection)

    cat_csep = catalogs.cat_oq2csep(cat)
    Exp = Experiment(cat_csep, None, None, None, None)
    Exp.nz_grid()
    Exp.filter_catalogs()
    Exp.get_distances(nproc=0)
    Exp.get_distances_cat(nproc=0)

    indx_training = np.sum(Exp.training_cat.get_epoch_times() <
                           training_end.timestamp()*1000)
    indx_testing = np.sum(Exp.test_cat.get_epoch_times() <
                           training_end.timestamp()*1000)

    Exp.test_cat.catalog = Exp.test_cat.catalog[indx_testing:]
    N_total = Exp.test_cat.get_number_of_events()

    gauss_stds = np.linspace(8, 60, 20)
    gaussian_forecasts = []
    gaussian_log_scores = []

    for gauss_std in gauss_stds:
        pdf = Exp.get_ssm(np.arange(0, indx_training),
                          model='gaussian',
                          KernelSize=gauss_std)
        forecast = GriddedForecast(region=Exp.test_region,
                                   magnitudes=Exp.test_region.magnitudes,
                                   data=pdf.reshape(-1, 1) * N_total)
        gaussian_forecasts.append(forecast)
        gaussian_log_scores.append(log_score(forecast, Exp.test_cat))
    gaussian_log_scores = np.array(gaussian_log_scores)

    cat_dc = catalogs.filter_cat(catalogs.get_cat_nz_dc(),
                                 mws=(3.95, 10.0), depth=[40, -2],
                                 start_time=dt(1964, 1, 1),
                                 shapefile=paths.region_nz_collection)
    cat_csep = catalogs.cat_oq2csep(cat_dc)
    Exp = Experiment(cat_csep, None, None, None, None)
    Exp.nz_grid()
    Exp.filter_catalogs()
    Exp.get_distances(nproc=0)
    Exp.get_distances_cat(nproc=0)
    indx_training = np.sum(Exp.training_cat.get_epoch_times() <
                           training_end.timestamp()*1000)
    indx_testing = np.sum(Exp.test_cat.get_epoch_times() <
                          training_end.timestamp()*1000)

    Exp.test_cat.catalog = Exp.test_cat.catalog[indx_testing:]
    N_total = Exp.test_cat.get_number_of_events()

    gaussian_forecasts_dc = []
    gaussian_log_scores_dc = []
    for gauss_std in gauss_stds:
        pdf = Exp.get_ssm(np.arange(0, indx_training),
                          model='gaussian',
                          KernelSize=gauss_std)
        forecast = GriddedForecast(region=Exp.test_region,
                                   magnitudes=Exp.test_region.magnitudes,
                                   data=pdf.reshape(-1, 1) * N_total)
        gaussian_forecasts_dc.append(forecast)
        gaussian_log_scores_dc.append(log_score(forecast, Exp.test_cat))
    gaussian_log_scores_dc = np.array(gaussian_log_scores_dc)

    plt.axvline(gauss_stds[gaussian_log_scores.argmax()], color='k',
                linestyle='--')
    plt.plot(gauss_stds, gaussian_log_scores, 'o-')
    plt.plot(gauss_stds[gaussian_log_scores.argmax()],
             gaussian_log_scores.max(), "^",
             label=f'Optimal $\sigma$:'
                   f' {gauss_stds[gaussian_log_scores.argmax()]:.1f}',
             markersize=13)
    plt.xlabel('Gaussian smoothing $\sigma$ [km]')
    plt.ylabel('Log-Likelihood')
    plt.title('Gaussian SSM fit - NZ Non-Declustered')
    plt.legend()
    plt.savefig(paths.get('temporal', 'fig', 'gaussian_fit'), dpi=300)
    plt.show()

    plt.axvline(gauss_stds[gaussian_log_scores_dc.argmax()],
                color='k', linestyle='--')
    plt.plot(gauss_stds, gaussian_log_scores_dc, 'o-')
    plt.plot(gauss_stds[gaussian_log_scores_dc.argmax()],
             gaussian_log_scores_dc.max(), "^",
             label=f'Optimal $\sigma$:'
                   f' {gauss_stds[gaussian_log_scores_dc.argmax()]:.1f}',
             markersize=13)
    plt.legend()
    plt.xlabel('Gaussian smoothing $\sigma$ [km]')
    plt.ylabel('Log-Likelihood')
    plt.title('Gaussian SSM fit - NZ Declustered')
    plt.savefig(paths.get('temporal', 'fig', 'gaussian_fit_dc'), dpi=300)
    plt.show()

    return gauss_stds[gaussian_log_scores_dc.argmax()], \
           gauss_stds[gaussian_log_scores.argmax()]


def calibrate_power(training_end=datetime.datetime(2010, 1, 1)):
    cat = catalogs.filter_cat(catalogs.get_cat_nz(),
                              mws=(3.95, 10.0), depth=[40, -2],
                              start_time=dt(1964, 1, 1),
                              shapefile=paths.region_nz_collection)

    cat_csep = catalogs.cat_oq2csep(cat)
    Exp = Experiment(cat_csep, None, None, None, None)
    Exp.nz_grid()
    Exp.filter_catalogs()
    Exp.get_distances(nproc=0)
    Exp.get_distances_cat(nproc=0)

    indx_training = np.sum(Exp.training_cat.get_epoch_times() <
                           training_end.timestamp()*1000)
    indx_testing = np.sum(Exp.test_cat.get_epoch_times() <
                          training_end.timestamp()*1000)
    Exp.test_cat.catalog = Exp.test_cat.catalog[indx_testing:]
    N_total = Exp.test_cat.get_number_of_events()

    dists = np.linspace(5, 25, 20)
    forecasts = []
    log_scores = []

    for dist in dists:
        pdf = Exp.get_ssm(np.arange(0, indx_training),
                          model='power',
                          KernelSize=dist)
        forecast = GriddedForecast(region=Exp.test_region,
                                   magnitudes=Exp.test_region.magnitudes,
                                   data=pdf.reshape(-1, 1) * N_total)
        forecasts.append(forecast)
        log_scores.append(log_score(forecast, Exp.test_cat))
    log_scores = np.array(log_scores)

    #### Declustered analysis
    cat_dc = catalogs.filter_cat(catalogs.get_cat_nz_dc(),
                                 mws=(3.95, 10.0), depth=[40, -2],
                                 start_time=dt(1964, 1, 1),
                                 shapefile=paths.region_nz_collection)
    cat_csep = catalogs.cat_oq2csep(cat_dc)
    Exp = Experiment(cat_csep, None, None, None, None)
    Exp.nz_grid()
    Exp.filter_catalogs()
    Exp.get_distances(nproc=0)
    Exp.get_distances_cat(nproc=0)
    indx_training = np.sum(Exp.training_cat.get_epoch_times() <
                           training_end.timestamp()*1000)
    indx_testing = np.sum(Exp.test_cat.get_epoch_times() <
                          training_end.timestamp()*1000)

    Exp.test_cat.catalog = Exp.test_cat.catalog[indx_testing:]
    N_total = Exp.test_cat.get_number_of_events()

    forecasts_dc = []
    log_scores_dc = []
    for dist in dists:
        pdf = Exp.get_ssm(np.arange(0, indx_training),
                          model='power',
                          KernelSize=dist)
        forecast = GriddedForecast(region=Exp.test_region,
                                   magnitudes=Exp.test_region.magnitudes,
                                   data=pdf.reshape(-1, 1) * N_total)
        forecasts_dc.append(forecast)
        log_scores_dc.append(log_score(forecast, Exp.test_cat))
    log_scores_dc = np.array(log_scores_dc)


    plt.axvline(dists[log_scores.argmax()], color='k',
                linestyle='--')
    plt.plot(dists, log_scores, 'o-')
    plt.plot(dists[log_scores.argmax()],
             log_scores.max(), "^",
             label=f'Optimal $d$:'
                   f' {dists[log_scores.argmax()]:.1f}',
             markersize=13)
    plt.xlabel('Power-law smoothing $d$ [km]')
    plt.ylabel('Log-Likelihood')
    plt.title('Power-law SSM fit - NZ Non-Declustered')
    plt.legend()
    plt.savefig(paths.get('temporal', 'fig', 'powerlaw_fit'), dpi=300)
    plt.show()

    plt.axvline(dists[log_scores_dc.argmax()],
                color='k', linestyle='--')
    plt.plot(dists, log_scores_dc, 'o-')
    plt.plot(dists[log_scores_dc.argmax()],
             log_scores_dc.max(), "^",
             label=f'Optimal $d$:'
                   f' {dists[log_scores_dc.argmax()]:.1f}',
             markersize=13)
    plt.legend()
    plt.xlabel('Power-law smoothing $d$ [km]')
    plt.ylabel('Log-Likelihood')
    plt.title('Power-law SSM fit - NZ Declustered')
    plt.savefig(paths.get('temporal', 'fig', 'powerlaw_fit_dc'), dpi=300)
    plt.show()

    return dists[log_scores_dc.argmax()], \
           dists[log_scores.argmax()]


def calibrate_adaptive(training_end=datetime.datetime(2010, 1, 1)):
    cat = catalogs.filter_cat(catalogs.get_cat_nz(),
                              mws=(3.95, 10.0), depth=[40, -2],
                              start_time=dt(1964, 1, 1),
                              shapefile=paths.region_nz_collection)

    cat_csep = catalogs.cat_oq2csep(cat)
    Exp = Experiment(cat_csep, None, None, None, None)
    Exp.nz_grid()
    Exp.filter_catalogs()
    Exp.get_distances(nproc=0)
    Exp.get_distances_cat(nproc=0)

    indx_training = np.sum(Exp.training_cat.get_epoch_times() <
                           training_end.timestamp()*1000)
    indx_testing = np.sum(Exp.test_cat.get_epoch_times() <
                          training_end.timestamp()*1000)
    Exp.test_cat.catalog = Exp.test_cat.catalog[indx_testing:]
    N_total = Exp.test_cat.get_number_of_events()

    neighbors = np.arange(1, 20)
    forecasts = []
    log_scores = []

    for neighbor in neighbors:
        pdf = Exp.get_ssm(np.arange(0, indx_training),
                          model='adapt',
                          nearest=neighbor)
        forecast = GriddedForecast(region=Exp.test_region,
                                   magnitudes=Exp.test_region.magnitudes,
                                   data=pdf.reshape(-1, 1) * N_total)
        forecasts.append(forecast)
        log_scores.append(log_score(forecast, Exp.test_cat))
    log_scores = np.array(log_scores)

    #### Declustered analysis
    cat_dc = catalogs.filter_cat(catalogs.get_cat_nz_dc(),
                                 mws=(3.95, 10.0), depth=[40, -2],
                                 start_time=dt(1964, 1, 1),
                                 shapefile=paths.region_nz_collection)
    cat_csep = catalogs.cat_oq2csep(cat_dc)
    Exp = Experiment(cat_csep, None, None, None, None)
    Exp.nz_grid()
    Exp.filter_catalogs()
    Exp.get_distances(nproc=0)
    Exp.get_distances_cat(nproc=0)
    indx_training = np.sum(Exp.training_cat.get_epoch_times() <
                           training_end.timestamp()*1000)
    indx_testing = np.sum(Exp.test_cat.get_epoch_times() <
                          training_end.timestamp()*1000)

    Exp.test_cat.catalog = Exp.test_cat.catalog[indx_testing:]
    N_total = Exp.test_cat.get_number_of_events()

    forecasts_dc = []
    log_scores_dc = []

    for neighbor in neighbors:
        pdf = Exp.get_ssm(np.arange(0, indx_training),
                          model='adapt',
                          nearest=neighbor)
        forecast = GriddedForecast(region=Exp.test_region,
                                   magnitudes=Exp.test_region.magnitudes,
                                   data=pdf.reshape(-1, 1) * N_total)
        forecasts_dc.append(forecast)
        log_scores_dc.append(log_score(forecast, Exp.test_cat))


    log_scores_dc = np.array(log_scores_dc)

    plt.axvline(neighbors[log_scores.argmax()], color='k',
                linestyle='--')
    plt.plot(neighbors, log_scores, 'o-')
    plt.plot(neighbors[log_scores.argmax()],
             log_scores.max(), "^",
             label=f'Optimal $K$:'
                   f' {neighbors[log_scores.argmax()]}',
             markersize=13)
    plt.xlabel('Nearest-neighbor smoothing $K$')
    plt.ylabel('Log-Likelihood')
    plt.title('Adaptive SSM fit - NZ Non-Declustered')
    plt.legend()
    plt.savefig(paths.get('temporal', 'fig', 'adaptive_fit'), dpi=300)
    plt.show()

    plt.axvline(neighbors[log_scores_dc.argmax()],
                color='k', linestyle='--')
    plt.plot(neighbors, log_scores_dc, 'o-')
    plt.plot(neighbors[log_scores_dc.argmax()],
             log_scores_dc.max(), "^",
             label=f'Optimal $K$:'
                   f' {neighbors[log_scores_dc.argmax()]}',
             markersize=13)
    plt.legend()
    plt.xlabel('Nearest-neighbor smoothing $K$')
    plt.ylabel('Log-Likelihood')
    plt.title('Adaptive SSM fit - NZ Declustered')
    plt.savefig(paths.get('temporal', 'fig', 'adaptive_fit_dc'), dpi=300)
    plt.show()


    return neighbors[log_scores_dc.argmax()], \
        neighbors[log_scores.argmax()]



if __name__ == '__main__':

    # gauss_opt = calibrate_gaussian()
    # power_opt = calibrate_power()
    # adapt_opt = calibrate_adaptive()

    # gauss_opt = [16.5, 18.9]
    # power_opt = [12.4, 12.4]
    # adapt_opt = [1, 7]

    n = np.arange(25, 450, 100)
    ssm_better = []
    ssm_worse = []
    ssm_undist = []

    cat_oq = catalogs.filter_cat(catalogs.get_cat_nz(),
                                 mws=(3.95, 10.0), depth=[40, -2],
                                 shapefile=paths.region_nz_collection)
    catalog = catalogs.cat_oq2csep(cat_oq)

    center = (177.3, -39.38)

    centers = [paths.Gisborne[0],
               # paths.Auckland[0],
               paths.Wellington[0],
               paths.Gisborne[0],
               paths.Queenstown[0],
               paths.Christchurch[0],
               # paths.Dunedin[0],
               paths.Tauranga[0]]

    # ax = catalog.plot(plot_args={'markercolor': 'blue', 'basemap': 'stock_img', 'alpha':0.1,
    #                         'projection': cartopy.crs.Mercator(central_longitude=179)})
    # for i in centers:
    #     ax.plot(i[0], i[1], 'o', markersize=20, color='red', transform=cartopy.crs.PlateCarree())
    # plt.show()

    np.random.seed(123)
    for center in centers:

        ssm_better_x = []
        ssm_worse_x = []
        ssm_undist_x = []

        A = Experiment(catalog, 10, center, 200., 150.)
        A.create_grids()
        A.filter_catalogs()
        A.get_distances(nproc=0)
        A.get_distances_cat(nproc=0)
        A.get_uniform_forecast()

        max_iters = 50

        for i in n:
            if i % 25 == 0:
                print('N_1: %i' % (i))

            ssm_performance = A.create_forecast(n=i,
                                                ssm='gaussian',
                                                KernelSize=18.9,
                                                nearest=7,
                                                max_iters=max_iters,
                                                show_plots=False)
            id_, counts = np.unique(ssm_performance, return_counts=True)
            assert (np.diff(id_) > 0).all()

            if -1 in id_:
                ssm_worse_n = counts[np.argwhere(id_ == -1)[0, 0]]
            else:
                ssm_worse_n = 0
            if 1 in id_:
                ssm_better_n = counts[np.argwhere(id_ == 1)[0, 0]]
            else:
                ssm_better_n = 0

            if 0 in id_:
                undist_n = counts[np.argwhere(id_ == 0)[0, 0]]
            else:
                undist_n = 0


            ssm_better_x.append(ssm_better_n)
            ssm_worse_x.append(ssm_worse_n)
            ssm_undist_x.append(undist_n)

        total = np.array(ssm_better_x) + np.array(ssm_worse_x) + np.array(ssm_undist_x)
        ssm_better.append(ssm_better_x/total)
        ssm_worse.append(ssm_worse_x/total)
        ssm_undist.append(ssm_undist_x/total)

        # ssm_better.append(ssm_better_x)
        # ssm_worse.append(ssm_worse_x)
        # ssm_undist.append(ssm_undist_x)

    ssm_better = np.array(ssm_better).sum(axis=0)
    ssm_worse = np.array(ssm_worse).sum(axis=0)
    ssm_undist = np.array(ssm_undist).sum(axis=0)


    frac_better = ssm_better / (ssm_better + ssm_worse + ssm_undist)
    frac_worse = ssm_worse / (ssm_better + ssm_worse + ssm_undist)
    frac_undist = ssm_undist / (ssm_better + ssm_worse + ssm_undist)

    plt.fill_between(n, np.zeros(len(n)), frac_better, color='g',label='Better SSM', alpha=0.2)
    plt.fill_between(n, frac_better, 1-frac_worse, color='gray', label='Better URZ', alpha=0.2)
    plt.fill_between(n, np.ones(len(n)), 1 - frac_worse, color='r', label='Undistinguishable',
                     alpha=0.2)
    # print('sim over')
    # # sum_ranking = [i + j + k for i, j, k in zip(frac_better, frac_undist, frac_worse)]
    # print(sum_ranking)
    # print(np.sum(sum_ranking)/len(sum_ranking))
    #
    # plt.plot(n, frac_better, 'g.-', label='Better SSM')
    # plt.plot(n, frac_worse, 'r.-', label='Better URZ')
    # plt.plot(n, frac_undist, color='gray', linestyle='-.',
    #          label='Undistinguishable')
    plt.legend()
    plt.show()
