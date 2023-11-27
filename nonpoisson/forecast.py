import shutil
from nonpoisson import temporal
from nonpoisson import paths
from nonpoisson import geo
from osgeo import gdal
import alphashape
import shapely

import itertools
import scipy as sp
import time
from lxml import etree
import csep
from csep.utils import time_utils
import cartopy
import numpy as np
import random
import psutil
import scipy.stats as st
import os
import datetime
import matplotlib.pyplot as plt
from shutil import copyfile
import openquake

from shapely.geometry import Polygon as shpPoly
from shapely.geometry import shape
from openquake.hmtk.seismicity import selector
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.source.rupture import BaseRupture
from openquake.hazardlib.scalerel.point import PointMSR
from openquake.hazardlib.geo.geodetic import geodetic_distance
from openquake.hazardlib.source.non_parametric import NonParametricSeismicSource
from openquake.hazardlib.source.point import PointSource
from openquake.hazardlib.geo.utils import OrthographicProjection
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.surface import PlanarSurface
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.geo import Polygon as oqpoly
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.scalerel.point import PointMSR
from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD
from openquake.hazardlib.mfd.evenly_discretized import EvenlyDiscretizedMFD
from openquake.hazardlib import sourcewriter, sourceconverter
from datetime import datetime as dt
from openquake.hazardlib.tom import PoissonTOM, NegativeBinomialTOM
from os.path import join
from os import makedirs
import gc
from multiprocessing import Pool
from csep.core.regions import geographical_area_from_bounds
import seaborn
import copy


def get_tvzpolygon(spatial=False, metric='j2', bin=3):

    if spatial:
        polygons = spatial.get_oqpolygons(metric, bin)
        return polygons[0][int(np.argmax(polygons[1]))]

    coords = np.genfromtxt(paths.region_tvz)
    shell_sphe = [tuple(i) for i in coords]
    proj = OrthographicProjection.from_lons_lats(np.array([i[0] for i in shell_sphe]),
                                                 np.array([i[1] for i in shell_sphe]))
    poly = shpPoly(shell=np.array(proj(*np.array(shell_sphe).T)).T)

    return oqpoly._from_2d(poly, proj)


def deg_dist(geom):
    """
    Calculate geodesic distance, using small_angle hyphotesis, between a point
    and an array of points.
    Input:
        - geom[0] (dtype=list/array, shape=(2,)):
                    Single point to evaluate distances (in deg)
        - geom[1] (dtype=list/array, shape=(2,n)):
                    Single point or list of points to evaluate distances
                    (in deg)
    """
    precision = 12

    x1 = [np.deg2rad(geom[0][0]), np.deg2rad(geom[0][1])]
    x2 = [[np.deg2rad(i) for i in geom[1][0]],
          [np.deg2rad(i) for i in geom[1][1]]]
    dtheta = np.sin(x1[1]) * np.sin(x2[1]) + \
             np.cos(x1[1]) * np.cos(x2[1]) * np.cos(x2[0] - x1[0])
    distances = 6371. * np.arccos(np.round(dtheta, precision))

    return distances


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
    # global d_type
    precision = 12
    dtheta = (np.sin(geom[0][1]) * geom[1][:, 2] +
              np.cos(geom[0][1]) * geom[1][:, 1] *
              np.cos(geom[1][:, 0] - geom[0][0]))

    row = 6371. * np.arccos(np.round(dtheta, precision))

    #    return row
    return row.astype('f8')


def findCells(points, bounds):
    points = points.reshape((-1, 2))

    allInBounds = (points[:, 0] >= bounds[:, None, 0])
    allInBounds &= (points[:, 1] >= bounds[:, None, 1])
    allInBounds &= (points[:, 0] < bounds[:, None, 2])
    allInBounds &= (points[:, 1] < bounds[:, None, 3])

    nz = np.nonzero(allInBounds)
    r = np.full(points.shape[0], np.nan)
    r[nz[1]] = nz[0]

    return r


def findCellsParallel(points, bounds, chunksize=100):
    func = partial(findCells, bounds=bounds)
    p = Pool()
    try:
        return np.hstack(p.map(func, points, chunksize))
    finally:
        p.close()



class ssmModel(object):
    """
    Class used to represent a seismicity & faults Smoothed Seismicity Model.
        ...


    Attributes
    - - - - - - - -
    name :  str
        the name of the model
    toc : datetime.datetime()
        the time of creation of the model
    catalog : list
        the complete seismicity catalog
    subcatalog : list
        subset of the complete catalog, filtered by depth, shapefile, time or
        specificied events
    grid : array
        grid of observation to produce the forecast
    s_probdens : array
        probability density field of the class's grid and subcatalog
    f_probdens : array
        probability density field of the class's grid and discretized faults


    Methods
    - - - - - - - -
    set_catalog(filename, db_format)
        Loads the complete catalog in specific format
    set_faults(filename, db_format)
        Loads the fault-base equivalent catalog
    set_grid(filename)
        Loads the observation grid to calculate the smoothed-seismicity model
    set_subcatalog(name, specific=None, depth=None, nan_inc=False,
                   polygon=None, time=None)
        Filters the complete catalog
    get_catalogSSM(self, cat_name='full', N=2, power=1.5, target_mw=5.0,
                   nproc=16)
        Calculates the SSM spatial smoothing based only in seismicity catalog
        Creates a normalized probability density field
    get_faultSSM(KernelSize=10., N=2, power=1.5, pond=1, nproc=16)
        Calculates the SSM spatial smoothing based only in faults
        Creates a normalized probability density field
    get_magPDF(Mw_max, a, b, Mw_target, bin_size=0.1)
        Calculates the SSM magnitude smoothing based only in seismicity
        statistics. Construct a tapered-GR curve
    save_txt(filename, att, fmt='%.18e', header='',delimiter=' ')
        Save into a txtfile a model result


    """

    def __init__(self, name):

        self.name = name
        self.toc = datetime.datetime.now()
        self.catalog = []
        self.faults = []
        self.dists = {}
        self.subcatalog = {}
        self.pdfx_catalog_vk = {}
        self.pdfx_catalog_fk = {}
        self.testgrid = {}
        self.omega = {}
        self.G_r = {}  # Calibration of fixed kernel
        self.G_n = {}  # Calibration of variable kernel
        self.mask_faults = {}  # Mask of fault pdf cutoff contribution
        self.seismic_weight = {}  # Weighing calibration

        # ============================= #

    #     Input loading methods     #
    # ============================= #

    def set_catalog(self, oq_catalog):

        """
        Load complete seismic catalog for the model
        ### Columns: [lon, lat, year, month, day, magnitude, depth,
        ###           hour, minute, second,
        ###           time_of_completeness, mag_of_completeness]  <<<< Deprecated, but function left in case a future correction of completeness is neeeded, bu
        """
        longitude = oq_catalog['longitude']
        latitude = oq_catalog['latitude']
        depth = oq_catalog['depth']
        datetime_ = [datetime.datetime(y, mo, d, h, m, int(s)) for y, mo, d, h, m ,s in zip(oq_catalog['year'],
                                                                                            oq_catalog['month'],
                                                                                            oq_catalog['day'],
                                                                                            oq_catalog['hour'],
                                                                                            oq_catalog['minute'],
                                                                                            oq_catalog['second'])]

        magnitude = oq_catalog['magnitude']

        self.catalog = [[lon, lat,  # Epicenter position
                         dt,
                         10 ** (1.5 * mw + 9.0),  # Magnitude (N · m)
                         d,  # depth (km)
                         1., 4.0]
                        # Event corrected completeness   ----------- Set HARDCODED for Mw_min = 5.0, and when Mw_forecsated_min = 5.0 as well.
                        for lon, lat, dt, mw, d in zip(longitude, latitude, datetime_, magnitude, depth )]
        self.subcatalog['full'] = {'cat': self.catalog[:],
                                   'ind': [i for i
                                           in range(len(self.catalog))]}

    def set_grid(self, grid, dh=0.1, mask=False):

        """
        Load the model's grid.
        """
        ### Pending: Provide flexibility for the mesh area and mask setting

        ### Input columns: [left-bot-corner lon, left-bot-corner lat,
        ###                 center lon, center lat, cell area]

        self.maskflag = mask

        area = np.array([geographical_area_from_bounds(x-dh/2, y-dh/2, x+dh/2, y+dh/2) for x, y in grid])  # km2
        self.grid = np.vstack((grid.T, area)).T
        if mask:
            self.gridmask = self.grid[:, -1]
            self.grid_int = self.grid[self.gridmask == 1]
        else:
            self.gridmask = np.ones(self.grid.shape[0])
            self.grid_int = self.grid

    # ============================= #
    #     Data handling methods     #
    # ============================= #

    def set_subcatalog(self, name, specific=None, minmag=None, maxmag=None,
                       depth=None, nan_inc=False, bbox=None, polygon=None,
                       time=None):

        """
        Filters the model catalog and creates a subcatalog. It filters by
        a specific index list, a depth range, a polygon shapefile a bound-box
        and a time window.

        Input
        -----
        specific : array/list
            Simply filters the catalog by an array of index
        depth : tuple/list
            Minimum and maximum depth windows (in negative values)
        nan_inc : boolean
            Flag to include nan values of depth
        bbox : list of two tuples/lists
            x and y ranges of the bounding box [(x_min,x_max),(y_min,y_max)]
        polygon : string
            Filepath of the shapefile to crop the catalog
        time : tuple/list of datetime.datetime() objects
            Time range to crop the catalog. Provide datetime.datetime.now() to
            unbound by above and datetime.datetime(0,0,0) by below

        Output
        -----
        self.subcatalog[name] : dict of two lists
            Creates two lists with the cropped catalog, and the corresponding
            indexes to the full catalog
        """
        if specific:
            subcatalog = [self.catalog[i] for i in specific]
        else:
            subcatalog = self.catalog[:]

        if minmag:
            subcatalog = [i for i in subcatalog if (np.log10(i[3]) - 9.) / 1.5
                          >= minmag]
        if maxmag:
            subcatalog = [i for i in subcatalog if (np.log10(i[3]) - 9.) / 1.5
                          <= maxmag]

        if depth:
            if nan_inc:
                subcatalog = [i for i in subcatalog
                              if depth[0] <= i[4] < depth[1]
                              or np.isnan(i[4])]
            else:
                subcatalog = [i for i in subcatalog
                              if depth[0] <= i[4] < depth[1]]

        if bbox:
            subcatalog = [i for i in subcatalog
                          if bbox[0][0] <= i[0] <= bbox[0][1]
                          and bbox[1][0] <= i[1] <= bbox[1][1]]

        if polygon:
            shpfile = fiona.open(polygon)
            poly = next(iter(shpfile))
            subcatalog = [i for i in subcatalog
                          if Point(i[0], i[1]).within(
                    shape(poly['geometry']))]

        if time:
            subcatalog = [i for i in subcatalog
                          if time[0] <= i[2] < time[1]]

        indexes = [self.catalog.index(i) for i in subcatalog
                   if i in self.catalog[:]]

        self.subcatalog[name] = {'cat': subcatalog, 'ind': indexes}

    def set_testgrid(self, res, mask=False):

        """
        Redefines grid in terms of each cell boundaries, so testing can be
        performed,
        Input:
            res (tuple): x and y resolution

        Output:
            self.testgrid['bounds']: redefined grid
            self.testgrid['res']: resolution
            self.testgrid['n']: number of cells in x and y directions
            self.testgrid['bbox']: expanded grid onto a rectangle.

        """

        if isinstance(mask, (list, np.ndarray)):
            grid = self.grid_int[mask == 1]
        elif mask == False:
            grid = self.grid_int
        else:
            raise Exception('Mask object type not understood')
        testgrid = []

        x = np.arange(grid[:, 0].min() - res[0] / 2.,
                      grid[:, 0].max() + 3. / 2. * res[0], res[0])
        y = np.arange(grid[:, 1].min() - res[1] / 2.,
                      grid[:, 1].max() + 3. / 2. * res[1], res[1])

        for i in grid:
            testgrid.append([i[0] - res[0] / 2., i[1] - res[1] / 2.,
                             i[0] + res[0] / 2., i[1] + res[1] / 2.])

        self.testgrid['bounds'] = np.array(testgrid[:])
        self.testgrid['res'] = res
        self.testgrid['n'] = (x.shape[0] - 1, y.shape[0] - 1)
        self.testgrid['bbox'] = np.meshgrid(x, y)

    # ============================= #
    #   Basic calculation methods   #
    # ============================= #

    def set_distances(self, att1, att2, drop=False, dtype='f', nproc=8):

        """
        Fast paralellized function that calculates the distances between two
        set of points.

        Input
        -------
        att1 :  str
            Name of the first class attribute, which must contain its position
            (i.e. 'catalog', 'grid', 'faults'). The referenced array must be
            of shape (n, >=2)
        att1 :  str
            Name of the second class attribute (i.e. 'catalog', etc.). The
            referenced array must be of shape (m, >=2)
        drop : bool
            Flag to drop the distance results into a hdf5 database, which could
            be loaded later for better performance
        nproc : int
            Number of processes for multiprocessing.Pool.map()

        Output
        --------
        self.dists[(attr, att2)] : array
            Distance matrix of shape n × m.

        """

        start = time.process_time()
        pointarray1 = np.deg2rad(np.array([[i[0], i[1]]
                                           for i in getattr(self, att1)]))
        pointarray2 = np.deg2rad(np.array([[i[0], i[1]]
                                           for i in getattr(self, att2)]))

        print("Calculate distances between " + att1 + ' and ' +
              att2 + "\n\t number of points: %i × %i" % (pointarray1.shape[0],
                                                         pointarray2.shape[0]))

        geom2 = np.vstack((pointarray2[:, 0],
                           np.cos(pointarray2[:, 1]),
                           np.sin(pointarray2[:, 1]))).T

        Input_list = [[i, geom2] for i in pointarray1]

        if nproc != 0:
            pool = Pool(nproc)
            A = np.array(pool.map(dist_p2grid, Input_list))
            pool.close()
            pool.join()

        if drop:

            filename = self.dir['store'] + 'd_' + att1 + '_' + att2
            with h5py.File(filename, 'w') as f:
                dset = f.create_dataset('dist', A.shape, dtype=dtype,
                                        chunks=True)
                dset[:A.shape[0], :A.shape[1]] = A
            f.close()
            del A
            gc.collect()
        else:
            self.dists[(att1, att2)] = A[:]
            del A
            gc.collect()

        print("Processing time: %.1f seconds" % (time.process_time() - start))

    def get_discrete_eqk_field(self, subcat):

        """
        Counts the number of earthquakes that occurs within each cell of the
        test grid.
        Input:
            subcat (string): name of the subcatalog
        Output:
            omega (array): 1-array of ints containing the cell earthquake count

        """

        testgrid = self.testgrid['bounds'][:]
        catalog = np.array(self.subcatalog[subcat]['cat'][:])

        omega = np.zeros(testgrid.shape[0])

        if catalog.shape[0] != 0:
            r = findCellsParallel(catalog[:, :2], testgrid)
        else:
            r = np.full(testgrid.shape, np.nan)
        results = np.unique(r, return_counts=True)
        inside = False == np.isnan(results[0])
        omega[results[0][inside].astype('int')] = results[1][inside]

        self.omega[subcat] = omega
        return omega

        # ============================== #

    #   Model construction methods   #
    # ============================== #

    def get_catalogSSM_varKernel(self, subcat='full',
                                 N=2, power=1.5,
                                 mc_min=4.5, dist_cutoff=0.05,
                                 area_norm=True, sum_norm=True,
                                 nproc=16, memclean=False):

        """
        Variable-sized Kernel smoothing from Helmstetter et al, 2007
        Modified from func_kernelvariable.m developed by Hiemer, S. in matlab

        Input
        - subcat (string): Name of the sub-catalog to calculate
        - N (int): Sorting position of the closest event for each
                         earthquake, which controlls initial Kernel Size
        - power (float): 1.0: Wang-Kernel, 1.5: Helmstetter-Kernel
        - mc_min (float): min magnitude used for completeness correction
        - area_norm (boolean): Flag to normalize each cell by its area
        - sum_norm (boolean): Flag to normalize every cell by the total sum of
                                of all cells
        - nproc (int): Number of processes for parallelization. If 0, no
                             parallelization scheme is used.
        - memclean (bool): Removes the cat2grid distance matrix

        Output
        - pdfx_catalog_vk[subcat] (dtype=array, shape=(m,)):
            Probability density for each grid cell, as the sum of every catalog
            event contribution.

        """

        # Initialize
        start = time.process_time()
        catalog = np.array(self.subcatalog[subcat]['cat'][:])
        cat_ind = np.array(self.subcatalog[subcat]['ind'][:])

        print("Starting catalog variable-size kernel smoothing\n" +
              "\tsub-catalog used: " + subcat + "\n" +
              "\tnumber of events: %i\n" % catalog.shape[0] +
              "\tnumber of cells: %i\n" % self.grid.shape[0] +
              "\tnumber of masked cells: %i\n" % len(self.grid_int) +
              "\tnumber of processes: %i\n" % nproc)

        # Calculate distances between all events
        d_cat2cat = self.dists[('catalog', 'catalog')][cat_ind, :][:, cat_ind]
        kernel_size = np.sort(d_cat2cat).T[N, :]
        if dist_cutoff:
            kernel_size[kernel_size < dist_cutoff] = dist_cutoff
        catalog[np.where(catalog[:, 6] >= mc_min), 6] = mc_min

        pdfX = np.zeros(self.grid.shape[0])

        for i, j, k in zip(self.dists[('catalog', 'grid')][cat_ind, :],
                           kernel_size, catalog[:, 5:]):
            # print(i, j)
            kernel_i = 1. / ((i.astype('f8') ** 2 + j.astype('f8') ** 2) ** power)
            kernel_i /= np.sum(kernel_i) / (10 ** (k[1] - mc_min) / k[0])
            pdfX += kernel_i

        pdfX = pdfX[self.gridmask == 1]

        if area_norm:
            pdfX /= self.grid_int[:, 2]

        if sum_norm:
            pdfX /= np.sum(pdfX)

        if memclean:
            self.dists.pop(('catalog', 'grid'))

        self.pdfx_catalog_vk[subcat] = pdfX[:]

        print("Catalog variable-size kernel smoothing complete.\n\
              \ttime taken: %i seconds \n" % (time.process_time() - start) +
              "\tmemory use: %.1f\n\n" % psutil.virtual_memory()[2])

    def get_catalogSSM_fixKernel(self, subcat='full', KernelSize=10.,
                                 power=1.5, area_norm=True, sum_norm=True,
                                 nproc=16, mag_scaling=False, mc_min=False,
                                 memclean=False, dist_cutoff=0.):

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

        # Initialize
        start = time.process_time()
        catalog = np.array(self.subcatalog[subcat]['cat'][:])
        cat_ind = np.array(self.subcatalog[subcat]['ind'][:])
        if mc_min:
            catalog[np.where(catalog[:, 6] >= mc_min), 6] = mc_min

        print("Starting catalog fixed-size kernel smoothing\n" +
              "\tnumber of events: %i\n" % catalog.shape[0] +
              "\tnumber of cells: %i\n" % self.grid.shape[0] +
              "\tnumber of masked cells: %i\n" % len(self.grid_int) +
              "\tnumber of processes: %i\n" % nproc +
              "\tmemory use: %.1f\n\n" % psutil.virtual_memory()[2])

        # Calculate distances between all events

        pdfX = np.zeros(self.grid.shape[0])
        for i, j in zip(self.dists[('catalog', 'grid')][cat_ind, :],
                        catalog[:, [3, 5, 6]]):
            dist = i.astype('f8')
            for d in range(dist.shape[0]):
                if dist[d] < dist_cutoff:
                    print('a')
                    dist[d] = dist_cutoff

            kernel_i = 1. / ((dist ** 2 + KernelSize ** 2) ** power)
            kernel_i /= (np.sum(kernel_i))

            if mag_scaling:
                kernel_i *= j[0]
            if mc_min:
                kernel_i *= (10 ** (j[2] - mc_min) / j[1])
            pdfX += kernel_i

        pdfX = pdfX[self.gridmask == 1]
        if area_norm:
            pdfX /= self.grid_int[:, 2]

        if sum_norm:
            pdfX /= np.sum(pdfX)

        if memclean:
            self.dists.pop(('catalog', 'grid'))
        self.pdfx_catalog_fk[subcat] = pdfX[:]

        print("Catalog fixed-size kernel smoothing complete.\n" +
              "\t time taken: %i seconds \n" % (time.process_time() - start) +
              "\tmemory use: %.1f\n\n" % psutil.virtual_memory()[2])

    # ====================================== #
    #   Model spatial calibration methods    #
    # ====================================== #

    def calibrate_fixKernel(self, training_cat, target_cat, r_disc,
                            model_area_norm=True, ref_area_norm=False,
                            mag_scaling=False, Mc_min=False,
                            plot=False):
        """
        Calibrates the fixed kernel method in terms of the smoothing distance
        using the Log-likelihood method.
        Input:
            training_cat (str): name of the catalog used to create the models
            target_cat (str): name of the catalog used to perform the retro-
                spective testing
            r_disc (list): values of r (smoothing distance) to create the models
            model_area_norm (bool): Normalize the cells spatial pdf by its area
            ref_area_norm (bool): normalize the reference model by cell area
            mag_scaling (bool): Ponderates the kernel by the event magnitude
            Mc_min (bool): Corrects the kernel by the catalog completeness
        Output:
            self.G_r[(training_cat,target_cat)] (dict):
                Dictionary containg the results of the calibration for the sub-
                catalogs pair. Contains the r_discretization, the joint log
                likelihod of the ref model (L_uni) and the fix kernel model (L)
                along with the Probability gain of the model (G)
        """
        logL = {}
        G = {}
        omega = self.get_discrete_eqk_field(target_cat)

        Nt = np.sum(omega)

        n_cells = self.grid_int.shape[0]
        pdfx_uniform = 1. / n_cells * np.ones(n_cells)
        if ref_area_norm:
            pdfx_uniform /= self.grid_int[:, 2]
            pdfx_uniform /= np.sum(pdfx_uniform)
        pdfx_uniform *= Nt

        logL_uni = 0
        for i in range(omega.shape[0]):
            logL_uni += -pdfx_uniform[i] + \
                        omega[i] * np.log10(pdfx_uniform[i]) - \
                        np.log10(scp.factorial(omega[i]))

        for r in r_disc:
            logL[r] = 0

            self.get_catalogSSM_fixKernel(subcat=training_cat,
                                          KernelSize=r,
                                          area_norm=model_area_norm,
                                          mag_scaling=mag_scaling,
                                          mc_min=Mc_min)

            mu = Nt * self.pdfx_catalog_fk[training_cat][:]

            for i in range(omega.shape[0]):
                logL[r] += -mu[i] + omega[i] * np.log10(mu[i]) - \
                           np.log10(scp.factorial(omega[i]))

            G[r] = 10 ** ((logL[r] - logL_uni) / Nt)

        y = np.array([G[i] for i in r_disc])
        self.G_r[(training_cat, target_cat)] = {'r': r_disc,
                                                'r_opt': r_disc[y.argmax()],
                                                'L_ref': logL_uni,
                                                'L': logL,
                                                'G': G}

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.plot(r_disc, y, 'o-')
            plt.plot(r_disc[y.argmax()], y.max(), "^", color='orange', markersize=13)
            plt.axvline(r_disc[y.argmax()], linestyle='--', color='orange')
            plt.set_title('Fixed-Kernel calibration', fontsize=16)
            textbox = 'Training cat: %s' % training_cat + r' $N_0=%i$' % len(self.subcatalog[training_cat]['cat']) + \
                      '\nTarget cat: %s' % target_cat + r' $N_t=%i$' % Nt
            props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
            plt.text(0.7, 0.95, textbox, transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)
            if len(r_disc) > 1:
                plt.xticks([int(i) for i in r_disc])
                plt.grid()
            plt.xlabel('Smoothing distance \n' +
                       r'$r \,[\mathrm{km}]$', fontsize=13)
            plt.ylabel(' Information Gain per earthquake \n' +
                       r'$G = \frac{L - L_0}{N_t}$', fontsize=13)
            plt.tight_layout()


        return G, r_disc[y.argmax()]

    def calibrate_varKernel(self, training_cat, target_cat, n_disc,
                            model_area_norm=True, ref_area_norm=False,
                            Mc_min=4.0, plot=False):

        """
        Calibrates the variable kernel method in terms of the smoothing
        distance using the Log-likelihood method.
        Input:
            training_cat (str): name of the catalog used to create the models
            target_cat (str): name of the catalog used to perform the retro-
                spective testing
            n_disc (list): values of N (nearest neighbors) to create the models
            model_area_norm (bool): Normalize the cells spatial pdf by its area
            ref_area_norm (bool): normalize the reference model by cell area
            Mc_min (bool): Corrects the kernel by the catalog completeness
        Output:
            self.G_n[(training_cat,target_cat)] (dict):
                Dictionary containg the results of the calibration for the sub-
                catalogs pair. Contains the n discretization, the joint log
                likelihod of the ref model (L_uni) and the fix kernel model (L)
                along with the Probability gain of the model (G)
        """

        logL = {}
        G = {}
        omega = self.get_discrete_eqk_field(target_cat)
        Nt = np.sum(omega)

        n_cells = self.grid_int.shape[0]
        pdfx_uniform = 1. / n_cells * np.ones(n_cells)
        if ref_area_norm:
            pdfx_uniform /= self.grid_int[:, 2]
            pdfx_uniform /= np.sum(pdfx_uniform)
        pdfx_uniform *= Nt

        logL_uni = 0
        for i in range(omega.shape[0]):
            logL_uni += -pdfx_uniform[i] + \
                        omega[i] * np.log10(pdfx_uniform[i]) - \
                        np.log10(scp.factorial(omega[i]))

        for n in n_disc:
            logL[n] = 0
            self.get_catalogSSM_varKernel(subcat=training_cat,
                                          N=n,
                                          mc_min=Mc_min,
                                          area_norm=model_area_norm)

            mu = Nt * self.pdfx_catalog_vk[training_cat][:]

            for i in range(omega.shape[0]):
                logL[n] += -mu[i] + omega[i] * np.log10(mu[i]) - \
                           np.log10(scp.factorial(omega[i]))

            G[n] = 10 ** ((logL[n] - logL_uni) / Nt)

        y = np.array([G[i] for i in n_disc])
        self.G_n[(training_cat, target_cat)] = {'N': n_disc,
                                                'N_opt': n_disc[y.argmax()],
                                                'L_ref': logL_uni,
                                                'L': logL,
                                                'G': G}
        if plot:

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.plot(n_disc, y, 'o-')
            ax.plot(n_disc[y.argmax()], y.max(), "^", color='orange', markersize=13)
            ax.axvline(n_disc[y.argmax()], linestyle='--', color='orange')
            ax.set_title('Adaptive-Kernel calibration', fontsize=16)
            textbox = 'Training cat: %s' % training_cat + r' $N_0=%i$' % len(self.subcatalog[training_cat]['cat']) + \
                      '\nTarget cat: %s' % target_cat + r' $N_t=%i$' % Nt
            props = dict(boxstyle='round', facecolor='gray', alpha=0.5)

            ax.text(0.7, 0.95, textbox, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            if len(n_disc) > 1:
                plt.xticks(n_disc)
                plt.grid()
            plt.xlabel('Nearest Neighbor \n' +
                       r'$N $', fontsize=13)
            plt.ylabel(' Information Gain per earthquake \n' +
                       r'$G = \frac{L - L_0}{N_t}$', fontsize=13)
            plt.tight_layout()


        return G, n_disc[y.argmax()]


class forecastModel(object):

    def __init__(self, name, folder='', time_span=None, mkdirs=True):


        self.name = name
        self.time_span = time_span
        self.model_path = paths.get_model('forecast', folder, name)
        if mkdirs:
            self.dirs = {'output': join(self.model_path, 'output'),
                         'vti': join(self.model_path, 'vti'),
                         'figures': join(self.model_path, 'figures'),
                         'forecast': join(self.model_path, 'forecast')}

            for dir_ in self.dirs.values():
                makedirs(dir_, exist_ok=True)

        self.cells_source = None
        self.cells_area = None
        self.total_area = None
        self.grid_source = None
        self.grid_obs = None

        self.poly2point_map = None
        self.nsources = None
        self.nobs = None
        self.magnitudes = None

        self.primary_keys = ['poly',        # Polygon ID
                             'bin',         # Bin of spatial model
                             'eventcount',  # Events within polygon
                             'rate_learning', # Events divided cell size
                             'bval',
                             'lmmin',
                             'mmin',
                             'mmax',
                             'model',       # Temporal model
                             'params',      # Parameter of point temporal model
                             'rates_bin',  # Events divided cell size
                             'rate',  # Rates per mag bin
                             ]

        print(f'Model folder: {self.name}')

    @property
    def target_rate(self):
        return np.sum([np.array([i['params']]).flatten()[0] for i in self.params_primary.values()])

    @property
    def target_rate_bins(self):
        return np.sum([i['rates_bin'] for i in self.params_primary.values() if i['rate'] is not None])

    @property
    def learning_rate(self):
        return np.sum([i['rate_learning'] for i in self.params_primary.values() if i['rate'] is not None])

    @property
    def mean_rate(self):
        return np.sum([i['rate'] for i in self.params_primary.values() if i['rate'] is not None])

    @staticmethod
    def get_polygon_counts(oqpolygons, catalog):

        poly_events = []
        for poly in oqpolygons:

            func = selector.CatalogueSelector(catalog)
            cut_cat = func.within_polygon(poly)
            if cut_cat.end_year and cut_cat.get_number_events() > 0:
                nevents = cut_cat.get_number_events()
            else:
                nevents = 0
            poly_events.append(nevents)
        return poly_events

    def get_rate(self, point):
        if isinstance(point, str):
            point = getattr(paths, point)
        if np.argwhere(np.all(np.isclose(self.grid_source, point), axis=1)).shape[0]:
            point = np.argwhere(np.all(np.isclose(self.grid_source, point), axis=1))[0, 0]
        else:
            point = np.argmin(np.sum((self.grid_source - point) ** 2, axis=1))
        return self.params_primary[point]['params'][0]


    def get_geometry(self, grid='csep_testing'):

        if grid == 'NSHM10':
            model_nshm10 = openquake.hazardlib.nrml.read(paths.nshm10_simp)
            conv = sourceconverter.SourceConverter()
            model = [node for node in conv.convert_node(model_nshm10[0][0]) if node]
            raise Exception('Cell Area method not yet implemented')
            self.nsources = len(model)
            self.grid_source = np.array([[point.location.x, point.location.y] for point in model])
            self.params_primary = {i: dict.fromkeys(self.primary_keys) for i in range(self.nsources)}
            self.params_secondary = {i: dict.fromkeys(self.secondary_keys) for i in range(self.nsources)}

        if grid == 'csep_testing' or grid == 'eepas':
            self.cells = np.genfromtxt(paths.csep_testing, delimiter=',')
            self.cells_area = np.array([geographical_area_from_bounds(i[0], i[2], i[1], i[3]) for i in self.cells])
            self.total_area = np.sum(self.cells_area)
            self.grid_source = self.cells[:, (0, 2)] + 0.05
            self.nsources = self.grid_source.shape[0]
            self.params_primary = {i: dict.fromkeys(self.primary_keys) for i in range(self.nsources)}

        if isinstance(grid, forecastModel):
            self.cells = copy.deepcopy(grid.cells)
            self.cells_area = copy.deepcopy(grid.cells_area)
            self.total_area = copy.deepcopy(grid.total_area)
            self.grid_source = copy.deepcopy(grid.grid_source)
            self.nsources = copy.deepcopy(grid.nsources)
            self.params_primary = {i: dict.fromkeys(self.primary_keys) for i in range(self.nsources)}

        elif isinstance(grid, np.ndarray):
            self.grid_source = grid
            self.nsources = grid.shape[0]
            self.params_primary = {i: dict.fromkeys(self.primary_keys) for i in range(self.nsources)}

    @staticmethod
    def truncated_GR(n, bval, mag_bin, learning_mmin, target_mmin, target_mmax):
        aval = np.log10(n) + bval * target_mmin
        mag_bins = np.arange(learning_mmin, target_mmax + mag_bin, mag_bin)
        mag_weights = (10 ** (aval - bval * (mag_bins - mag_bin / 2.)) - 10 ** (aval - bval * (mag_bins + mag_bin / 2.))) / \
                      (10 ** (aval - bval * (learning_mmin - mag_bin/2.)) - 10 ** (aval - bval * (target_mmax + mag_bin/2.)))
        return aval, mag_bins, mag_weights

    def set_rate_from_models(self, temporal_model, spatial_model, catalog, years=None,
                             measure='j2', nbins=4, target_mmin=5.0,
                             target_mmax=8.0):

        print('Creating Analytical Hazard from models')
        catalog_span = catalog.end_year - catalog.end_year + 1
        if years:
            self.time_span = years
        else:
            self.time_span = catalog_span

        learn_mmin = min(catalog.data['magnitude'])

        oq_polygons, poly_bin = spatial_model.get_oqpolygons(measure, nbins)
        nevents = self.get_polygon_counts(oq_polygons, catalog)
        polygons_area = [poly._polygon2d.area for poly in oq_polygons]

        # Create grid with offset, so it does not collide with polygon objects
        offset = 0.0001
        mesh = Mesh(self.grid_source[:, 0] + offset, self.grid_source[:, 1] + offset)

        # Get the indices of points located within each polygon
        poly_point_map = []
        for id_poly, poly in enumerate(oq_polygons):
            ids = np.argwhere(poly.intersects(mesh)).ravel()
            poly_point_map.append(ids)
        self.poly2point_map = poly_point_map
        print(f'Polygon events: {[i for i in nevents if i != 0]}')

        c_i = 0
        c_a = 0
        for i, (poly2point, eventcount, poly_area, bin) in enumerate(zip(poly_point_map, nevents, polygons_area, poly_bin)):

            if len(poly2point) == 0:
                continue

            if eventcount in temporal_model.params_local.keys():
                point_model = temporal_model.__class__.__name__
                params = temporal_model.params_local[eventcount]

            else:
                point_model = temporal.backwardPoisson
                params = eventcount
            if poly_area < 15000:
                point_model = temporal.backwardPoisson
                params = eventcount
            for point in poly2point:
                rate_density = eventcount * self.cells_area[point] / np.sum(self.cells_area[poly2point]) * self.time_span/catalog_span

                if point_model == 'NegBinom':
                    mu_i = params[2]*self.cells_area[point] / np.sum(self.cells_area[poly2point]) * self.time_span/catalog_span
                    alpha = params[3]
                    params_i = [mu_i, alpha]
                else:
                    params_i = [rate_density, 0]
                self.params_primary[point]['poly'] = i
                self.params_primary[point]['bin'] = bin
                self.params_primary[point]['eventcount'] = eventcount
                self.params_primary[point]['rate_learning'] = rate_density
                self.params_primary[point]['lmmin'] = learn_mmin
                self.params_primary[point]['mmin'] = target_mmin
                self.params_primary[point]['mmax'] = target_mmax
                self.params_primary[point]['model'] = point_model
                self.params_primary[point]['params'] = params_i
                self.params_primary[point]['mfd'] = None

    def get_rate_from_hybrid(self, path_model, target_mmin=5.0, target_mmax=8.05):

        model = np.genfromtxt(path_model, delimiter=',')

        if model.shape[1] > 15:
            for point, rows in enumerate(model):

                mag_rates = rows[2:]
                rate_density = np.sum(rows[2:])
                mfd = None

                self.params_primary[point]['poly'] = 1
                self.params_primary[point]['eventcount'] = 1
                self.params_primary[point]['rate_learning'] = rate_density
                self.params_primary[point]['rate'] = rate_density
                self.params_primary[point]['mfd'] = mfd
                self.params_primary[point]['lmmin'] = target_mmin
                self.params_primary[point]['mmin'] = target_mmin
                self.params_primary[point]['mmax'] = target_mmax
                self.params_primary[point]['model'] = 1
                self.params_primary[point]['params'] = [rate_density, 0]
                self.params_primary[point]['rates_bin'] = mag_rates

        else:
            for point, rows in enumerate(model):
                mag_rates = rows[8:9]
                rate_density = np.sum(rows[8:9])
                mfd = None

                self.params_primary[point]['poly'] = 1
                self.params_primary[point]['bin'] = 1
                self.params_primary[point]['eventcount'] = 1
                self.params_primary[point]['rate_learning'] = rate_density
                self.params_primary[point]['rate'] = rate_density
                self.params_primary[point]['mfd'] = mfd
                self.params_primary[point]['lmmin'] = target_mmin
                self.params_primary[point]['mmin'] = target_mmin
                self.params_primary[point]['mmax'] = target_mmax
                self.params_primary[point]['model'] = 1
                self.params_primary[point]['params'] = [rate_density, 0]
                self.params_primary[point]['rates_bin'] = mag_rates

    def gr_scale(self, bval_nz, bval_tvz, poly_tvz):
        offset = 0.0001
        mesh = Mesh(self.grid_source[:, 0] + offset, self.grid_source[:, 1] + offset)
        points_tvz = np.argwhere(poly_tvz.intersects(mesh)).ravel()
        for point in range(self.nsources):
            if point in points_tvz:
                bval = bval_tvz
            else:
                bval = bval_nz

            t_mmin = self.params_primary[point]['mmin']
            lmmin = self.params_primary[point]['lmmin']
            rate_learning = self.params_primary[point]['rate_learning']
            aval = np.log10(rate_learning) + bval * lmmin
            rate_target = 10 ** (aval - bval * t_mmin)

            if self.params_primary[point]['model'] == 'NegBinom':
                rate_np_l = self.params_primary[point]['params'][0]
                aval = np.log10(rate_np_l) + bval * lmmin
                mu_i = 10 ** (aval - bval * t_mmin)
                self.params_primary[point]['params'] = [mu_i, self.params_primary[point]['params'][1]]
            else:
                self.params_primary[point]['params'] = [rate_target, 0]


            self.params_primary[point]['rate'] = rate_target
            self.params_primary[point]['bval'] = bval

    def set_mfd(self, bval, eepas=False, mbin=0.1,  polygons=None, polygon_bvalues=None):

        if not eepas:
            for point in range(self.nsources):
                t_mmin = self.params_primary[point]['mmin']
                mmax = self.params_primary[point]['mmax']
                rate_total = self.params_primary[point]['rate']
                if self.params_primary[point]['model'] == 'NegBinom':
                    rate_total = self.params_primary[point]['params'][0]

                tgr = self.truncated_GR(rate_total, bval, mbin, t_mmin, t_mmin, mmax)
                if self.magnitudes is None:
                    self.magnitudes = tgr[1]
                rates_bin = rate_total*tgr[2]

                if rate_total == 0:
                    rates_bin = np.ones(rates_bin.shape)*1e-10
                if not np.any(rates_bin):
                    rates_bin[0] = 1e-10

                mfd = EvenlyDiscretizedMFD(t_mmin, mbin, rates_bin)
                self.params_primary[point]['bval'] = bval
                self.params_primary[point]['rates_bin'] = rates_bin
                self.params_primary[point]['mfd'] = mfd


        else:
            tgr = self.truncated_GR(1, bval, 0.1, 5.0, 5.0, 8.0)
            weights = tgr[2]
            if self.magnitudes is None:
                self.magnitudes = tgr[1]

            for point in range(self.nsources):
                t_mmin = self.params_primary[point]['mmin']
                rates_bin = weights*self.params_primary[point]['rates_bin']
                rate_target = np.sum(rates_bin)
                if not np.any(rates_bin):
                    rates_bin[0] = 1e-6
                MFD = EvenlyDiscretizedMFD(t_mmin, mbin, rates_bin)
                self.params_primary[point]['rate'] = rate_target
                self.params_primary[point]['params'][0] = rate_target
                self.params_primary[point]['bval'] = bval
                self.params_primary[point]['rates_bin'] = rates_bin
                self.params_primary[point]['mfd'] = MFD

    def normalize(self, scale=1, mbin=0.1):

        factor = scale / np.sum([i['rate'] for i in self.params_primary.values() if i['rate'] is not None])
        for point, vals in self.params_primary.items():

            if vals['rate'] is None:
                continue

            rate_normalized = vals['rate'] * factor
            rates_bin_norm = vals['rates_bin'] * factor
            t_mmin = self.params_primary[point]['mmin']

            self.params_primary[point]['mfd'] = EvenlyDiscretizedMFD(t_mmin, mbin, rates_bin_norm)
            self.params_primary[point]['rate'] = rate_normalized
            self.params_primary[point]['rates_bin'] = rates_bin_norm
            if self.params_primary[point]['model'] == 'NegBinom':
                mu = self.params_primary[point]['params'][0]*factor
                self.params_primary[point]['params'] = np.array([mu, self.params_primary[point]['params'][1]])
            else:
                self.params_primary[point]['params'] = np.array([rate_normalized, 0])

    def normalize_m_bins(self, scale=1):

        factor_bin = scale / np.sum([i['rates_bin'] for i in self.params_primary.values() if i['rate'] is not None], axis=0)
        for point, vals in self.params_primary.items():
            if vals['rate'] is None:
                continue
            ratio_mean = vals['params'][0] / vals['rate']
            rates_bin_norm = vals['rates_bin'] * factor_bin * ratio_mean
            t_mmin = vals['mmin']

            self.params_primary[point]['mfd'] = EvenlyDiscretizedMFD(t_mmin, 0.1, rates_bin_norm)
            self.params_primary[point]['rates_bin'] = rates_bin_norm

    def plot_mfd(self, points, axes=None, color=None):

        ids = []
        for i in points:
            if isinstance(i, str):
                i = getattr(paths, i)

            if isinstance(i, int):
                point = i
            else:
                if np.argwhere(np.all(np.isclose(self.grid_source, i), axis=1)).shape[0]:
                    point = np.argwhere(np.all(np.isclose(self.grid_source, i), axis=1))[0, 0]
                else:
                    point = np.argmin(np.sum((self.grid_source - i)**2, axis=1))
            ids.append(point)
        print('aa')
        if axes is None:

            fig, axes = plt.subplots(1 if len(points) <= 3 else 2, 3, figsize=(12, 4))

        for ax, p in zip(axes, ids):
            rates = self.params_primary[p]['rates_bin']
            m = np.linspace(self.params_primary[p]['mmin'], self.params_primary[p]['mmax'], 31)
            ax.semilogy(m, rates, '-.', color=color)
        return axes

    @classmethod
    def floor_2models(cls, name, model_a, model_b, bin=None, folder='', floor_type='count'):

        model = cls(name, folder=folder, time_span=None)
        if not np.all(model_a.grid_source == model_b.grid_source):
            print('Grids not matching')

        model.get_geometry(grid=model_a)
        import copy
        model_a = copy.deepcopy(model_a)
        model_b = copy.deepcopy(model_b)
        model.magnitudes = model_b.magnitudes
        if floor_type == 'count':
            rates_a = np.sum([i['rate'] for i in model_a.params_primary.values() if i['rate'] is not None])
            rates_b = np.sum([i['rate'] for i in model_a.params_primary.values() if i['rate'] is not None])
            print('Model_rates: ', rates_a, rates_b)

            for points_a, points_b in zip(model_a.params_primary.items(), model_b.params_primary.items()):

                pa = points_a[0]
                pb = points_b[0]

                if isinstance(bin, int) and points_b[1]['bin'] != bin:
                    points_b[1]['rate'] = None
                    points_b[1]['params'][0] = 0
                    poly = 1

                if points_a[1]['rate'] is None:
                    rate_training = copy.deepcopy(points_b[1]['rate'])
                    temp_model = copy.deepcopy(points_b[1]['model'])
                    params = copy.deepcopy(points_b[1]['params'])
                    poly = 1

                elif points_b[1]['rate'] is None:
                    rate_training = copy.deepcopy(points_a[1]['rate'])
                    temp_model = copy.deepcopy(points_a[1]['model'])
                    params = copy.deepcopy(points_a[1]['params'])
                    poly = 0

                elif (points_b[1]['rate'] is not None) and (points_a[1]['rate'] is not None):
                    max_model = np.argmax([points_a[1]['params'][0], points_b[1]['params'][0]])
                    rate_training = np.max([points_a[1]['rate'], points_b[1]['rate']])

                    if max_model == 0:
                        temp_model = copy.deepcopy(points_a[1]['model'])
                        params = copy.deepcopy(points_a[1]['params'])
                        poly = 0

                    else:
                        temp_model = copy.deepcopy(points_b[1]['model'])
                        params = copy.deepcopy(points_b[1]['params'])
                        poly = 1


                model.params_primary[pa]['poly'] = points_b[1]['poly']
                model.params_primary[pa]['bin'] = points_b[1]['bin']
                model.params_primary[pa]['eventcount'] = 1
                model.params_primary[pa]['rate'] = rate_training
                model.params_primary[pa]['rate_learning'] = rate_training
                model.params_primary[pa]['lmmin'] = model_a.params_primary[pa]['lmmin']
                model.params_primary[pa]['mmin'] = model_a.params_primary[pa]['mmin']
                model.params_primary[pa]['mmax'] = model_a.params_primary[pa]['mmax']
                model.params_primary[pa]['model'] = temp_model
                model.params_primary[pa]['params'] = params

        elif floor_type == 'bin':
            rates_a = np.sum([i['rate'] for i in model_a.params_primary.values() if i['rate'] is not None])
            rates_b = np.sum([i['rate'] for i in model_b.params_primary.values() if i['rate'] is not None])
            print('rates', rates_a, rates_b)
            for (k_a, v_a), (k_b, v_b) in zip(model_a.params_primary.items(), model_b.params_primary.items()):


                if isinstance(bin, int) and v_b['bin'] != bin:
                    temp_model = copy.deepcopy(v_a['model'])
                    v_b['rate'] = None
                    v_b['rates_bin'] = np.zeros(31)
                    alpha = 0
                else:
                    temp_model = copy.deepcopy(v_b['model'])
                    alpha = v_b['params'][1]

                rates_bin = []
                mmin = v_b['mmin']
                mag_bin = 0.1

                for m_a, m_b in zip(v_a['rates_bin'], v_b['rates_bin']):
                    rates_bin.append(np.max([m_a, m_b]))

                rates_bin = np.array(rates_bin)
                rate = np.sum(rates_bin)
                params = [rate, alpha]
                mfd = EvenlyDiscretizedMFD(mmin, mag_bin, rates_bin)

                model.params_primary[k_a]['poly'] = v_b['poly']
                model.params_primary[k_a]['bin'] =  v_b['bin']
                model.params_primary[k_a]['eventcount'] = 1
                model.params_primary[k_a]['rate'] = rate
                model.params_primary[k_a]['rates_bin'] = rates_bin
                model.params_primary[k_a]['mfd'] = mfd
                model.params_primary[k_a]['bval'] = model_a.params_primary[k_a]['bval']
                model.params_primary[k_a]['lmmin'] = model_a.params_primary[k_a]['lmmin']
                model.params_primary[k_a]['mmin'] = model_a.params_primary[k_a]['mmin']
                model.params_primary[k_a]['mmax'] = model_a.params_primary[k_a]['mmax']
                model.params_primary[k_a]['model'] = temp_model
                model.params_primary[k_a]['params'] = params

        return model

    def fill_towards(self, nbin, method='count', alpha=5.0):

        if method == 'count':
            if np.argwhere(np.all(np.isclose(self.grid_source, paths.Dunedin), axis=1)).shape[0]:
                target_point = np.argwhere(np.all(np.isclose(self.grid_source, paths.Dunedin), axis=1))[0, 0]
            else:
                target_point = np.argmin(np.sum((self.grid_source - paths.Dunedin) ** 2, axis=1))

            points = np.array([self.grid_source[p, :] for p in range(self.nsources) if self.params_primary[p]['poly'] == nbin])
            id_points = [p for p in range(self.nsources) if self.params_primary[p]['poly'] == nbin]

            rate_target = self.params_primary[target_point]['params'][0]
            rate_training = self.params_primary[target_point]['rate']
            model = self.params_primary[target_point]['model']
            alpha_param = self.params_primary[target_point]['params'][1]
            # rate_target = np.mean([self.params_primary[p]['params'][0]
            #                              for p in range(self.nsources) if self.params_primary[p]['poly'] == nbin])

            # rate_training = np.mean([self.params_primary[p]['rate']
            #                              for p in range(self.nsources) if self.params_primary[p]['poly'] == nbin])

            cutoff_val = 1.02 * rate_target
            points_base = np.array([self.grid_source[p, :]
                                    for p in range(self.nsources) if self.params_primary[p]['params'][0] >= cutoff_val])


            alpha_shape = alphashape.alphashape([(i[0], i[1]) for i in np.vstack((points, points_base))], alpha)
            mesh = shapely.geometry.MultiPoint(self.grid_source)
            points_polygon = np.array([[i.x, i.y] for i in alpha_shape.intersection(mesh).geoms])
            # return points_polygon, points, points_base
            points_seg = []
            for i in points_polygon:
                if i.tolist() not in points_base.tolist() and i.tolist() not in points.tolist():
                    points_seg.append(i)

            points_seg = np.array(points_seg)
            id_seg = []
            for po in points_seg:
                if np.argwhere(np.all(np.isclose(self.grid_source, po), axis=1)).shape[0]:
                    pt = np.argwhere(np.all(np.isclose(self.grid_source, po), axis=1))[0, 0]
                    id_seg.append(pt)
            area_prop = self.cells_area[np.array(id_points)].sum() / \
                        (self.cells_area[np.array(id_points)].sum() + self.cells_area[np.array(id_seg)].sum())
            for p in range(self.nsources):
                if p in id_seg:
                    if self.params_primary[p]['params'][0] < cutoff_val:
                        self.params_primary[p]['params'] = [cutoff_val, alpha_param]
                        self.params_primary[p]['model'] = model
                        self.params_primary[p]['rate'] = rate_training

        elif method == 'bin':
            if np.argwhere(np.all(np.isclose(self.grid_source, paths.Dunedin), axis=1)).shape[0]:
                target_point = np.argwhere(np.all(np.isclose(self.grid_source, paths.Dunedin), axis=1))[0, 0]
            else:
                target_point = np.argmin(np.sum((self.grid_source - paths.Dunedin) ** 2, axis=1))

            for i, m in enumerate(self.magnitudes):

                points = np.array([self.grid_source[p, :] for p in range(self.nsources) if self.params_primary[p]['poly'] == nbin])
                id_points = [p for p in range(self.nsources) if self.params_primary[p]['poly'] == nbin]

                rate_target = self.params_primary[target_point]['rates_bin'][i]
                model = self.params_primary[target_point]['model']
                alpha_param = self.params_primary[target_point]['params'][1]

                cutoff_val = 1.02 * rate_target
                points_base = np.array([self.grid_source[p, :]
                                        for p in range(self.nsources) if self.params_primary[p]['rates_bin'][i] >= cutoff_val])

                alpha_shape = alphashape.alphashape([(k[0], k[1]) for k in np.vstack((points, points_base))], alpha)
                mesh = shapely.geometry.MultiPoint(self.grid_source)
                points_polygon = np.array([[k.x, k.y] for k in alpha_shape.intersection(mesh).geoms])
                points_seg = []
                for j in points_polygon:
                    if j.tolist() not in points_base.tolist() and j.tolist() not in points.tolist():
                        points_seg.append(j)

                points_seg = np.array(points_seg)
                id_seg = []
                for po in points_seg:
                    if np.argwhere(np.all(np.isclose(self.grid_source, po), axis=1)).shape[0]:
                        pt = np.argwhere(np.all(np.isclose(self.grid_source, po), axis=1))[0, 0]
                        id_seg.append(pt)
                area_prop = self.cells_area[np.array(id_points)].sum() / \
                            (self.cells_area[np.array(id_points)].sum() + self.cells_area[np.array(id_seg)].sum())
                for p in range(self.nsources):
                    if p in id_seg:
                        if self.params_primary[p]['rates_bin'][i] < cutoff_val:
                            self.params_primary[p]['params'][1] = alpha_param
                            self.params_primary[p]['model'] = model
                            self.params_primary[p]['rates_bin'][i] = cutoff_val

    def write_forecast(self, format='spatial', mmin=4.95):

        if format == 'spatial':

            n = self.cells.shape[0]
            rates = np.array([self.params_primary[i]['params'][0] for i in range(n)])
            dispersion = np.array([self.params_primary[i]['params'][1] for i in range(n)])
            cells = np.hstack((self.cells, np.array([mmin, 10.5]) * np.ones((n, 2))))
            data = np.hstack((cells, rates.reshape((-1, 1)), dispersion.reshape((-1, 1)), np.ones((n, 1))))
            header = 'lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, m_min, m_max, rate, dispersion, mask'
            np.savetxt(join(self.dirs['forecast'], f'{self.name}.csv'),
                       data, fmt=8 * ['%.1f'] + 1 * ['%.16e'] + 1 * ['%.6f'] + ['%i'],
                       delimiter=',', comments='', header=header)

        if format == 'mbin':

            n = self.cells.shape[0]
            rates = np.array([self.params_primary[i]['rates_bin'] for i in range(n)])
            dispersion = np.array([self.params_primary[i]['params'][1] for i in range(n)])
            magnitudes = np.linspace(5, 8, 31)
            data = np.hstack((self.cells, rates, dispersion.reshape((-1, 1))))

            header = 'lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, ' + ' '.join([f'{i:.1f}' for i in magnitudes]) + ', dispersion'
            np.savetxt(join(self.dirs['forecast'], f'{self.name}.csv'),
                       data, fmt=6 * ['%.1f'] + 32 * ['%.16e'],
                       header=header, delimiter=',', comments='')

    def write_vti(self, vtk_name=None,
                  path=None, epsg='EPSG:4326',
                  res=None, crop=False,
                  res_method='nearest'):

        if vtk_name is None:
            vtk_name = self.name

        attributes = self.primary_keys
        data = []
        datatype = []
        for key in attributes:
            if key == 'model':

                model_id = []
                for n in range(self.nsources):
                    if self.params_primary[n][key] is temporal.backwardPoisson:
                        id_ = 1
                    elif self.params_primary[n][key] is temporal.NegBinom:
                        id_ = 4
                    else:
                        id_ = 0
                    model_id.append(id_)
                data.append(model_id)
                datatype.append(float)

            elif key == 'mfd':

                continue

            elif key == 'rates_bin':
                rates = []
                mark = 0
                for point, vals in self.params_primary.items():
                    if vals[key] is not None:
                        if mark == 0:
                            rates = np.zeros((self.nsources, len(vals[key])))
                            rates[point, :] = vals[key]
                            mark += 1
                        else:
                            rates[point, :] = vals[key]

                data.append(rates)
                datatype.append(float)
                continue
            elif key == 'params':

                params = np.zeros((self.nsources, 2))

                for point, vals in self.params_primary.items():
                    for i, p in enumerate(vals['params']):
                        params[point, i] = p

                data.append(params)
                datatype.append(float)
            elif key == 'mmax':

                mm = []

                for point in range(self.nsources):
                    mm.append(point)


                data.append(mm)
                datatype.append(float)
            else:
                data.append([self.params_primary[i][key] if self.params_primary[i][key] is not None else np.nan for i in range(self.nsources)])
                datatype.append(float)
        # return data, datatype
        # Reproject grid
        if epsg == 'EPSG:4326':
            grid_source = np.vstack((self.grid_source[:, 0] - 0.05, self.grid_source[:, 1] + 0.05)).T
            res_x = np.min(np.diff(np.unique(np.sort(self.grid_source[:, 0]))))
            res_y = np.min(np.diff(np.unique(np.sort(self.grid_source[:, 1]))))
            res0 = (res_x + 0.001, res_y+ 0.001)
            path_crop = paths.region_nz_test

        else:
            grid_moved = np.vstack((self.grid_source[:, 0], self.grid_source[:, 1])).T
            grid_source = geo.reproject_coordinates(grid_moved, 'EPSG:4326', epsg)
            res_x = np.min(np.diff(np.unique(np.sort(self.grid_source[:, 0]))))
            res_y = np.min(np.diff(np.unique(np.sort(self.grid_source[:, 1]))))
            if epsg == 'EPSG:2193':
                res0 = (res_x * 111050,  res_y * 111050)   # approximate degrees to m
                path_crop = paths.region_nz_test_2193
            elif epsg == 'EPSG:3857':
                res0 = (res_x * 160000,  res_y * 160000)   # approximate degrees to m

        raster2vti_names = []
        for array, raster_fn0 in zip(data, attributes):
            _, raster_fn = geo.source_model2raster(array, datatype, self.dirs['output'], raster_fn0, grid_source, res0, srs=epsg)
            if res:
                raster_fn_f = raster_fn.replace('.tiff', '') + '_rs.tiff'

                ds = gdal.Translate(raster_fn_f, raster_fn, xRes=res[0], yRes=res[1], resampleAlg=res_method)
                ds = None
                if crop:
                    geo.mask_raster(path_crop, raster_fn_f, raster_fn_f, all_touched=False, crop=False)
            else:
                raster_fn_f = raster_fn
                if crop:
                    geo.mask_raster(path_crop, raster_fn_f, raster_fn_f, all_touched=False, crop=False)
            raster2vti_names.append(raster_fn_f)

        if path:
            image_filename = path
        else:
            image_filename = join(self.dirs['vti'], vtk_name + '.vti')
        _ = geo.source_raster2vti(image_filename, raster2vti_names, attributes,
                                  offset=10)
