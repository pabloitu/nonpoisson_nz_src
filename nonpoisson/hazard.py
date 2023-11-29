from os import makedirs
from os.path import join
import pickle
import h5py
from functools import partial
import fiona
import scipy.special as scp
import logging
import subprocess
import shutil
from nonpoisson import temporal
from nonpoisson import paths
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
import gc
from multiprocessing import Pool
from csep.core.regions import geographical_area_from_bounds
import seaborn
import copy
seaborn.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})
# seaborn.set(rc={"xtick.bottom": True, "ytick.left" : True,
#                 "xtick.color": 'darkgrey', 'xtick.labelcolor': 'black',
#                 "ytick.color": 'darkgrey', 'ytick.labelcolor': 'black'})
seaborn.set(rc={"xtick.bottom": True, "ytick.left" : True,
                "xtick.color": 'darkgrey',
                "ytick.color": 'darkgrey'})

soil_types = {'A': 1000,
              'B': 620,
              'C': 370,
              'D': 265,
              'E': 150}


def setup_log(file='oq.log'):
    log_file = file
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.propagate = False
    logger.propagate = False


def run_folder(folder_name, model=None):
    folder = paths.get_model('hazard', folder_name)
    setup_log(join(folder, 'log'))
    logger = logging.getLogger(__name__)

    job_path = join(folder, 'job.ini')
    print('Executing hazard model: %s' % folder)
    process_1 = subprocess.call(['oq', 'engine', '--run', job_path],
                                stdout=subprocess.PIPE)
    print('Run done')

    process_2 = subprocess.run(['oq', 'engine', '--lhc'],
                               stdout=subprocess.PIPE)
    last_log = str(process_2).split(r'\n')[-2]
    logger.info(f'\t{last_log}\t{job_path}')
    print('logging done')
    read_log(folder_name, copy=True)


def read_log(folder_name, copy=False, oq_folder='/home/pciturri/oqdata'):
    folder = paths.get_model('hazard', folder_name)
    models = {}
    with open(join(folder, 'log'), 'r') as file_:
        for line_ in file_.readlines():
            line = line_.split()
            name = line[10]
            calc_id = int(line[3])
            if name in models.keys():
                models[name] = max(models[name], calc_id)

            else:
                models[name] = calc_id
    with open(join(folder, 'calc_log'), 'w') as file_:
        for i, j in models.items():
            file_.write(f'{i},{j}\n')
    if copy:
        os.path.expanduser('~')
        print(models)
        for i, j in models.items():
            src = join(oq_folder, f'calc_{j}.hdf5')
            dest = join(folder, 'output/')
            shutil.copy(src, dest)


def get_nshm10_basemodel():

    path = paths.nshm10_50yr
    Model_original = openquake.hazardlib.nrml.read(path)
    ns = '{' + Model_original.attrib['xmlns'] + '}'
    sm = Model_original[0]
    ind = []

    ## Filter by Source Type (ignore fault sources)
    for i, source in enumerate(sm):
        if source.tag.split(ns)[1] != 'pointSource':
            ind.append(i)
    ind.sort()
    ind.reverse()
    for i in ind:
        Model_original[0].__delitem__(i)

    conv = sourceconverter.SourceConverter(discard_trts='Subduction Intraslab')
    Model = [node for node in conv.convert_node(Model_original[0]) if node]

    grid = np.array([[Point.location.x, Point.location.y] for Point in Model])
    unique_grid = np.array(np.unique(grid, axis=0))
    mapping = [np.where(np.all((grid == i), axis=1))[0] for i in unique_grid]

    rupture_mesh_spacing = 5.0
    rupture_aspect_ratio = 1
    upper_seismogenic_depth = 0
    lower_seismogenic_depth = 30
    simplified_model = []
    tectonic_region_type = 'Active Shallow Crust'
    for n in range(len(unique_grid)):
        source_id = '%05i' %  n
        name = 'point%05i' % n
        if len(mapping[n]) == 2:
            a1 = Model[mapping[n][0]].mfd.a_val
            a2 = Model[mapping[n][1]].mfd.a_val
            a = a1 + np.log10(1 + 10**(a2-a1))
            # tectonic_region_type = Model[mapping[n][0]].tectonic_region_type
        else:
            a = Model[mapping[n][0]].mfd.a_val
            # tectonic_region_type = Model[mapping[n][0]].tectonic_region_type
        npd = Model[mapping[n][0]].nodal_plane_distribution
        mfd = Model[mapping[n][0]].mfd
        mfd.a_val = a
        msr = Model[mapping[n][0]].magnitude_scaling_relationship
        loc = Model[mapping[n][0]].location
        tom = Model[mapping[n][0]].temporal_occurrence_model
        hd = Model[mapping[n][0]].hypocenter_distribution
        hd.data = [(0.5, 10.0), [0.5, 30.0]]
        simplified_model.append(PointSource(source_id=source_id,
                                            name=name,
                                            tectonic_region_type=tectonic_region_type,
                                            mfd=mfd,
                                            rupture_mesh_spacing=rupture_mesh_spacing,
                                            magnitude_scaling_relationship=msr,
                                            rupture_aspect_ratio=rupture_aspect_ratio,
                                            temporal_occurrence_model=tom,
                                            upper_seismogenic_depth=upper_seismogenic_depth,
                                            lower_seismogenic_depth=lower_seismogenic_depth,
                                            location=loc,
                                            nodal_plane_distribution=npd,
                                            hypocenter_distribution=hd))

    sourcewriter.write_source_model(paths.nshm10_simp, simplified_model, name=None, investigation_time=None)

    return simplified_model


class sourceLogicTree(object):

    def __init__(self, name):
        """


        """
        self.name = name
        self.xml_version = '1.0'
        self.xml_encoding = "utf-8"

        self.element_tree = None
        self.eroot = None

        self.nsmap = None
        self.ns = None
        self.logicTree = None
        # self.nsmap = {None: 'http://openquake.org/xmlns/nrml/0.5',
        #                        'gml': 'http://www.opengis.net/gml'}
        # self.ns = '{%s}' % self.nsmap[None]

    def load(self, filename):

        self.element_tree = etree.parse(filename)
        self.eroot = self.element_tree.getroot()

        self.xml_version = self.element_tree.docinfo.xml_version
        self.xml_encoding = self.element_tree.docinfo.encoding
        self.nsmap = self.eroot.nsmap
        self.ns = '{%s}' % self.nsmap[None]

        self.logicTree = {'attrib': dict(self.eroot[0].attrib), 'Levels': {}}
        for bl in self.eroot[0].getchildren():
            self.logicTree['Levels'][bl.attrib['branchingLevelID']] = {}
            for bs in bl.getchildren():
                self.logicTree['Levels'][bl.attrib['branchingLevelID']][bs.attrib['branchSetID']] =\
                    {'uncertaintyType': bs.attrib['uncertaintyType'], 'branches': {}}

                for ltb in bs.getchildren():
                    self.logicTree['Levels'][bl.attrib['branchingLevelID']][bs.attrib['branchSetID']]['branches'][ltb.attrib['branchID']] =\
                        {'file': ltb[0].text.replace(' ', '').replace('\n', ''), 'weight': float(ltb[1].text)}

    def create_branches(self, filenames):
        names = ['source_%s' % i for i in range(len(filenames))]
        weight = 1. / (len(filenames))
        self.logicTree['Levels']['bl1']['bs1']['branches'] = {i: {'file': fn, 'weight': weight}
                                                                     for i, fn in zip(names, filenames)}

    def save(self, filename):

        pre = self.ns
        nrml = etree.Element(pre + 'nrml', nsmap=self.nsmap)
        logicTree = etree.SubElement(nrml, 'logicTree', attrib=self.logicTree['attrib'], nsmap=self.nsmap)
        for i, j in self.logicTree['Levels'].items():
            logicTreeBranchingLevel = etree.SubElement(logicTree,
                                                       'logicTreeBranchingLevel',
                                                       attrib={'branchingLevelID': i})
            for k, l in j.items():

                logicTreeBranchSet = etree.SubElement(logicTreeBranchingLevel,
                                                      'logicTreeBranchSet',
                                                      attrib={'branchSetID': k,
                                                              'uncertaintyType': l['uncertaintyType']})
                for m, n in l['branches'].items():
                    logicTreeBranch = etree.SubElement(logicTreeBranchSet,
                                                       'logicTreeBranch',
                                                       attrib={'branchID': m})
                    um = etree.SubElement(logicTreeBranch, 'uncertaintyModel')
                    um.text = '\n\t\t\t%s\n' % n['file']
                    uw = etree.SubElement(logicTreeBranch, 'uncertaintyWeight')
                    uw.text = '\n\t\t\t%s\n' % n['weight']

        with open(filename, 'wb') as file_:
            file_.write(etree.tostring(nrml, pretty_print=True, xml_declaration=True, encoding=self.xml_encoding))


class hazardModel(object):

    def __init__(self, forecast, time_span=None):


        self.name = forecast.name
        self.time_span = time_span


        self.model_path = forecast.model_path
        self.dirs = {'output': join(self.model_path, 'output'),
                     'vti': join(self.model_path, 'vti'),
                     'figures': join(self.model_path, 'figures')}
        for dir_ in self.dirs.values():
            makedirs(dir_, exist_ok=True)

        self.cells_source = forecast.cells_source
        self.cells_area = forecast.cells_area
        self.total_area = forecast.total_area
        self.grid_source = forecast.grid_source
        self.grid_obs = None

        self.nsources = forecast.nsources

        self.sources = None
        self.params_primary = forecast.params_primary

        self.secondary_keys = ['trt',     # tectonic_region_type
                               'rms',     # rupture_mesh_spacing
                               'msr',     # magnitude_scaling_relationship
                               'rar',     # rupture_aspect_ratio
                               'usd',     # upper_seismogenic_depth
                               'lsd',     # lower_seismogenic_depth
                               'npd',     # nodal_plane_distribution
                               'hd']      # hypocenter_distribution
        self.params_secondary = {i: dict.fromkeys(self.secondary_keys) for i in range(self.nsources)}


        print(f'Model folder: {self.name}')

    def set_secondary_attr(self, trt='Active Shallow Crust', rms=5.0, msr=PointMSR(), rar=1.0,
                           usd=0, lsd=30, npd=[(1, NodalPlane(0, 90,0.))], hd=[(0.5, 10.0),(0.5, 30.0)]):

        npd = PMF(npd)
        hd = PMF(hd)
        for source in range(self.nsources):
            self.params_secondary[source]['trt'] = trt       # tectonic_region_type
            self.params_secondary[source]['rms'] = rms       # rupture_mesh_spacing
            self.params_secondary[source]['msr'] = msr       # magnitude_scaling_relationship
            self.params_secondary[source]['rar'] = rar       # rupture_aspect_ratio
            self.params_secondary[source]['usd'] = usd       # upper_seismogenic_depth
            self.params_secondary[source]['lsd'] = lsd       # lower_seismogenic_depth
            self.params_secondary[source]['npd'] = npd       # nodal_plane_distribution
            self.params_secondary[source]['hd'] = hd         # hypocenter_distribution

    def get_point_sources(self,  location=None, R=9, r_min=5):

        sources = []
        if location:
            poi = getattr(paths, location)
            idx = np.argwhere([geodetic_distance(poi[0, 0], poi[0, 1], i[0], i[1]) <= R and
                               geodetic_distance(poi[0, 0], poi[0, 1], i[0], i[1]) >= r_min for i in self.grid_source]).ravel()

        for point_id in range(self.nsources):
            if location and point_id not in idx:
                continue

            if self.params_primary[point_id]['model'] is None:
                continue
            tom = PoissonTOM(1)


            if self.params_primary[point_id]['model'] == 'NegBinom':
                mu = self.params_primary[point_id]['params'][0]
                alpha = self.params_primary[point_id]['params'][1]
                tom = NegativeBinomialTOM(1, mu=mu, alpha=alpha)


            source = PointSource(source_id='%05i' % point_id,
                                 name='point%05i' % point_id,
                                 tectonic_region_type=self.params_secondary[point_id]['trt'],
                                 mfd=self.params_primary[point_id]['mfd'],
                                 rupture_mesh_spacing=self.params_secondary[point_id]['rms'],
                                 magnitude_scaling_relationship=self.params_secondary[point_id]['msr'],
                                 rupture_aspect_ratio=self.params_secondary[point_id]['rar'],
                                 temporal_occurrence_model=tom,
                                 upper_seismogenic_depth=self.params_secondary[point_id]['usd'],
                                 lower_seismogenic_depth=self.params_secondary[point_id]['lsd'],
                                 location=Point(self.grid_source[point_id, 0],
                                                self.grid_source[point_id, 1]),
                                 nodal_plane_distribution=self.params_secondary[point_id]['npd'],
                                 hypocenter_distribution=self.params_secondary[point_id]['hd'])
            sources.append(source)
        self.sources = sources

    def get_rate_value(self, location):
        point = getattr(paths, location)
        point = np.argmin(np.sum((self.grid_source - point) ** 2, axis=1))
        rate = self.params_primary[point]['rate']
        return rate


    def write_model(self, n_gmpe=5, lt_samples=0, imtl={"PGA": list(np.logspace(-2, 0.2, 40))}, soil='C'):

        print('Writing files')
        source_filename = join(self.model_path, 'source.xml')
        job_filename = join(self.model_path, 'job.ini')
        gmpe_lt_filename = join(self.model_path, 'gmpe_logic_tree.xml')
        source_lt_filename = join(self.model_path, 'source_logic_tree.xml')
        sourcewriter.write_source_model(source_filename, self.sources, name=self.name,
                                        investigation_time=1)

        if self.grid_obs is None:
            poi = self.grid_source
        else:
            if isinstance(self.grid_obs, str):
                poi = getattr(paths, self.grid_obs)
            elif all(isinstance(loc, str) for loc in self.grid_obs):
                poi = np.vstack([getattr(paths, loc) for loc in self.grid_obs])

            elif isinstance(self.grid_obs, (list, np.ndarray)):
                poi = self.grid_obs


        source_model_lt = sourceLogicTree('base')
        source_model_lt.load(paths.base_sm_ltree)
        source_model_lt.create_branches([source_filename.split('/')[-1]])
        source_model_lt.save(source_lt_filename)

        np.savetxt(join(self.model_path, 'grid.txt'), poi, fmt='%.2f', delimiter=',')

        copyfile(getattr(paths, f'gmpe_lt_{n_gmpe}'), gmpe_lt_filename)
        copyfile(paths.job_classical,  job_filename)
        import fileinput

        with fileinput.FileInput(job_filename, inplace=True) as file:
            for line in file:
                print(line.replace('{name}', self.name), end='')
        with fileinput.FileInput(job_filename, inplace=True) as file:
            for line in file:
                print(line.replace('{lt_samples}', f'{lt_samples}'), end='')
        with fileinput.FileInput(job_filename, inplace=True) as file:
            for line in file:
                print(line.replace('{imtl}', f'{imtl}'), end='')
        with fileinput.FileInput(job_filename, inplace=True) as file:
            for line in file:
                print(line.replace('{reference_vs30_value}', f'{soil_types[soil]}'), end='')
        print('Writing ready')


class analyticToyModel(object):

    def __init__(self, name, folder='', time_span=None):


        self.name = name
        self.time_span = time_span

        self.model_path = paths.get_oqpath(join(folder, name))
        self.dirs = {'output': join(self.model_path, 'output'),
                     'vti': join(self.model_path, 'vti'),
                     'figures': join(self.model_path, 'figures')}
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

        self.sources = None
        self.params_primary = {}
        self.params_secondary = {}

        self.primary_keys = ['poly',        # Polygon ID
                             'poly_npoints',
                             'bin',         # Bin of spatial model
                             'eventcount',  # Events within polygon
                             'rate_learning', # Events divided cell size
                             'rate',  # Events divided cell size
                             'rates_bin', # Rates per mag bin
                             'aval',
                             'bval',
                             'lmmin',
                             'mmin',
                             'mmax',
                             'model',       # Temporal model
                             'params',      # Parameter of point temporal model
                             'corr']        # Correlation to other sources

        self.secondary_keys = ['trt',     # tectonic_region_type
                               'rms',     # rupture_mesh_spacing
                               'msr',     # magnitude_scaling_relationship
                               'rar',     # rupture_aspect_ratio
                               'usd',     # upper_seismogenic_depth
                               'lsd',     # lower_seismogenic_depth
                               'npd',     # nodal_plane_distribution
                               'hd']      # hypocenter_distribution


        print(f'Model folder: {self.name}')


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

    def get_geometry(self, grid='csep_testing', dh=0.05):

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
            self.params_secondary = {i: dict.fromkeys(self.secondary_keys) for i in range(self.nsources)}

        elif isinstance(grid, np.ndarray):
            self.cells = np.vstack((grid[:, 0] - dh, grid[:,0] + dh, grid[:, 1] - dh, grid[:, 1] + dh)).T
            self.cells_area = np.array([geographical_area_from_bounds(i[0], i[2], i[1], i[3]) for i in self.cells])
            self.total_area = np.sum(self.cells_area)
            self.grid_source = self.cells[:, (0, 2)] + 0.05
            self.nsources = grid.shape[0]
            self.params_primary = {i: dict.fromkeys(self.primary_keys) for i in range(self.nsources)}
            self.params_secondary = {i: dict.fromkeys(self.secondary_keys) for i in range(self.nsources)}

    def set_secondary_attr(self, trt='Active Shallow Crust', rms=5.0, msr=PointMSR(), rar=1.0,
                           usd=0, lsd=30, npd=[(1, NodalPlane(0,90,0.))], hd=[(0.5, 10.0),(0.5, 30.0)]):

        npd = PMF(npd)
        hd = PMF(hd)
        for source in range(self.nsources):
            self.params_secondary[source]['trt'] = trt       # tectonic_region_type
            self.params_secondary[source]['rms'] = rms       # rupture_mesh_spacing
            self.params_secondary[source]['msr'] = msr       # magnitude_scaling_relationship
            self.params_secondary[source]['rar'] = rar       # rupture_aspect_ratio
            self.params_secondary[source]['usd'] = usd       # upper_seismogenic_depth
            self.params_secondary[source]['lsd'] = lsd       # lower_seismogenic_depth
            self.params_secondary[source]['npd'] = npd       # nodal_plane_distribution
            self.params_secondary[source]['hd'] = hd         # hypocenter_distribution

    def set_params(self, model, params, bval=1.0, target_mmin=5.0, target_mmax=8.0):

            if model == 'poisson':
                rate_total = params
                model = 'poisson'
            if model == 'negbinom':
                rate_total = params[0]
                alpha = params[1]
                model = 'negbinom'

            for point in np.arange(self.nsources):

                rate_density = rate_total * self.cells_area[point] / np.sum(self.cells_area)
                if model == 'negbinom':
                    params[0] = rate_density
                else:
                    params = rate_density
                aval = np.log10(rate_density) + bval * target_mmin

                self.params_primary[point]['eventcount'] = rate_total
                self.params_primary[point]['rate'] = rate_density
                self.params_primary[point]['aval'] = aval
                self.params_primary[point]['mmin'] = target_mmin
                self.params_primary[point]['mmax'] = target_mmax
                self.params_primary[point]['model'] = model
                self.params_primary[point]['params'] = params
                self.params_primary[point]['corr'] = np.zeros(self.nsources)
                self.params_primary[point]['mfd'] = None

    def set_mfd(self, bval_base, mfd='TGR', mag_bin=0.1):

        if mfd == 'TGR':

            print('MODIFY FOR NBINOM')
            for point in np.arange(self.nsources):

                aval = self.params_primary[point]['aval']
                t_mmin = self.params_primary[point]['mmin']
                mmax = self.params_primary[point]['mmax']
                rate_target = 10 ** (aval - bval_base * t_mmin)
                mw_area = (10**(aval - bval_base*(t_mmin - mag_bin/2.)) - 10**(aval - bval_base * (mmax + mag_bin/2.)))
                mbins = np.arange(t_mmin, mmax + mag_bin/2., mag_bin)
                rates_bin = np.array([10 ** (aval - bval_base * (i - mag_bin/2)) -
                                      10 ** (aval - bval_base * (i + mag_bin/2.)) for i in mbins]) / (mw_area/rate_target)

                MFD = EvenlyDiscretizedMFD(t_mmin, mag_bin, rates_bin)
                self.params_primary[point]['rate'] = rate_target
                self.params_primary[point]['bval'] = bval_base
                self.params_primary[point]['rates_bin'] = rates_bin
                self.params_primary[point]['mfd'] = MFD

        elif mfd is None:
            print('MODIFY FOR NBINOM')
            for point in np.arange(self.nsources):
                rate = self.params_primary[point]['rate']
                aval = self.params_primary[point]['aval']
                t_mmin = self.params_primary[point]['mmin']
                self.params_primary[point]['mmax'] = t_mmin  + mag_bin

                mbins = np.array([t_mmin + mag_bin])
                rates_bin = np.array([rate])
                MFD = EvenlyDiscretizedMFD(t_mmin, mag_bin, rates_bin)
                self.params_primary[point]['rate'] = rate
                self.params_primary[point]['bval'] = bval_base
                self.params_primary[point]['rates_bin'] = rates_bin
                self.params_primary[point]['mfd'] = MFD

    def normalize_rate(self, scale=1):

            factor = scale / np.sum([i['rate'] for i in self.params_primary.values() if i['rate'] is not None])
            for point, vals in self.params_primary.items():
                if vals['rate'] is None:
                    continue

                rate_normalized = vals['rate'] * factor
                rates_bin_norm = vals['rates_bin'] * factor
                t_mmin = self.params_primary[point]['mmin']
                aval = np.log10(rate_normalized) + self.params_primary[point]['bval'] * self.params_primary[point]['mmin']

                self.params_primary[point]['mfd'] = EvenlyDiscretizedMFD(t_mmin, 0.1, rates_bin_norm)
                self.params_primary[point]['rate'] = rate_normalized
                self.params_primary[point]['rates_bin'] = rates_bin_norm
                self.params_primary[point]['aval'] = aval
                self.params_primary[point]['params'] = rate_normalized
                self.params_primary[point]['corr'] = np.zeros(self.nsources)

    def get_point_sources(self, grid_offset=0.05):

        sources = []

        for point_id in range(self.nsources):
            model = self.params_primary[point_id]['model']
            tom = PoissonTOM(1)
            if model is None or model == 'poisson':
                tom = PoissonTOM(1)
            elif model is temporal.NegBinom or model == 'negbinom':
                mu = self.params_primary[point_id]['params'][0]
                alpha = self.params_primary[point_id]['params'][1]
                # tau = 1 / alpha
                # theta = tau / (tau + mu)
                # aval = self.params_primary[point_id]['aval']
                # bval = self.params_primary[point_id]['bval']
                # mmin = self.params_primary[point_id]['mmin']
                # mmax = self.params_primary[point_id]['mmax']
                tom = NegativeBinomialTOM(1, parameters=[mu, alpha])

            source = PointSource(source_id='%05i' % point_id,
                                 name='point%05i' % point_id,
                                 tectonic_region_type=self.params_secondary[point_id]['trt'],
                                 mfd=self.params_primary[point_id]['mfd'],
                                 rupture_mesh_spacing=self.params_secondary[point_id]['rms'],
                                 magnitude_scaling_relationship=self.params_secondary[point_id]['msr'],
                                 rupture_aspect_ratio=self.params_secondary[point_id]['rar'],
                                 temporal_occurrence_model=tom,
                                 upper_seismogenic_depth=self.params_secondary[point_id]['usd'],
                                 lower_seismogenic_depth=self.params_secondary[point_id]['lsd'],
                                 location=Point(self.grid_source[point_id, 0] + grid_offset,
                                                self.grid_source[point_id, 1] + grid_offset),
                                 nodal_plane_distribution=self.params_secondary[point_id]['npd'],
                                 hypocenter_distribution=self.params_secondary[point_id]['hd'])
            sources.append(source)
        self.sources = sources

    def write_model(self, n_gmpe=5, lt_samples=0, imtl={"PGA": list(np.logspace(-2, 0.2, 40))}, soil='C'):

        print('Writing files')
        source_filename = join(self.model_path, 'source.xml')
        job_filename = join(self.model_path, 'job.ini')
        gmpe_lt_filename = join(self.model_path, 'gmpe_logic_tree.xml')
        source_lt_filename = join(self.model_path, 'source_logic_tree.xml')
        sourcewriter.write_source_model(source_filename, self.sources, name=self.name,
                                        investigation_time=1)

        if self.grid_obs is None:
            poi = self.grid_source
        else:
            if isinstance(self.grid_obs, str):
                poi = getattr(paths, self.grid_obs)
            elif all(isinstance(loc, str) for loc in self.grid_obs):
                poi = np.vstack([getattr(paths, loc) for loc in self.grid_obs])

            elif isinstance(self.grid_obs, (list, np.ndarray)):
                poi = self.grid_obs


        source_model_lt = sourceLogicTree('base')
        source_model_lt.load(paths.base_sm_ltree)
        source_model_lt.create_branches([source_filename.split('/')[-1]])
        source_model_lt.save(source_lt_filename)

        np.savetxt(join(self.model_path, 'grid.txt'), poi, fmt='%.2f')

        copyfile(getattr(paths, f'gmpe_lt_{n_gmpe}'), gmpe_lt_filename)
        copyfile(paths.job_classical,  job_filename)
        import fileinput

        with fileinput.FileInput(job_filename, inplace=True) as file:
            for line in file:
                print(line.replace('{name}', self.name), end='')
        with fileinput.FileInput(job_filename, inplace=True) as file:
            for line in file:
                print(line.replace('{lt_samples}', f'{lt_samples}'), end='')
        with fileinput.FileInput(job_filename, inplace=True) as file:
            for line in file:
                print(line.replace('{imtl}', f'{imtl}'), end='')
        with fileinput.FileInput(job_filename, inplace=True) as file:
            for line in file:
                print(line.replace('{reference_vs30_value}', f'{soil_types[soil]}'), end='')
        print('Writing ready')

    def write_vti(self, vtk_name=None, epsg='EPSG:4326', res=None, crop=False, res_method='nearest'):
        import geo
        import gdal
        if vtk_name is None:
            vtk_name = self.name


        attributes = self.primary_keys
        data = []
        datatype = []
        for key in attributes:
            if key == 'corr':
                data.append([np.max(self.params_primary[i][key]) if self.params_primary[i][key] is not None else 0 for i in range(self.nsources)])
                datatype.append(float)
            elif key == 'model':

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
                max_params = 0
                for n in range(self.nsources):
                    if isinstance(self.params_primary[n][key], list):
                        max_params = max(max_params, len(self.params_primary[n][key]))
                    elif isinstance(self.params_primary[n][key], (float, int)):
                        max_params = max(max_params, 1)
                params = np.zeros((self.nsources, max_params))

                for point, vals in self.params_primary.items():
                    if isinstance(vals['params'], list):
                        for i, p in enumerate(vals['params']):
                            params[point, i] = p
                    elif isinstance(vals['params'], (int, float)):
                        params[point, 0] = vals['params']

                data.append(params)
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

        image_filename = join(self.dirs['vti'], vtk_name + '.vti')
        _ = geo.source_raster2vti(image_filename, raster2vti_names, attributes, offset=(0, 0, 10))


class hazardResults(object):

    def __init__(self, name, path, calcpath=None):

        self.name = name
        self.path = path
        print(path)
        self.calcpath = calcpath
        if path:
            self.dirs = {'input': join(path, 'input'),
                         'output': join(path, 'output'),
                         'vti': join(path, 'vti'),
                         'figures': join(path, 'figures')}

        if isinstance(calcpath, int):
            self.calcpath = paths.get_oqcalc(calcpath)
        elif calcpath is None:
            files = list(os.walk(self.dirs['output']))[0][2]
            calcs = [i for i in files if 'calc' in i]
            ids = [int(i.split('_')[1].split('.')[0]) for i in calcs]
            self.calcpath = join(self.dirs['output'], calcs[int(np.argmax(ids))])

        # create dirs


            # for dir in self.dirs.values():
            #     makedirs(dir, exist_ok=True)

        # Initialize class atritubtes
        self.ns = None  # Number of sites
        self.nb = None  # Number of branches
        self.ni = None  # Number of intensity measures
        self.nl = None  # Number of intensity measure levels
        self.nps = 0  # Number of intensity measure poes

        self.trt = []  # Tectonic region types
        self.gmpe_lt = {}  # Idem for the GMPE logic tree
        self.sm_lt = {}  # Creates a simplified structure for a source model logic tree
        self.branches = None  # Branch id, components and weight of each realization

        self.grid = None  # 2D np array
        self.hmaps = None  # 4D np array, containing hazard maps
        self.hcurves = None  # 4D h5py database , containing hazard curves
        self.immp = {}
        self.imtl = {}

        self.hmaps_stats = None  # Mean, 0.1q, 0.15q, 0.25q, 0.5q (median), 0.75q, 0.85q, 0.9q
        self.hcurves_stats = None

        self.ks1_map = None  # n_points x n_dists x n_measures x n_levels
        self.chi_map = None  # n_points x n_dists x n_measures x n_levels
        self.ks1_curve = None  # n_points x n_dists x n_measures x n_levels
        self.chi_curve = None  # n_points x n_dists x n_measures x n_levels


    def parse_db(self, imtl):
        db = h5py.File(self.calcpath, 'r')
        try:
            shape = db['hcurves-rlzs'].shape
        except:
            shape = db['hcurves-stats'].shape
        self.ns = shape[0]
        self.nb = shape[1]
        self.ni = shape[2]
        self.nl = shape[3]

        self.grid = np.stack((db['sitecol']['lon'], db['sitecol']['lat'])).T
        self.sm_lt = [(sm[1], sm[4]) for sm in db['full_lt']['source_model_lt']]
        self.trt = np.unique([tr[0] for tr in db['full_lt']['gsim_lt']])
        self.gsim_lt = {tr: [(br[2], br[3])
                             for br in db['full_lt']['gsim_lt']
                                   if (br[0] == tr)] for tr in self.trt}

        branch_components = itertools.product([sm[0] for sm in self.sm_lt], *[[i[0] for i in j]
                                                                  for j in self.gsim_lt.values()])
        branch_weights = db['weights'][:]
        self.branches = [(n, comp, w) for n, comp, w in zip(range(self.nb), branch_components, branch_weights)]
        try:
            self.hcurves = db['hcurves-rlzs']
        except:
            self.hcurves = db['hcurves-stats']
        self.hmaps = np.zeros((self.ns, self.nb, self.ni, 0))

        #todo automatize
        if imtl:
            self.imtl = imtl
        else:
            imtls = [i for i in db['oqparam'] if 'hazard_imtls' in i[0].decode('utf-8')]
            self.imtl = {i[0].decode('utf-8').split('.')[-1]: np.array(eval(i[1])) for i in imtls}

        # self.hmaps = db['hmaps-rlzs'][:]
        # self.immp = {i: np.array([0.002105]) for i in self.imtl.keys()}  #todo automatize
        # self.nps = 1


    @staticmethod
    def compute_hazard_maps(A):
        """
        Modification of openquake.commonlib.calc.compute_hazard_maps() for efficiency
        """
        EPSILON = 1E-30

        curves, log_imls, log_poes = A[0], A[1], A[2]
        P = len(log_poes)
        N, L = curves.shape  # number of levels

        if L != len(log_imls):
            raise ValueError('The curves have %d levels, %d were passed' %
                             (L, len(log_imls)))

        hmap = np.zeros((N, P))

        for n, curve in enumerate(curves):
            # the hazard curve, having replaced the too small poes with EPSILON
            log_curve = np.log([max(poe, EPSILON) for poe in curve[::-1]])
            for p, log_poe in enumerate(log_poes):
                hmap[n, p] = np.exp(np.interp(log_poe, log_curve, log_imls))
        return hmap

    def get_maps_from_curves(self, measures, poes, nproc=16):

        if isinstance(poes, (float, int, list)):
            poes = np.array([poes]).ravel()

        self.hmaps = np.zeros((self.ns, self.nb, self.ni, len(poes)))

        index_measure = list(range(self.ni))
        self.immp = {measure: np.zeros(0) for measure in measures}

        t0 = time.time()

        for ind, measure in zip(index_measure, measures):
            pool = Pool(nproc)
            log_poes = np.log(poes)
            log_levels = np.log(np.array(self.imtl[measure][::-1]))
            A = pool.map(self.compute_hazard_maps, [(i, log_levels, log_poes) for i in self.hcurves[:, :, ind, :].squeeze()])
            pool.close()

            t1 = time.time()
            print(f'Processing {self.name} maps in: {t1-t0} seconds')
            self.immp[measure] = np.append(self.immp[measure], poes)
            map = np.array(A)
            self.hmaps[:,:,ind, :] = map.reshape(self.ns, self.nb, len(poes))
        self.nps = len(poes)
        #                        map.reshape(self.ns, self.nb, 1, len(poes)), axis=-1)
        # self.nps += len(poes)

        # measure_new_array = np.copy(self.hmaps)
        # map = np.array(A).reshape((self.ns, self.nb, len(poes)))
        # new_array = np.append(measure_new_array[:, :, index_measure, :], map, axis=-1)
        # measure_new_array[:, :, index_measure, :] = new_array.reshape((self.ns, self.nb, 1, len(poes)))
        # print(new_array.shape,
        #       'aaa')  # self.hmaps[:,:,index_measure,:]  # self.hmaps = np.append(self.hmaps[:, :, index_measure, :],  # map.reshape(self.ns, self.nb, 1, len(poes)), axis=-1)  # self.nps += len(poes)

    @staticmethod
    def log_interp1d(x, xx, yy, kind='linear', axis=-1):
        lin_interp = sp.interpolate.interp1d(np.log10(xx), np.log10(yy), axis=axis, kind=kind, fill_value='extrapolate')
        log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
        interp = log_interp(x)

        return interp

    @staticmethod
    def quantile_curve(quantile, curves, weights=None):
        """
        Modification of openquake.hazardlib.stats.quantile_curve() for efficiency
        """

        R = len(curves)
        if weights is None:
            weights = np.ones(R) / R
        else:
            weights = np.array(weights)
            assert len(weights) == R, (len(weights), R)
        result = np.zeros((len(quantile), *curves.shape[1:]))
        for idx, _ in np.ndenumerate(np.zeros(curves.shape[1:])):
            data = curves[:, idx[0], idx[1]]
            sorted_idxs = np.argsort(data)
            cum_weights = np.cumsum(weights[sorted_idxs])
            result[:, idx[0], idx[1]] = np.interp(quantile, cum_weights, data[sorted_idxs])
        return result

    @staticmethod
    def geometric_mean(a, axis=0, dtype=None, weights=None):

        if not isinstance(a, np.ndarray):
            log_a = np.log(np.array(a, dtype=dtype))
        elif dtype:
            # Must change the default dtype allowing array type
            if isinstance(a, np.ma.MaskedArray):
                log_a = np.log(np.ma.asarray(a, dtype=dtype))
            else:
                log_a = np.log(np.asarray(a, dtype=dtype))
        else:
            log_a = np.log(a)

        if weights is not None:
            weights = np.asanyarray(weights, dtype=dtype)
        return np.exp(np.average(log_a, axis=axis, weights=weights))

    def get_stats(self, attr, measure):

        # Stats >  hmaps_stats, hcurves_stats

        # 0 arithmetic mean
        # 1 geometric mean
        # 2 0.1 quantile
        # 3 0.25 quantile
        # 4 median
        # 5 0.75 quantile
        # 6 0.9 quantile

        if attr == 'hcurves':
            pointer = self.imtl
            if self.hcurves_stats is None:
                self.hcurves_stats = np.zeros((self.ns, 7, self.ni, self.nl))
            Stats = self.hcurves_stats

        elif attr == 'hmaps':
            pointer = self.immp
            if self.hmaps_stats is None:
                self.hmaps_stats = np.zeros((self.ns, 7, self.ni, self.nps))
            Stats = self.hmaps_stats

        t0 = time.time()
        measure_ind = np.argwhere(np.in1d(sorted(list(pointer.keys())), measure)).ravel()[0]
        levels_ind = np.arange(0, len(pointer[measure]))

        # Arithmetic mean
        # print(len([i[2] for i in self.branches]), getattr(self, attr).shape)
        mean = np.average(getattr(self, attr)[:, :, measure_ind, levels_ind],
                          weights=[i[2] for i in self.branches], axis=1)
        Stats[:, 0, measure_ind, :] = mean

        # Geometric mean
        geom_mean = self.geometric_mean(getattr(self, attr)[:, :, measure_ind, levels_ind],
                                   weights=[i[2] for i in self.branches], axis=1)
        Stats[:, 1, measure_ind, :] = geom_mean

        qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q = self.quantile_curve(qs, np.swapaxes(getattr(self, attr)[:, :, measure_ind, levels_ind], 0, 1),
                           weights=[i[2] for i in self.branches])
        Stats[:, 2:, measure_ind, :] = np.swapaxes(q, 0, 1)
        t1 = time.time()
        print('Processing stats in: %.1f seconds' % (t1 - t0))

    def plot_pointcurves(self, measure, point, ax=None, title=None, plot_args={}, filename=None, yrs=None):
        xlims = plot_args.get('xlims', [1e-2, 1.5])
        ylims = plot_args.get('ylims', [1e-3, 1.2])
        poes_label = plot_args.get('poes', False)
        plot_branches = plot_args.get('plot_branches', False)
        plot_mean = plot_args.get('plot_mean', True)
        plot_geomean = plot_args.get('plot_geomean', False)
        plot_median = plot_args.get('plot_median', False)
        plot_quantile = plot_args.get('plot_quantile', False)
        plot_env = plot_args.get('plot_env', False)
        branches_lw = plot_args.get('branches_lw', 0.05)
        branches_c = plot_args.get('branches_c', 'steelblue')
        branches_alpha = plot_args.get('branches_alpha', 0.1)
        stats_lw = plot_args.get('stats_lw', 2)
        mean_c = plot_args.get('mean_c', 'steelblue')
        mean_s = plot_args.get('mean_s', '-')
        geomean_c = plot_args.get('geomean_c', 'green')
        geomean_s = plot_args.get('geomean_s', '-')
        median_c = plot_args.get('median_c', 'green')
        median_s = plot_args.get('median_s', '-')
        quantile_c = plot_args.get('quantile_c', 'gold')
        quantile_s = plot_args.get('quantile_s',  '--')
        env_c = plot_args.get('env_c', 'steelblue')
        env_alpha = plot_args.get('env_alpha', 0.3)
        labels = plot_args.get('labels', None)  # branch, am, gm, med, q, env:  Labels
        xlabel = plot_args.get('xlabel', None)
        ylabel = plot_args.get('ylabel', None)
        legend = plot_args.get('legend', None)
        if labels is None:
            labels = [self.name]
        else:
            if not isinstance(labels, list):
                labels = [self.name]

        index_measure = np.argwhere(np.in1d(sorted(list(self.imtl.keys())), measure)).ravel()[0]
        if isinstance(point, int):
            point = point
        else:
            if np.argwhere(np.all(np.isclose(self.grid, point), axis=1)).shape[0]:
                point = np.argwhere(np.all(np.isclose(self.grid, point), axis=1))[0, 0]
            else:
                point = np.argmin(np.sum((self.grid - point)**2, axis=1))
                # print(point)
                # point = np.argwhere(np.all(np.isclose(self.grid, point + 0.05), axis=1))[0, 0]

        title = title if title else '%s - $x=(%.1f,%.1f)$' % (self.name, self.grid[point, 0], self.grid[point, 1])

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_title(title, fontsize=16)

        if plot_branches:
            ax.loglog(self.imtl[measure], self.hcurves[point, :, index_measure, :].T,
                       linewidth=branches_lw, alpha=branches_alpha, color=branches_c, label=labels.pop(0))
        if plot_mean:
            ax.loglog(self.imtl[measure], self.hcurves_stats[point, 0, index_measure, :].T,
                       linewidth=stats_lw, color=mean_c, linestyle=mean_s, label=labels.pop(0))
        if plot_geomean:
            ax.loglog(self.imtl[measure], self.hcurves_stats[point, 1, index_measure, :].T,
                       linewidth=stats_lw, linestyle=geomean_s, color=geomean_c, label=labels.pop(0))
        if plot_median:
            ax.loglog(self.imtl[measure], self.hcurves_stats[point, 4, index_measure, :].T,
                       linewidth=stats_lw, linestyle=median_s, color=median_c, label=labels.pop(0))
        if plot_quantile:
            ax.loglog(self.imtl[measure], self.hcurves_stats[point, 2, index_measure, :].T,
                       linewidth=stats_lw, linestyle=quantile_s, color=quantile_c, label=labels.pop(0))
            ax.loglog(self.imtl[measure], self.hcurves_stats[point, -1, index_measure, :].T,
                       linewidth=stats_lw, linestyle=quantile_s, color=quantile_c)

        if plot_env:
            ax.fill_between(self.imtl[measure], self.hcurves_stats[point, 2, index_measure, :].T,
                             self.hcurves_stats[point, -1, index_measure, :].T,
                             color=env_c, alpha=env_alpha, label=labels[5])

        if not xlims:
            xlims = [min(self.imtl[measure]), max(self.imtl[measure])]

        ax.set_xlim(xlims)
        if ylims:
            ax.set_ylim(ylims)

        if poes_label == True:
            if yrs == 50:
                ax.axhline(0.1, linestyle='--', linewidth=0.8, color='black')
                ax.text(min(xlims)*1.1, 0.102, '10% in ' + '50 yr.', fontsize=12)
                ax.axhline(0.02, linestyle='--', linewidth=0.8, color='black')
                ax.text(min(xlims)*1.1, 0.0202, '2% in '+'50 yr.', fontsize=12)
            elif yrs == 1:
                ax.axhline(0.00205, linestyle='--', linewidth=0.8, color='black')
                ax.text(min(xlims)*1.1, 0.00215, '10% in ' + '50 yr.', fontsize=12)
                ax.axhline(0.000404, linestyle='--', linewidth=0.8, color='black')
                ax.text(min(xlims)*1.1, 0.000454, '2% in '+'50 yr.', fontsize=12)
        if xlabel is False:
            ax.set_xlabel(f'{measure} $[g]$', fontsize=14)

        if ylabel is False:
            if yrs:
                ylabel = 'Probability of exceedance - %i years' % yrs
            else:
                ylabel = 'Probability of exceedance'

            ax.set_ylabel(ylabel, fontsize=14)

        if legend:
            ax.legend(loc='upper right', fontsize=14)
        if filename:
            plt.savefig(join(self.dirs['figures'], filename), dpi=300)
        return ax


    def get_map_histogram(self, point, title=None, measure='PGA', poe=0.002105, weighted=False,
                          bins=50, plot=False, color='steelblue', lw=0.2, alpha=0.7, filename=False):

        if isinstance(point, int):
            point = point
        else:
            point = np.argwhere(np.all(np.isclose(self.grid, point), axis=1))[0, 0]
        index_measure = np.argwhere(np.in1d(sorted(list(self.immp.keys())), measure)).ravel()[0]
        index_poe = np.argwhere(self.immp[measure] == poe).ravel()[0]

        distribution = self.hmaps[point, :, index_measure, index_poe]

        if weighted:
            weights = np.array([i[2] for i in self.branches])
            med = self.hmaps_stats[point, 4, index_measure, index_poe]
            mean = self.hmaps_stats[point, 0, index_measure, index_poe]
            quart_1 = self.hmaps_stats[point, 3, index_measure, index_poe]
            quart_3 = self.hmaps_stats[point, 5, index_measure, index_poe]
        else:
            weights = None
            med = np.median(distribution)
            mean = np.mean(distribution)
            quart_1 = np.quantile(distribution, 0.25)
            quart_3 = np.quantile(distribution, 0.75)

        if plot:
            bin_cutoffs = np.linspace(0, 1.1 * np.percentile(distribution, 99), bins)
            plt.hist(distribution, density=True,
                     color=color, bins=bin_cutoffs, linewidth=lw, alpha=alpha, weights=weights)
            plt.axvline(med, linestyle='-', color='purple', label='Median')
            plt.axvline(mean, linestyle='-', color='green', label='Arithmetic mean')
            plt.axvline(quart_1, linestyle='--', color='orange', label='1st and 3rd Quartiles')
            plt.axvline(quart_3, linestyle='--', color='orange')

            plt.ylabel('Probability Density', fontsize=14)
            plt.xlabel('$\mathrm{%s}_{\mathrm{PoE}=%s}$' % (measure, str(poe)), fontsize=14)
            plt.legend()
            plt.xlim(0, np.percentile(distribution, 99))
            plt.title(title, fontsize=16)
            if filename:
                plt.savefig(join(self.dirs['figures'], filename), dpi=300)
            plt.show()
        histogram = np.histogram(distribution,
                                 range=(np.nanmin(distribution), np.nanmax(distribution)),
                                 bins=bins, density=True)

        bins = histogram[1]
        Hist = histogram[0]

        return Hist, bins, distribution

    def get_curve_histogram(self, point, title=None, measure='PGA', level=0.1, weighted=False,
                            bins=50, plot=False, color='steelblue', lw=0.2, alpha=0.7, filename=False):

        if isinstance(point, int):
            point = point
        else:
            point = np.argwhere(np.all(np.isclose(self.grid, point), axis=1))[0, 0]
        index_measure = np.argwhere(np.in1d(sorted(list(self.imtl.keys())), measure)).ravel()[0]
        index_level = np.argwhere(self.imtl[measure] == level).ravel()[0]

        distribution = self.hcurves[point, :, index_measure, index_level]

        if weighted:
            weights = np.array([i[2] for i in self.branches])
            med = self.hcurves_stats[point, 4, index_measure, index_level]
            mean = self.hcurves_stats[point, 0, index_measure, index_level]
            quart_1 = self.hcurves_stats[point, 3, index_measure, index_level]
            quart_3 = self.hcurves_stats[point, 5, index_measure, index_level]

        else:
            weights = None
            med = np.median(distribution)
            mean = np.mean(distribution)
            quart_1 = np.quantile(distribution, 0.25)
            quart_3 = np.quantile(distribution, 0.75)

        if plot:
            bin_cutoffs = np.linspace(0, 1.1 * np.percentile(distribution, 99), bins)
            plt.hist(distribution, density=True,
                     color=color, bins=bin_cutoffs, linewidth=lw, alpha=alpha, weights=weights)
            plt.axvline(med, linestyle='-', color='purple', label='Median')
            plt.axvline(mean, linestyle='-', color='green', label='Arithmetic mean')
            plt.axvline(quart_1, linestyle='--', color='orange', label='1st and 3rd Quartiles')
            plt.axvline(quart_3, linestyle='--', color='orange')

            plt.ylabel('Probability Density', fontsize=14)
            plt.xlabel('$\mathrm{PoE}_{\mathrm{%s}=%s}$' % (measure, str(level)), fontsize=14)
            plt.legend()
            plt.xlim(0, np.percentile(distribution, 99))
            plt.title(title, fontsize=16)
            if filename:
                plt.savefig(join(self.dirs['figures'], filename), dpi=300)
            plt.show()
        histogram = np.histogram(distribution,
                                 range=(np.nanmin(distribution), np.nanmax(distribution)),
                                 bins=bins, density=True)

        bins = histogram[1]
        Hist = histogram[0]

        return Hist, bins, distribution

    def model2vti(self, filename, attr, measures, levels=None,
                  branches=None, res=None, res_method='nearest', crs_f='EPSG:4326', crop=False):
        """
        Complete method to create a VTK 2D image to read in Paraview from class attributes.
        Calls a data_structure organizer to pre-arrange the data, in a readable format.
        Returns raster and shapefiles as intermediate step

        :param filename: (str) Name of the produced output files. No extension is needed
        :param attribute: (str)  Name of the class attribute (e.g. 'hazard_maps', 'hazard_curves', 'quantiles', 'mean'
        :param measure: (str) Intensity measure, e.g. 'PGA', 'SA0.1', 'SA1', etc.
        :param levels: (float/str/list) Levels (intensity levels, or poes, depending if maps/curves are flagged)
        :param structure_type (str): Organization of data
        :param id_elements (str/int/list): Elements to be plotted. In case of structure_type 'bylevel',
                        returns data structured by branch, etc. see e.g.: get_datastruct_bybranch()
        :param res_0 (tuple/list):  Resolution of the original crs_0, or input data. Default: min grid distance
        :param res_i: (tuple/list): Exporting resolution in crs_i. Default: res_0
        :param log:  (not implemented) log of variables
        :param bounds: Bounding box of the output raster/image
        :param resample: Method on which doing resample

        :param crs_0:  Coordinate reference system of the initial data. Usually same as OQ: epsg:4326
        :param crs_f:  Output of the raster and vti images

        :return:
        Creates as intermediate step, a GeoJSON file, a GeoTiff file with multi-band corresponding to all data arrays,
        and a .vti image, to be loaded in ParaView

        """
        from nonpoisson import geo
        from osgeo import gdal
        if isinstance(measures, str):
            measures = [measures]
        if isinstance(levels, float):  # Supports only same shape of levels for all measures
            levels = np.array([levels])
        else:
            levels = np.array(levels)

        if 'curve' in attr:
            pointer = self.imtl
        elif 'map' in attr:
            pointer = self.immp

        measure_ind = np.argwhere(np.in1d(sorted(list(pointer.keys())), measures)).ravel()
        levels_ind = np.argwhere(np.in1d(pointer[measures[0]], levels)).ravel()
        names = ['_'.join([str(j) for j in i]) for i in list(itertools.product(measures, levels))]
        indexes = list(itertools.product(measure_ind, levels_ind))

        if branches is None:
            data = [getattr(self, attr)[:, :, i, j] for i, j in indexes]

        elif isinstance(branches, (np.ndarray, list, range, int)):
            data = [getattr(self, attr)[:, np.array(branches), i, j] for i, j in indexes]


        # Reproject grid
        # if crs_f != 'EPSG:4326':
        #     grid = geo.reproject_coordinates(self.grid, 'EPSG:4326', crs_f)
        # else:
        #     grid = self.grid
        #
        # res_x = np.min(np.diff(np.unique(np.sort(grid[:, 0]))))
        # res_y = np.min(np.diff(np.unique(np.sort(grid[:, 1]))))
        # res0 = (res_x, res_y)

        if crs_f == 'EPSG:4326':
            grid = self.grid
            res_x = np.min(np.diff(np.unique(np.sort(self.grid[:, 0]))))
            res_y = np.min(np.diff(np.unique(np.sort(self.grid[:, 1]))))
            res0 = (res_x + 0.001, res_y + 0.001)

        else:
            grid = geo.reproject_coordinates(self.grid, 'EPSG:4326', crs_f)
            res_x = np.min(np.diff(np.unique(np.sort(self.grid[:, 0]))))
            res_y = np.min(np.diff(np.unique(np.sort(self.grid[:, 1]))))
            res0 = (res_x * 111100,  res_y * 111100)   # approximate degrees to m



        raster2vti_names = []
        for array, raster_fn0 in zip(data, [filename + '_' + name for name in names]):
            _, raster_fn = geo.hazard_model2raster(array, self.dirs['output'], raster_fn0, grid, res0, srs=crs_f)
            if res:
                raster_fn_f = raster_fn.replace('.tiff', '') + '_rs.tiff'
                ds = gdal.Translate(raster_fn_f, raster_fn, xRes=res[0], yRes=res[1], resampleAlg=res_method)
                ds = None
                if crop:
                    geo.mask_raster(paths.nz_coastlines_2193, raster_fn_f, raster_fn_f, all_touched=False, crop=False)
            else:
                raster_fn_f = raster_fn
                if crop:
                    geo.mask_raster(paths.nz_coastlines_2193, raster_fn_f, raster_fn_f, all_touched=False, crop=False)
            raster2vti_names.append(raster_fn_f)

        # Creates VTI Image from the raster file
        image_filename = join(self.dirs['vti'], filename + '.vti')
        _ = geo.rasters2vti(image_filename, raster2vti_names, names, offset=10)

    def load_data(self, filename=None):
        """
        Loads a serialized Model_results object
        :param filename:
        :return:
        """
        if filename is None:
            filename = join(self.dirs['output'], 'data.obj')

        with open(filename, 'rb') as f:
            A = pickle.load(f)
            self.hmaps = A[0]
            self.immp = A[1]
            self.hcurves_stats = A[2]
            self.hmaps_stats = A[3]

            self.ks1_curve = A[4]
            self.ks1_map = A[5]
            self.chi_curve = A[6]
            self.chi_map = A[7]

    def save_data(self, filename=None):
        """
        Serializes Model_results object into a file
        :param filename: If None, save in results folder named with self.name
        """
        if filename is None:
            filename = join(self.dirs['output'], 'data.obj')

        with open(filename, 'wb') as hazardobj:
            Data = (self.hmaps, self.immp, self.hcurves_stats, self.hmaps_stats,
                    self.ks1_curve, self.ks1_map, self.chi_curve, self.chi_map)
            pickle.dump(Data, hazardobj)


if __name__ == '__main__':

    pass