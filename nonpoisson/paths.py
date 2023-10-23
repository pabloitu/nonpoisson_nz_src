from os import makedirs
from os.path import join, dirname, realpath
import glob
import numpy as np

# Main folders
codes = dirname(realpath(__file__))
main = dirname(codes)
data = join(main, 'data')
results = join(main, 'results')
export = join(main, 'export')

# Raw data subfolders
basemaps = join(data, 'basemaps')
catalogs = join(data, 'catalogs')
gpsmodels = join(data, 'gnss_models')
forecasts = join(data, 'forecast_models')
regions = join(data, 'regions')
hazardmodels = join(data, 'hazard_models')

# Basemaps
basemap_bluebrown = join(basemaps, 'bluebrown.tiff')
nz_coastlines_2193 = join(basemaps, 'new_zealand', 'polygons', 'nz_poly.shp')

# Regions
region_tvz = join(regions, 'tvz/tvz.txt')
region_tvz_corr = join(regions, 'tvz/tvzcorr.dat')
region_japan = join(regions, 'japan/japan_region.shp')
region_nz_collection = join(regions, 'new_zealand/nz_collection.shp')
region_nz_test = join(regions, 'new_zealand/nz_testing.shp')
region_nz_buff = join(regions, 'new_zealand/nz_testing_buff.shp')

region_nz_test_2193 = join(regions, 'new_zealand/nz_testing_2193.shp')
region_it = join(regions, 'italy/italy_region.shp')
csep_nz_grid = join(regions, 'csep_grid.txt')
csep_testing = join(regions, 'new_zealand/csep_testing.csv')

# Geodetic Model
points = join(gpsmodels, 'points.csv')
gps_source = join(gpsmodels, 'source')
gps_processed = join(gpsmodels, 'processed')

## Final models batch
gps_secondmodels = join(gps_source, 'new_models')
gps_hw_final = join(gps_secondmodels, 'VDoHS_solution_corr_sill.csv')

## Processed models
model_names = ['hw_final']
gps_proc_models = {i: join(gps_processed, '%s.csv' % i) for i in model_names}

# Catalogs
cat_global = join(catalogs, 'cat_global.csv')
cat_nz = join(catalogs, 'cat_nz.csv')
cat_japan = join(catalogs, 'cat_japan.csv')
cat_ca = join(catalogs, 'cat_ca.csv')
cat_it = join(catalogs, 'cat_it.csv')
cats_etas = join(catalogs, 'syn_etas_nz')

# cat_nz_dcz = join(catalogs, 'cat_nz_dc.csv')
# cat_japan_dcz = join(catalogs, 'cat_japan_dc.csv')
# cat_ca_dcz = join(catalogs, 'cat_ca_dc.csv')
# cat_it_dcz = join(catalogs, 'cat_it_dc.csv')

# Forecasts
ADDTOT23456GRuEEPAScomb = join(forecasts, 'ADDTOT23456GRuEEPAScomb.csv')
MULTOT123456GruEEPAScomb = join(forecasts, 'MULTOT123456GruEEPAScomb.csv')
MULTOT123456r1 = join(forecasts, 'MULTOT123456r1.csv')
MULTOT1346GRU = join(forecasts, 'MULTOT1346GRU.csv')
MULTOT1346GruEEPAScomb = join(forecasts, 'MULTOT1346GruEEPAScomb.csv')
ADDTOT346ave = join(forecasts, 'ADDTOT346ave.csv')
ADDTOT346aveEEPAScomb = join(forecasts, 'ADDTOT346aveEEPAScomb.csv')
GrandADDTOTopti = join(forecasts, 'GrandADDTOTopti.csv')
AddoptiEEPAScomb = join(forecasts, 'AddoptiEEPAScomb.csv')


# Results
results_path = {'catalogs': {'dir': join(results, 'catalogs'),
                             'fig': None,
                             'csv': None},
                'temporal': {'dir': join(results, 'temporal',),
                             'serial': None,
                             'csv': None},
                'spatial': {'dir': join(results, 'spatial')},
                'hazard': {'dir': join(results, 'hazard')}}

subfolders = ['fig', 'csv', 'raster', 'shp', 'serial', 'vtk']
subfolder_formats = ['png', 'csv', 'tiff', 'shp', 'pickle', 'vti']
extensions = {i: j for i, j in zip(subfolders, subfolder_formats)}

for key, val in results_path.items():
    for subfolder in subfolders:
        if subfolder in [i[0] for i in list(val.items())]:
            val[subfolder] = join(val['dir'], subfolder)
            makedirs(val[subfolder], exist_ok=True)


def get(model, result_type, name, ext=None):
    if ext is None:
        ext = extensions[result_type]
    if ext:
        return join(results_path[model][result_type], '%s.%s' % (name, ext))
    else:
        return join(results_path[model][result_type], '%s' % name)


def get_oq(name):
    oq_dir = join(results, 'hazard', 'oq')
    if isinstance(name, str):
        return join(oq_dir, name)
    elif isinstance(name, (list, tuple)):
        return join(oq_dir, *name)


def get_oqcalc(num, folder=None):

    if folder:
        h5_path = join(folder, 'calc_%i.hdf5' % num)
    else:
        h5_path = join('/home/pciturri/oqdata/', 'calc_%i.hdf5' % num)
    return h5_path


def get_oqjobid(name):
    results_dir = join(get_oq(name), 'results')
    path = glob.glob1(results_dir, 'realizations*')
    ids = [int(id.split('.')[0].split('_')[-1]) for id in path]
    return max(ids)


# Hazard Models

input_oqfiles = join(hazardmodels, 'openquake_files')
base_sm_ltree = join(input_oqfiles, 'base_source_lt.xml')
gmpe_lt_1 = join(input_oqfiles, 'gmpe_1.xml')
gmpe_lt_2 = join(input_oqfiles, 'gmpe_2.xml')
gmpe_lt_5 = join(input_oqfiles, 'gmpe_5.xml')
job_ses = join(input_oqfiles, 'job_ses.ini')
job_classical = join(input_oqfiles, 'job_classical.ini')


def replace_path(dict_, suffix=''):
    new_dict = {}
    if suffix:
        suffix = '_' + suffix
    for key, item in dict_.items():
        path, extension = item.split('.')
        new_dict[key] = path + suffix + '.' + extension
    return new_dict


Auckland = np.array([[174.7, -36.8]])  # < AUCKLAND
Dunedin = np.array([[170.5, -45.9]])  ### < DUNEDIN
Wellington = np.array([[174.8, -41.3]])  # # wellington
Christchurch = np.array([[172.6, -43.5]])  # < AUCKLAND
Queenstown = np.array([[168.1, -45.0]])
Napier = np.array([[176.9, -39.5]])
Tauranga = np.array([[176.2, -37.7]])
Gisborne = np.array([[178.0, -38.7]])
Invercargill = np.array([[168.4, -46.4]])

ms_figroot = join(main, 'figures')
forecast_ms = join(ms_figroot, 'forecasts_ms')
hazard_ms = join(ms_figroot, 'hazard_ms')

ms1_figs = {f'fig{i}': join(forecast_ms, f'figure{i}') for i in range(1, 23)}
ms2_figs = {f'fig{i}': join(hazard_ms, f'figure{i}') for i in range(1, 13)}


for fold in ms1_figs.values():
    makedirs(fold, exist_ok=True)
for fold in ms2_figs.values():
    makedirs(fold, exist_ok=True)
