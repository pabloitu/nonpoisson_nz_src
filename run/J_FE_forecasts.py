import os
import time

from nonpoisson.zonation import GeodeticModel
from nonpoisson.temporal import NegBinom, Poisson
from nonpoisson.forecast import forecastModel, get_tvzpolygon
from nonpoisson import paths
from nonpoisson import catalogs

import numpy as np
from datetime import datetime as dt


def make_models_FE(N, years, bval, folder='', vti=False,
                        write_forecast=True):

    res = (1000, 1000)
    crs = 'EPSG:2193'
    bins = 3
    metric = 'j2'

    fig14_path = os.path.join(paths.ms1_figs['fig14'], 'paraview')
    os.makedirs(fig14_path, exist_ok=True)

    poisson = Poisson()
    spatial = GeodeticModel.load('hw_final')
    spatial.bins_polygonize([metric], [bins], load=True, post_proc=True)
    # spatial.intersect_by_polygon(paths.region_nz_buff, 'j2', 3)

    catalog = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(4.0, 10.0),
                                  depth=(40, -2),
                                  start_time=dt(1964, 1, 1), end_time=None,
                                  shapefile=paths.region_nz_test)

    testing_years = catalog.end_year - catalog.start_year + 1
    names = ['m']
    hybrid_paths = [paths.MULTOT1346GRU]

    hybrid = forecastModel(names[0], folder=folder, time_span=testing_years)
    hybrid.get_geometry()
    hybrid.get_rate_from_hybrid(hybrid_paths[0])
    hybrid.set_mfd(bval)
    hybrid.normalize(N * years)

    pua = forecastModel(f'pua', folder=folder, mkdirs=True)
    pua.get_geometry()
    pua.set_rate_from_models(poisson, spatial, catalog, measure=metric,
                             nbins=bins, target_mmin=5.0)
    pua.gr_scale(bval, 1.2, get_tvzpolygon(spatial, metric, bins))
    pua.set_mfd(bval)
    pua.normalize(N * years)

    fe = forecastModel.floor_2models(f'fe', hybrid, pua, bin=None,
                                         floor_type='count', folder=folder)
    fe.set_mfd(bval)
    fe.normalize(N*years)

    fe_low = forecastModel.floor_2models(f'fe_low', hybrid, pua, bin=0,
                                     floor_type='count', folder=folder)
    fe_low.fill_towards(8)
    fe_low.set_mfd(bval)
    fe_low.normalize(N*years)


    fig_path = paths.ms1_figs['fig16']
    os.makedirs(os.path.join(fig_path, 'paraview'), exist_ok=True)

    if write_forecast:
        hybrid.write_forecast()
        pua.write_forecast()
        fe.write_forecast()
        fe_low.write_forecast()

    if vti:
        hybrid.normalize()
        pua.normalize()
        fe.normalize()
        fe_low.normalize()
        hybrid.write_vti(path=os.path.join(fig_path, 'paraview', f'hybrid.vti'),
                         res=res, epsg=crs, crop=True, res_method='nearest')
        pua.write_vti(path=os.path.join(fig_path, 'paraview', f'pua.vti'),
                      res=res, epsg=crs, crop=True, res_method='nearest')
        fe.write_vti(path=os.path.join(fig_path, 'paraview', f'fe.vti'),
                     res=res, epsg=crs, crop=True, res_method='nearest')
        fe_low.write_vti(path=os.path.join(fig_path, 'paraview', f'fe_low.vti'),
                         res=res, epsg=crs, crop=True, res_method='nearest')



if __name__ == '__main__':

    make_models_FE(5, 50, 0.925,
                   vti=True, write_forecast=True)
