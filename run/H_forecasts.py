import os
import time

from nonpoisson.zonation import GeodeticModel
from nonpoisson.temporal import NegBinom, Poisson
from nonpoisson.forecast import forecastModel, get_tvzpolygon
from nonpoisson import paths
from nonpoisson import catalogs

import numpy as np
from datetime import datetime as dt


def make_models_poisson(N, years, bval, folder='', vti=False,
                        write_forecast=True):

    res = (1500, 1500)
    crs = 'EPSG:2193'
    bin_array = [3, 4, 5]
    metric = 'j2'

    fig14_path = os.path.join(paths.ms1_figs['fig14'], 'paraview')
    os.makedirs(fig14_path, exist_ok=True)

    poisson = Poisson()
    spatial = GeodeticModel.load('hw_final')
    spatial.bins_polygonize([metric], bin_array, load=True)
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
    hybrid.normalize()

    if vti:
        hybrid.write_vti(path=os.path.join(fig14_path, 'hybrid.vti'),
                         res=res, epsg=crs, crop=True, res_method='nearest')

    hybrid.normalize(N * years)
    if write_forecast:
        hybrid.write_forecast()

    for bins in bin_array:
        pua = forecastModel(f'pua_{bins}', folder=folder, mkdirs=True)
        pua.get_geometry()
        pua.set_rate_from_models(poisson, spatial, catalog, measure=metric,
                                 nbins=bins, target_mmin=5.0)
        pua.gr_scale(bval, 1.2, get_tvzpolygon(spatial, metric, bins))
        pua.set_mfd(bval)
        pua.normalize()
        if vti:
            pua.write_vti(path=os.path.join(fig14_path, f'pua_{bins}.vti'),
                          res=res, epsg=crs, crop=True, res_method='nearest')
        pua.normalize(N * years)
        if write_forecast:
            pua.write_forecast()


if __name__ == '__main__':

    make_models_poisson(5, 50, 0.925,
                        vti=True, write_forecast=True)


    # from nonpoisson import geo

    # auckland = paths.Auckland
    # array = np.array([[174.5, -37.0], [174.5, -36.5], [175.0, -36.5], [175.0, -37.0]])
    # a = geo.reproject_coordinates(auckland, 'epsg:4326', 'epsg:2193')
