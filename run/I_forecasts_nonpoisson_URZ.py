from nonpoisson import zonation, temporal, catalogs, paths
from nonpoisson.forecast import forecastModel, get_tvzpolygon
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from os.path import join
import seaborn as sns
import os
sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})


def create_forecast_nb(bval=0.925, bval_tvz=1.2, vti=True):

    crs = 'EPSG:2193'
    res = (1200, 1200)

    temporal_model = temporal.NegBinom.load('negbinom_nz')
    ax = temporal_model.plot_params()
    temporal_model.extend_params(1000, 1000, 10, plot=True, limit=False, axes=ax)
    temporal_model.save(paths.get('spatial', 'serial', 'negbinom_nz_ext'))

    fig_path = paths.ms1_figs['fig15']
    plt.savefig(join(fig_path, 'nb_params.png'), dpi=1500, bbox_inches='tight')
    plt.show()

    spatial_model = zonation.GeodeticModel.load('hw_final')
    spatial_model.bins_polygonize(['j2'], [3], load=True, post_proc=True)
    # # spatial.intersect_by_polygon(paths.region_nz_buff, 'j2', 3)
    catalog = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(4.0, 10.0),
                                  depth=(40, -2),
                                  start_time=dt(1964, 1, 1), end_time=None,
                                  shapefile=paths.region_nz_test)

    forecast = forecastModel('nb', folder=folder,
                                      time_span=50)
    forecast.get_geometry()
    forecast.set_rate_from_models(temporal_model, spatial_model, catalog,
                                  measure='j2', nbins=3, target_mmin=5.0)
    forecast.gr_scale(bval, bval_tvz, get_tvzpolygon(spatial_model, 'j2', 3))
    forecast.set_mfd(bval)
    forecast.normalize()

    os.makedirs(join(fig_path, 'paraview'), exist_ok=True)
    if vti:
        forecast.write_vti(path=os.path.join(fig_path, 'paraview', f'npua_3.vti'),
                           res=res, epsg=crs, crop=True, res_method='nearest')

if __name__ == '__main__':

    # create_temporal_models()
    folder = 'forecast_temporal'
    create_forecast_nb()
    # create_forecast_ln(folder)
    # create_forecast_filtered(folder)

    # plot_models(folder, cities)

    sns.reset_defaults()
