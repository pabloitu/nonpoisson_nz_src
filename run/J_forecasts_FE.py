import os
import time

from nonpoisson.zonation import GeodeticModel
from nonpoisson.temporal import NegBinom, Poisson
from nonpoisson.forecast import forecastModel, get_tvzpolygon
from nonpoisson import paths
from nonpoisson import catalogs

import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from os.path import join
import seaborn as sns
def make_models_FE(N, years, bval, folder='', vti=False,
                        write_forecast=True):

    res = (1000, 1000)
    crs = 'EPSG:2193'
    bins = 3
    metric = 'j2'

    fig14_path = os.path.join(paths.ms1_figs['fig14'], 'paraview')
    os.makedirs(fig14_path, exist_ok=True)

    poisson = Poisson()
    negbinom = NegBinom.load('negbinom_nz')
    spatial = GeodeticModel.load('hw_final')
    spatial.bins_polygonize([metric], [bins], load=True, post_proc=True)
    spatial.intersect_by_polygon(paths.region_nz_buff, 'j2', 3)

    catalog = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(4.99, 10.0),
                                  depth=(40, -2),
                                  start_time=dt(1950, 1, 1),
                                  end_time=None,
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

    npua = forecastModel(f'npua', folder=folder, mkdirs=True)
    npua.get_geometry()
    npua.set_rate_from_models(negbinom, spatial, catalog, measure=metric,
                              nbins=bins, target_mmin=5.0)
    npua.gr_scale(bval, 1.2, get_tvzpolygon(spatial, metric, bins))
    npua.set_mfd(bval)
    npua.normalize(N * years)


    fe = forecastModel.floor_2models(f'fe', hybrid, pua, bin=None,
                                         floor_type='count', folder=folder)
    fe.set_mfd(bval)
    fe.normalize(N*years)

    fe_low = forecastModel.floor_2models(f'fe_low', hybrid, pua, bin=0,
                                     floor_type='count', folder=folder)
    fe_low.fill_towards(8)
    fe_low.set_mfd(bval)
    fe_low.normalize(N*years)


    npfe = forecastModel.floor_2models(f'npfe', hybrid, npua, bin=None,
                                           floor_type='count', folder=folder)
    npfe.set_mfd(bval)
    npfe.normalize(N*years)


    npfe_low = forecastModel.floor_2models(f'npfe', hybrid, npua, bin=0,
                                       floor_type='count', folder=folder)
    npfe_low.fill_towards(8)
    npfe_low.set_mfd(bval)
    npfe_low.normalize(N*years)

    fig_path = paths.ms1_figs['fig16']
    os.makedirs(os.path.join(fig_path, 'paraview'), exist_ok=True)

    if write_forecast:
        hybrid.write_forecast()
        pua.write_forecast()
        fe.write_forecast()
        fe_low.write_forecast()
        npfe.write_forecast()
        npfe_low.write_forecast()


    if vti:
        hybrid.normalize()
        pua.normalize()
        fe.normalize()
        fe_low.normalize()
        npfe.normalize()
        npfe_low.normalize()
        # hybrid.write_vti(path=os.path.join(fig_path, 'paraview', f'hybrid.vti'),
        #                  res=res, epsg=crs, crop=True, res_method='nearest')
        # pua.write_vti(path=os.path.join(fig_path, 'paraview', f'pua.vti'),
        #               res=res, epsg=crs, crop=True, res_method='nearest')
        # fe.write_vti(path=os.path.join(fig_path, 'paraview', f'fe.vti'),
        #              res=res, epsg=crs, crop=True, res_method='nearest')
        # fe_low.write_vti(path=os.path.join(fig_path, 'paraview', f'fe_low.vti'),
        #                  res=res, epsg=crs, crop=True, res_method='nearest')
        npfe_low.write_vti(path=os.path.join(fig_path, 'paraview', f'npfe_low.vti'),
                           res=res, epsg=crs, crop=True, res_method='nearest')
        hybrid.normalize(N*years)
        pua.normalize(N*years)
        fe.normalize(N*years)
        fe_low.normalize(N*years)
        fe_low.save()
        npfe_low.normalize(N*years)
        npfe_low.save()
        npfe.normalize(N*years)
        npfe.save()

    npua.save()
    fe.save()
    fe_low.save()
    npfe_low.save()
    npfe.save()

    return hybrid, pua, fe, fe_low, npfe, npfe_low


def plot_rate_cities(p, h, f, nf):
    sns.set_style("darkgrid", {"ytick.left" : True,
        "axes.facecolor": ".9", "axes.edgecolor": "0",
        'axes.linewidth': '1', 'font.family': 'Ubuntu'})
    N = 4.6
    years = 50
    i = 0
    cities = ['Auckland', 'Tauranga', 'Gisborne', 'Napier', 'Wellington', 'Christchurch', 'Queenstown', 'Dunedin', 'Invercargill']
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for city in cities:
        print(city, f.get_rate(city)/N/years)
        ax.semilogy([4 * i], [h.get_rate(city)/N/years], 'o', color='steelblue')
        ax.semilogy([4 * i + 2], [p.get_rate(city)/N/years], 'o', color='red')
        ax.semilogy([4 * i + 1], [f.get_rate(city)/N/years], 'o', color='purple')
        ax.semilogy([4 * i + 3], [nf.get_rate(city)/N/years], 'o', color='green')
        i += 1

    a = []
    for i in cities:
        a.extend([None, i, None, None])
    ax.set_xticks(np.arange(0, 4*len(cities)))
    ax.set_xticklabels(a, rotation=45, fontsize=12)
    xTickPos = ax.get_xticks()
    xTickPos = xTickPos[:-1]
    ax.bar(xTickPos[::4] +1.5, 2000 * np.array([60] * len(xTickPos[::4])),
           bottom=-1000, width=4 * (xTickPos[1] - xTickPos[0]), color=['w', 'gray'], alpha=0.3)
    ax.set_ylim([1e-6, 1e-3])
    ax.set_xlim([-0.5, 4*len(cities) - 0.5])
    ax.grid(axis='y', which='major', linewidth=3)
    ax.grid(axis='y', which='minor')

    # ax.tick_params(which='major', axis='x', length=0, color='gray')
    ax.tick_params(which='major', axis='y', length=10, color='gray', width=1)
    ax.tick_params(which='minor', axis='y', length=5, color='gray', width=1)
    # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel('Forecast mean rates - $\mu(x)$ \nEvents $M\geq 4.95$',
                  fontsize=15)

    legend_elements = [Line2D([0], [0], color='steelblue', lw=0, marker='o', label=r'Hybrid Multiplicative'),
                       Line2D([0], [0], color='red', lw=0, marker='o', label=r'Poisson URZ'),
                       Line2D([0], [0], color='purple', lw=0, marker='o', label=r'Poisson FE'),
                       Line2D([0], [0], color='green', lw=0, marker='o', label=r'Negbinom FE')]

    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(join(paths.ms1_figs['fig17'], 'forecast_values.png'), dpi=300, bbox_inches='tight')
    plt.savefig(join(paths.ms1_figs['fig17'], 'fig17.png'), dpi=1200, bbox_inches='tight')
    plt.show()
    sns.reset_defaults()

if __name__ == '__main__':

    hybrid, pua, fe, fe_low, npfe, npfe_low = make_models_FE(5, 50, 0.925,
                              vti=False, write_forecast=True)

    plot_rate_cities(pua, hybrid, fe, npfe)
