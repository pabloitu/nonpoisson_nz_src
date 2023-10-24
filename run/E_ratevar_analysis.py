from nonpoisson import paths
from nonpoisson import catalogs
from nonpoisson.temporal import CatalogAnalysis

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from datetime import datetime as dt
from os.path import join

sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})

time_folder = paths.results_path['temporal']['dir']
etas_folder = join(time_folder, 'etas')
figure_folder = join(time_folder, 'figures')
fn_store_simulation = join(etas_folder, 'simulated_catalog.csv')

N_MAX = 600
N_ITER = 2000
RATE_VAR_PARAMS = {'n_disc': np.arange(1, N_MAX),
                   'max_iters': N_ITER}


def rate_var_regions():

    cat_nz = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(3.99, 10),
                                 depth=(40, -2),
                                 start_time=dt(1980, 1, 1),
                                 shapefile=paths.region_nz_collection)
    cat_japan = catalogs.filter_cat(catalogs.get_cat_japan(),
                                    start_time=dt(1985, 1, 1),
                                    shapefile=paths.region_japan,
                                    mws=(3.99, 10),
                                    depth=(30, -2))
    cat_ca = catalogs.filter_cat(catalogs.get_cat_ca(query=False),
                                 start_time=dt(1981, 1, 1),
                                 mws=(3.99, 10),
                                 depth=(30, -2),
                                 )
    cat_it = catalogs.filter_cat(catalogs.get_cat_it(),
                                 start_time=dt(1960, 1, 1),
                                 shapefile=paths.region_it,
                                 mws=(3.99, 10),
                                 depth=(30, -2))
    cat_globe = catalogs.filter_cat(catalogs.get_cat_global(),
                                    start_time=dt(1990, 1, 1),
                                    mws=(5.99, 10),
                                    depth=(70, -2))

    for cat in [cat_nz, cat_japan, cat_it, cat_ca, cat_globe]:
        analysis = CatalogAnalysis(cat, name=cat.name, params=RATE_VAR_PARAMS)
        analysis.get_ratevar()
        analysis.cat_var.get_stats()
        analysis.cat_var.purge()
        analysis.save()


def rate_var_etas():

    cats = catalogs.get_cat_etas(fn_store_simulation)

    for cat in cats:
        catalogs.filter_cat(cat, mws=(3.99, 10))
        analysis = CatalogAnalysis(cat, name=cat.name,
                                   params={'n_disc': np.arange(1, N_MAX),
                                           'max_iters': N_ITER})
        analysis.get_ratevar()
        analysis.cat_var.get_stats()
        analysis.cat_var.purge()
        analysis.save()

    return analysis


def fig_regions_stats(figure_folder, savefig=True):

    n_max = 300
    nz = CatalogAnalysis.load(paths.get('temporal', 'serial', 'new_zealand'))
    japan = CatalogAnalysis.load(paths.get('temporal', 'serial', 'japan'))
    california = CatalogAnalysis.load(paths.get('temporal', 'serial', 'california'))
    italy = CatalogAnalysis.load(paths.get('temporal', 'serial', 'italy'))
    globe = CatalogAnalysis.load(paths.get('temporal', 'serial', 'globe'))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    nz.cat_var.plot_stats('mean', label='New Zealand', color='steelblue', linewidth=1.0)
    japan.cat_var.plot_stats('mean', label='Japan', color='orange')
    california.cat_var.plot_stats('mean', label='California', color='green')
    globe.cat_var.plot_stats('mean', label='Globe', color='red')
    italy.cat_var.plot_stats('mean', label='Italy', color='purple')
    ax.plot(np.arange(0, 1.2*n_max), np.arange(0, 1.2*n_max), color='black', linestyle='--', linewidth=1,
            label='Poisson')
    ax.legend()
    n_max = 300
    ax.set_xlim([0, n_max])
    ax.set_ylim([0, 1.2*n_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$N_1=m$')
    ax.set_ylabel(r'$\mathrm{E}(N_2|N_1=m)$')
    plt.tight_layout()
    if savefig:
        plt.savefig(join(figure_folder, f'regions_mean.png'), dpi=300)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nz.cat_var.plot_stats('median', label='New Zealand', color='steelblue', linewidth=1.0)
    japan.cat_var.plot_stats('median', label='Japan', color='orange')
    california.cat_var.plot_stats('median', label='California', color='green')
    globe.cat_var.plot_stats('median', label='Globe', color='red')
    italy.cat_var.plot_stats('median', label='Italy', color='purple')
    ax.plot(np.arange(0, 1.2*n_max), np.arange(0, 1.2*n_max), color='black', linestyle='--', linewidth=1,
            label='Poisson')
    ax.set_xlim([0, n_max])
    ax.set_ylim([0, 1.2*n_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$N_1=m$')
    ax.set_ylabel(r'$\mathrm{Med}(N_2|N_1=m)$')
    plt.tight_layout()
    if savefig:
        plt.savefig(join(figure_folder, f'regions_median.png'), dpi=300)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nz.cat_var.plot_stats('std', label='New Zealand', color='steelblue', linewidth=1.0)
    japan.cat_var.plot_stats('std', label='Japan', color='orange')
    california.cat_var.plot_stats('std', label='California', color='green')
    globe.cat_var.plot_stats('std', label='Globe', color='red')
    italy.cat_var.plot_stats('std', label='Italy', color='purple')
    ax.plot(np.arange(0, 1.2*n_max), np.sqrt(np.arange(0, 1.2*n_max)), color='black', linestyle='--', linewidth=1,
            label='Poisson')
    ax.set_xlim([0, n_max])
    ax.set_ylim([0, 2*n_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$N_1=m$')
    ax.set_ylabel(r'$\mathrm{Std}(N_2|N_1=m)$')
    plt.tight_layout()
    if savefig:
        plt.savefig(join(figure_folder, f'regions_std.png'), dpi=300)
    plt.show()


def fig_etas_stats(figure_folder, nsims=10, savefig=True):

    n_max = 300

    nz = CatalogAnalysis.load(paths.get('temporal', 'serial', 'new_zealand'))
    etas = [CatalogAnalysis.load(paths.get('temporal', 'serial', f'etas_{i}'))
            for i in range(nsims)]

    print('stats calculated')
    etas_means = [np.mean([i.cat_var.stats['mean'][j] for i in etas])
                   for j in range(N_MAX)]
    etas_5 = [np.quantile([i.cat_var.stats['mean'][j] for i in etas], 0.025)
              for j in range(N_MAX)]
    etas_95 = [np.quantile([i.cat_var.stats['mean'][j] for i in etas], 0.975)
              for j in range(N_MAX)]

    print('overall stats calculated')

    # plt.plot(etas_means, label='mean')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nz.cat_var.plot_stats('mean', label='New Zealand', color='red', linewidth=1.0)
    ax.plot(etas_means, color='steelblue', linewidth=0.6,
             label=r'ETAS - Sim. mean')

    ax.fill_between(np.arange(n_max), etas_5, etas_95, alpha=0.2,
                     label=r'ETAS - Sim. $95\%$ envelope ')
    ax.plot(np.arange(0, 1.2*n_max), np.arange(0, 1.2*n_max),
             color='black', linestyle='--', linewidth=1, label='Poisson')
    ax.legend()
    ax.set_xlim([0, n_max])
    ax.set_ylim([0, 1.2*n_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$N_1=m$')
    ax.set_ylabel(r'$\mathrm{E}(N_2|N_1=m)$')
    plt.tight_layout()
    if savefig:
        plt.savefig(join(figure_folder, f'etas_mean.png'), dpi=300)
    plt.show()


if __name__ == '__main__':

    fig_folder = paths.ms1_figs['fig11']
    rate_var_regions()
    rate_var_etas()
    # fig_regions_stats(fig_folder, savefig=True)
    # fig_etas_stats(fig_folder, nsims=10, savefig=True)




