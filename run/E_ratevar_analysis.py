from nonpoisson import paths
from nonpoisson import catalogs
from nonpoisson.temporal import CatalogAnalysis

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from os.path import join

sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})

time_folder = paths.results_path['temporal']['dir']
etas_folder = join(time_folder, 'etas')
figure_folder = join(time_folder, 'figures')
fn_store_simulation = join(etas_folder, 'simulated_catalog.csv')

N_MAX = 600
N_ITER = 2000
ETAS_NSIMS = 1000
RATE_VAR_PARAMS = {'n_disc': np.arange(1, N_MAX),
                   'max_iters': N_ITER}


def rate_var_regions():

    cat_nz = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(3.99, 10),
                                 depth=(40, -2),
                                 start_time=dt(1980, 1, 1),
                                 shapefile=paths.region_nz_collection)
    cat_japan = catalogs.filter_cat(catalogs.get_cat_japan(),
                                    start_time=dt(1985, 1, 1),
                                    end_time=dt(2011, 1, 1),
                                    shapefile=paths.region_japan,
                                    mws=(3.99, 10),
                                    depth=(30, -2))
    cat_ca = catalogs.filter_cat(catalogs.get_cat_ca(query=False),
                                 start_time=dt(1962, 1, 1),
                                 mws=(3.99, 10),
                                 depth=(30, -2),
                                 )
    cat_it = catalogs.filter_cat(catalogs.get_cat_it(),
                                 start_time=dt(1960, 1, 1),
                                 mws=(3.99, 10),
                                 depth=(30, -2))
    cat_globe = catalogs.filter_cat(catalogs.get_cat_global(),
                                    start_time=dt(1990, 1, 1),
                                    mws=(5.99, 10),
                                    depth=(70, -2))

    for cat in [cat_nz, cat_japan, cat_it, cat_ca, cat_globe][2:3]:
        analysis = CatalogAnalysis(cat, name=cat.name, params=RATE_VAR_PARAMS)
        analysis.get_ratevar()
        analysis.cat_var.get_stats()
        analysis.cat_var.purge()
        analysis.save()


def rate_var_etas():

    n_iter_etas = 500
    cats = catalogs.get_cat_etas(fn_store_simulation)

    for cat in cats:
        catalogs.filter_cat(cat, mws=(3.99, 10))
        analysis = CatalogAnalysis(cat, name=cat.name,
                                   params={'n_disc': np.arange(1, N_MAX),
                                           'max_iters': n_iter_etas})
        analysis.get_ratevar()
        analysis.cat_var.get_stats()
        analysis.cat_var.purge()
        analysis.save()


def fig_ratevar(figpath, nsims=1000, savefig=True):

    n_max = 400
    regions = ['nz', 'japan', 'california', 'italy', 'global']
    names = ['New Zealand', 'Japan', 'California', 'Italy', 'Globe']
    colors = ['steelblue', 'orange', 'green', 'red', 'purple']
    analyses = [CatalogAnalysis.load(paths.get('temporal', 'serial', i))
                for i in regions]
    etas = [CatalogAnalysis.load(paths.get('temporal', 'serial', f'etas_{i}'))
            for i in range(nsims)]

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    # Mean plot
    etas_025 = [np.quantile([i.cat_var.stats['mean'][j] for i in etas], 0.025)
                for j in range(n_max)]
    etas_0975 = [np.quantile([i.cat_var.stats['mean'][j] for i in etas], 0.975)
                 for j in range(n_max)]
    axs[0].fill_between(np.arange(n_max), etas_025, etas_0975, alpha=0.2,
                        label=r'ETAS - Sim. $95\%$ envelope ')
    for i, j, k in zip(analyses, names, colors):
        i.cat_var.plot_stats('mean', ax=axs[0], label=j, color=k, linewidth=1)
    axs[0].plot(np.arange(0, 1.2*n_max), np.arange(0, 1.2*n_max),
                color='black', linestyle='--', linewidth=1, label='Poisson')
    axs[0].set_xlim([0, n_max])
    axs[0].set_ylim([0, 1.2*n_max])
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlabel(r'$N_1$', fontsize=14)
    axs[0].set_ylabel(r'$\mathrm{Mean}(N_2|N_1)$', fontsize=14)
    axs[0].set_title(r'$\bf{a\,)}$', fontsize=12, loc='left')

    # Median plot
    etas_025 = [np.quantile([i.cat_var.stats['median'][j] for i in etas],
                            0.025) for j in range(n_max)]
    etas_0975 = [np.quantile([i.cat_var.stats['median'][j] for i in etas],
                             0.975) for j in range(n_max)]
    axs[1].fill_between(np.arange(n_max), etas_025, etas_0975, alpha=0.2,
                        label=r'ETAS - Sim. $95\%$ envelope ')
    for i, j, k in zip(analyses, names, colors):
        i.cat_var.plot_stats('median', ax=axs[1],
                             label=j, color=k, linewidth=1)
    axs[1].plot(np.arange(0, 1.2*n_max), np.arange(0, 1.2*n_max),
                color='black', linestyle='--', linewidth=1, label='Poisson')
    axs[1].set_xlim([0, n_max])
    axs[1].set_ylim([0, 1.2*n_max])
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlabel(r'$N_1$', fontsize=14)
    axs[1].set_ylabel(r'$\mathrm{Med}(N_2|N_1)$', fontsize=14)
    axs[1].set_title(r'$\bf{b\,)}$', fontsize=12, loc='left')

    # Standard Dev. plot
    for i, j, k in zip(analyses, names, colors):
        i.cat_var.plot_stats('std', ax=axs[2], label=j, color=k, linewidth=1)
    axs[2].plot(np.arange(0, 1.2*n_max), np.sqrt(np.arange(0, 1.2*n_max)),
                color='black', linestyle='--', linewidth=1, label='Poisson')
    etas_025 = [np.quantile([i.cat_var.stats['var'][j]**0.5 for i in etas],
                            0.025) for j in range(n_max)]
    etas_0975 = [np.quantile([i.cat_var.stats['var'][j]**0.5 for i in etas],
                             0.975) for j in range(n_max)]
    axs[2].fill_between(np.arange(n_max), etas_025, etas_0975, alpha=0.2,
                        label=r'ETAS NZ - $95\%$')

    axs[2].set_xlim([0, n_max])
    axs[2].set_ylim([0, 1.2*n_max])
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].set_xlabel(r'$N_1$', fontsize=14)
    axs[2].set_ylabel(r'$\mathrm{Std}(N_2|N_1)$', fontsize=14)
    axs[2].set_title(r'$\bf{c\,)}$', fontsize=12, loc='left')
    axs[2].legend(ncol=2, columnspacing=0.8, fontsize=10.5)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    if savefig:
        plt.savefig(figpath, dpi=300)
    plt.show()


if __name__ == '__main__':

    rate_var_regions()
    rate_var_etas()
    ms_figpath = join(paths.ms1_figs['fig11'], f'rate_variability.png')
    fig_ratevar(ms_figpath, nsims=ETAS_NSIMS, savefig=True)
