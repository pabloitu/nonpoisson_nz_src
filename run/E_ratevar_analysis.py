import os

from nonpoisson import paths
from nonpoisson import catalogs
from nonpoisson import temporal
from nonpoisson.temporal import CatalogAnalysis

import numpy as np
import random
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
ETAS_NSIMS = 1000
RATE_VAR_PARAMS = {'n_disc': np.arange(1, N_MAX),
                   'max_iters': N_ITER}


def fig_sampling(figpath):

    np.random.seed(32)
    random.seed(32)

    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    cat_nz = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(3.99, 10),
                                 depth=(40, -2),
                                 start_time=dt(1960, 1, 1),
                                 shapefile=paths.region_nz_collection)
    cat_nz.sort_catalogue_chronologically()
    nz_analysis = CatalogAnalysis(cat_nz, name='nz',
                                  params={'n_disc': np.arange(1, N_MAX),
                                          'max_iters': N_ITER})
    nz_analysis.get_ratevar()

    model_i = temporal.backwardPoisson()
    model_i.get_params(nz_analysis)
    model_i.simulate(nsims=N_ITER)

    color = 'red'
    msize = 0.005
    ksize = 5
    n2_ylims = [0, 1100]
    ration2_ylims = [-1.3, 1.3]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8),
                            constrained_layout=True,
                            gridspec_kw={'height_ratios': [5, 4]})

    # Top-Left figure
    legend_elements = [Line2D([0], [0], color='steelblue', lw=0,
                              marker='.', label=r'Catalogue'),
                       Line2D([0], [0], color=color, lw=0,
                              marker='.', label='Poisson'),
                       Line2D([0], [0], color='black', lw=1,
                              linestyle='-', label=r'Envelope $\alpha=0.05$')]
    ax = nz_analysis.cat_var.plot_n2(ax=axs[0, 0], markersize=0.02,
                                     kernel_size=ksize)
    ax = model_i.sim_var.plot_n2(ax=ax, color=color, ylims=n2_ylims,
                                 markersize=msize, kernel_size=ksize)
    ax.vlines(80, 0, 1100, color='black', linestyle='--', linewidth=1)
    ax.text(80, 300, 'c)', fontsize=14, horizontalalignment='right',
            rotation=90)
    ax.vlines(180, 0, 1100, color='black', linestyle='--', linewidth=1)
    ax.text(180, 300, 'd)', fontsize=14, horizontalalignment='right',
            rotation=90)

    ax.set_xlim([1, 300])
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    ax.set_title('a)', loc='left', fontsize=16, fontweight="bold")

    # Top-right figure
    ax = nz_analysis.cat_var.plot_logratio(ax=axs[0, 1], markersize=0.02,
                                           kernel_size=ksize)
    ax = model_i.sim_var.plot_logratio(ax=ax, color=color, ylims=ration2_ylims,
                                       markersize=msize, kernel_size=ksize)
    ax.set_xlim([1, 300])
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax.set_title('b)', loc='left', fontsize=16, fontweight="bold")
    # plt.savefig(join(figure_folder, 'nz_cat_poisson_logratio.png'), dpi=dpi)
    # plt.show()

    # Bottom figures
    index = ['c)', 'd)']
    for m, ax, idx in zip([80, 180], axs[1, :], index):

        bins = np.linspace(0, 700, 71)
        range_ = nz_analysis.cat_var.plot_histogram(m, ax=ax, bins=bins,
                                                    label=f'Catalogue')
        model_i.sim_var.plot_histogram(m, ax=ax, range_=range_,  bins=bins,
                                       color=color, label=f'Poisson')
        ax.set_xlabel(r'$N_2$', fontsize=14)
        ax.set_xlim([-10, 600])
        ax.set_ylabel(f"$P\{{ N_2 | N_1 = {m} \}}$", fontsize=14)
        ax.legend(fontsize=12)
        ax.set_title(f'{idx}', fontsize=16, loc='left', fontweight="bold")

    plt.savefig(figpath, dpi=500)
    plt.show()


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


def fig_ratevar(figpath, nsims=1000):


    os.makedirs(os.path.dirname(figpath), exist_ok=True)

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
    plt.savefig(figpath, dpi=900)
    plt.show()


if __name__ == '__main__':

    # fig_sampling(join(paths.ms1_figs['fig7'], 'cat_variability_poisson.png'))
    # rate_var_regions()
    # rate_var_etas()
    ms_figpath = join(paths.ms1_figs['fig8'], f'rate_variability.png')
    fig_ratevar(ms_figpath, nsims=ETAS_NSIMS)
