
from nonpoisson import paths
from nonpoisson import catalogs
from nonpoisson import temporal
from nonpoisson.temporal import CatalogAnalysis

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from datetime import datetime as dt
from os.path import join
sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})

N_MAX = 400
N_ITER = 2000
SEED = 23
RATE_VAR_PARAMS = {'n_disc': np.arange(1, N_MAX),
                   'max_iters': N_ITER}
np.random.seed(SEED)
random.seed(SEED)


def create_negbinom(fig_folder):

    hist_n = [80, 150]
    cat = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(3.99, 10),
                                 depth=(40, -2),
                                 start_time=dt(1980, 1, 1),
                                 shapefile=paths.region_nz_collection)

    Cat_analysis = CatalogAnalysis(cat, name=cat.name, params=RATE_VAR_PARAMS)
    Cat_analysis.get_ratevar()
    Cat_analysis.cat_var.get_stats()

    negbinom = temporal.NegBinom()
    negbinom.get_params(Cat_analysis)
    negbinom.save()

    model_name = 'negbinom'
    color = 'darkgreen'
    title = 'Negative Binomial'

    msize = 0.009
    ksize = 3
    ration2_ylims = [-2.1, 2.1]

    legend_elements = [Line2D([0], [0], color='steelblue',
                              lw=0, marker='.', label=r'Catalogue'),
                       Line2D([0], [0], color=color,
                              lw=0, marker='.', label=title),
                       Line2D([0], [0], color='black', lw=1,
                              linestyle='-', label=r'Envelope $\alpha=0.05$')]

    negbinom.metaparams.update({'sim_iters': 4000,
                                'sim_subiters': 4000})
    negbinom.simulate(nsims=5000)

    fig = plt.figure(figsize=(12, 6), constrained_layout=True, dpi=100)
    GridSpec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig,
                                 width_ratios=[3, 2])
    subfigure_1 = fig.add_subfigure(GridSpec[:, 0])
    ax = subfigure_1.subplots(1, 1)
    subfigure_1.suptitle('a)', ha='left', x=0.12,  fontweight="bold")
    Cat_analysis.cat_var.plot_logratio(ax=ax, markersize=0.02,
                                       kernel_size=ksize)
    negbinom.sim_var.plot_logratio(ax=ax, color=color, ylims=ration2_ylims,
                                   markersize=msize, kernel_size=ksize)
    ax.vlines(80, -2.1, 2.1, color='gray', lw=1, linestyle='--')
    ax.vlines(150, -2.1, 2.1, color='gray', lw=1, linestyle='--')
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=12, title_fontsize=12)

    subfigure_2 = fig.add_subfigure(GridSpec[:, 1])
    axs = subfigure_2.subplots(2, 1)
    subfigure_2.suptitle('b)', ha='left', x=0.17, fontweight="bold")

    for m, ax in zip(hist_n, axs):
        plt.figure(figsize=(6, 4))
        range_ = Cat_analysis.cat_var.plot_histogram(m, ax=ax,
                                                     label=f'Catalogue')
        negbinom.sim_var.plot_histogram(m, ax=ax, range_=range_, color=color,
                                        label=f'{title}')
        ax.set_xlabel(r'$N_2$', fontsize=14)
        ax.set_ylabel(f"$P\{{ N_2 | N_1 = {m} \}}$", fontsize=14)

        ax.legend(title=f'$N_1={m}$', fontsize=12, title_fontsize=12)
        ax.set_xlim([-10, 800])

    fname = join(fig_folder, f'model_{model_name}.jpg')
    fig.savefig(fname, dpi=800)
    plt.show()


def create_lognorm(fig_folder):

    hist_n = [80, 150]
    cat = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(3.99, 10),
                              depth=(40, -2),
                              start_time=dt(1980, 1, 1),
                              shapefile=paths.region_nz_collection)

    Cat_analysis = CatalogAnalysis(cat, name=cat.name, params=RATE_VAR_PARAMS)
    Cat_analysis.get_ratevar()
    Cat_analysis.cat_var.get_stats()

    lognorm = temporal.MixedPoisson(metaparams={'distribution': 'lognorm'})
    lognorm.get_params(Cat_analysis)
    lognorm.save()

    model_name = 'lognorm'
    color = 'purple'
    title = 'Lognormal-Poisson'

    msize = 0.009
    ksize = 3
    ration2_ylims = [-2.1, 2.1]

    legend_elements = [Line2D([0], [0], color='steelblue',
                              lw=0, marker='.', label=r'Catalogue'),
                       Line2D([0], [0], color=color,
                              lw=0, marker='.', label=title),
                       Line2D([0], [0], color='black', lw=1,
                              linestyle='-', label=r'Envelope $\alpha=0.05$')]

    lognorm.metaparams.update({'sim_iters': 2000,
                                'sim_subiters': 2000})
    lognorm.simulate(nsims=5000)

    fig = plt.figure(figsize=(12, 6), constrained_layout=True, dpi=100)
    GridSpec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig,
                                 width_ratios=[3, 2])
    subfigure_1 = fig.add_subfigure(GridSpec[:, 0])
    ax = subfigure_1.subplots(1, 1)
    subfigure_1.suptitle('a)', ha='left', x=0.12,  fontweight="bold")
    Cat_analysis.cat_var.plot_logratio(ax=ax, markersize=0.02, kernel_size=ksize)
    lognorm.sim_var.plot_logratio(ax=ax, color=color, ylims=ration2_ylims,
                                   markersize=msize, kernel_size=ksize)
    ax.vlines(80, -2.1, 2.1, color='gray', lw=1, linestyle='--')
    ax.vlines(150, -2.1, 2.1, color='gray', lw=1, linestyle='--')
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=12, title_fontsize=12)

    subfigure_2 = fig.add_subfigure(GridSpec[:, 1])
    axs = subfigure_2.subplots(2, 1)
    subfigure_2.suptitle('b)', ha='left', x=0.17, fontweight="bold")

    for m, ax in zip(hist_n, axs):
        plt.figure(figsize=(6, 4))
        range_ = Cat_analysis.cat_var.plot_histogram(m, ax=ax, label=f'Catalogue')
        lognorm.sim_var.plot_histogram(m, ax=ax, range_=range_, color=color,
                                        label=f'{title}')
        ax.set_xlabel(r'$N_2$', fontsize=14)
        ax.set_ylabel(f"$P\{{ N_2 | N_1 = {m} \}}$", fontsize=14)

        ax.legend(title=f'$N_1={m}$', fontsize=12, title_fontsize=12)
        ax.set_xlim([-10, 800])

    fname = join(fig_folder, f'model_{model_name}.jpg')
    fig.savefig(fname, dpi=800)
    plt.show()


if __name__ == '__main__':
    folder = paths.ms1_figs['fig9']
    create_negbinom(folder)
    folder = paths.ms1_figs['fig10']
    create_lognorm(folder)
