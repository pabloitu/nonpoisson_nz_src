import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn
from datetime import datetime as dt
from nonpoisson import catalogs, paths
from matplotlib.lines import Line2D

seaborn.set_style("darkgrid",
                  {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})


def get_rate_time(cat, hmin=0.2, hmax=1, ndisc=30, toplines=np.array([2, 4])):

    origin_times = cat.get_decimal_time()
    cat_timelength = origin_times.max() - origin_times.min()

    window_lengths = np.linspace(hmin, hmax, ndisc)
    window_lengths = np.append(window_lengths, toplines)

    Tw = []
    Count = []
    Rates = []

    for t in window_lengths:
        edges = np.append(origin_times.min() - 0.001,
                          np.flip(origin_times.max() + 0.001
                                  - np.arange(0, cat_timelength, t)))
        windows = [(i, j) for i, j in zip(edges[:-1], edges[1:])]
        disc = np.digitize(origin_times, edges)
        bin_id, count = np.unique(disc, return_counts=True)

        n_events = np.zeros(len(windows))
        n_events[bin_id - 1] = count
        n_events[n_events == 0] = np.nan

        rates = np.array([i/(j[1] - j[0]) for i, j in zip(n_events, windows)])
        rates[np.isnan(rates)] = 0

        Tw.append(windows)
        Count.append(n_events)
        Rates.append(rates)

    return Tw, Count, Rates, window_lengths


def plot_rate_series(cat, ax, ylims=None, title=None, **kwargs):

    timewindows, counts, rates, dh = get_rate_time(cat, **kwargs)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(timewindows)))

    max_n = np.max(np.hstack(rates))*1.05
    for n, (tw, rate) in enumerate(zip(timewindows, rates)):
        starts = [i[0] for i in tw]
        ends = [i[1] for i in tw]
        t = list(itertools.chain.from_iterable(zip(starts, ends)))
        r = list(itertools.chain.from_iterable(zip(rate, rate)))
        if n == len(rates) - 2:
            ax.plot(t, r, color='black', linestyle='--')
        elif n == len(rates) - 1:
            ax.plot(t, r, color='black', linestyle=':')
        else:
            ax.plot(t, r, color=colors[n])
        ax.set_xlim([np.min(t), np.max(t)])

        if ylims:
            ax.set_ylim(ylims)
        else:
            ax.set_ylim([-10, max_n])
    if title:
        ax.set_title(title, loc='left', fontsize=14)


def plot_rateseries_catalogs():

    ratecalc_args = {'hmin': 0.2, 'hmax': 1, 'ndisc': 30,
                     'toplines': np.array([2, 4])}

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    cat_globe = catalogs.filter_cat(catalogs.get_cat_global(),
                                    start_time=dt(1989, 1, 1),
                                    end_time=dt(2020, 1, 1),
                                    mws=(6.0, 11),
                                    depth=(70, -2))
    plot_rate_series(cat_globe, ax[0][0], title=r'Global $M\geq6.0$',
                     **ratecalc_args)

    cat_jp = catalogs.filter_cat(catalogs.get_cat_japan(),
                                 start_time=dt(1984, 1, 1),
                                 end_time=dt(2011, 1, 1),
                                 mws=(5.0, 11),
                                 depth=(30, -2))
    plot_rate_series(cat_jp, ax[0][1], ylims=[-10, 350],
                     title=r'Japan $M\geq5.0$', **ratecalc_args)

    cat_ca = catalogs.filter_cat(catalogs.get_cat_ca(query=False),
                                 start_time=dt(1982, 1, 1),
                                 mws=(5.0, 11),
                                 depth=(30, -2))
    plot_rate_series(cat_ca, ax[1][0], ylims=[-2.5, 70],
                     title=r'California $M\geq5.0$', **ratecalc_args)

    cat_nz = catalogs.filter_cat(catalogs.get_cat_nz(),
                                 start_time=dt(1979, 1, 1),
                                 mws=(5.0, 11),
                                 depth=(40, -2))
    plot_rate_series(cat_nz, ax[1][1], ylims=[-5, 150],
                     title=r'New Zealand $M\geq5.0$', **ratecalc_args)

    fig.supylabel(r'Mean rate   $\mu\left[\frac{\mathrm{EQ}}'
                  r'{\mathrm{yr}}\right]$', fontsize=14)
    fig.supxlabel('Year', fontsize=14)
    fig.tight_layout()

    sm = plt.cm.ScalarMappable(cmap='coolwarm')
    sm.set_clim(ratecalc_args['hmin'], ratecalc_args['hmax'])
    legend_elements = [Line2D([0], [0], color='black', lw=1, linestyle='--',
                              label=f"{ratecalc_args['toplines'][1]:.1f}"),
                       Line2D([0], [0], color='black', lw=1, linestyle=':',
                              label=f"{ratecalc_args['toplines'][0]:.1f}")]
    fig.legend(handles=legend_elements, title=r'$\Delta h$ [yr]',
               title_fontsize=18, bbox_to_anchor=(0.905, 0.925), framealpha=0.,
               labelspacing=1.2, fontsize=12)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
    cbar.ax.tick_params(labelsize=12)

    path_fig = os.path.join(paths.ms1_figs['fig7'], 'catalog_rates.png')
    plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    plot_rateseries_catalogs()
