
from nonpoisson import paths
from nonpoisson import catalogs
from nonpoisson import temporal
from nonpoisson.temporal import CatalogAnalysis


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from datetime import datetime as dt
from os.path import join
from statsmodels.tsa.stattools import adfuller, kpss
sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})

time_folder = paths.results_path['temporal']['dir']
etas_folder = join(time_folder, 'etas')
figure_folder = join(time_folder, 'figures')
fn_store_simulation = join(etas_folder, 'catalogs/simulated_catalog.csv')


def fig_regions_stats(figure_folder, rerun=True, savefig=True):
    n_max = 300
    n_iterations = 2000
    params = {'n_disc': np.arange(1, n_max),
              'max_iters': n_iterations}
    if rerun:
        cat_nz = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(3.95, 10),
                                     depth=(40, -2),
                                     start_time=dt(1980, 1, 1),
                                     shapefile=paths.region_nz_collection)
        nz = CatalogAnalysis(cat_nz, name='new_zealand',
                             params=params)
        nz.get_ratevar()
        nz.cat_var.get_stats()
        nz.cat_var.purge()
        nz.save()

        cat_japan = catalogs.filter_cat(catalogs.get_cat_japan(),
                                        start_time=dt(1985, 1, 1),
                                        shapefile=paths.region_japan,
                                        mws=(3.99, 10),
                                        depth=(30, -2))
        japan = CatalogAnalysis(cat_japan,
                                name='japan',
                                params=params)
        japan.get_ratevar()
        japan.cat_var.get_stats()
        japan.cat_var.purge()
        japan.save()

        cat_ca = catalogs.filter_cat(catalogs.get_cat_ca(query=False),
                                     start_time=dt(1981, 1, 1),
                                     mws=(3.99, 10),
                                     depth=(30, -2),
                                     )
        california = CatalogAnalysis(cat_ca,
                                     name='california',
                                     params=params)
        california.get_ratevar()
        california.cat_var.get_stats()
        california.cat_var.purge()
        california.save()

        cat_it = catalogs.filter_cat(catalogs.get_cat_it(),
                                     start_time=dt(1960, 1, 1),
                                     shapefile=paths.region_it,
                                     mws=(3.99, 10),
                                     depth=(30, -2))

        italy = CatalogAnalysis(cat_it, name='italy',
                                params=params)
        italy.get_ratevar()
        italy.cat_var.get_stats()
        italy.cat_var.purge()
        italy.save()

        cat_globe = catalogs.filter_cat(catalogs.get_cat_global(),
                                        start_time=dt(1990, 1, 1),
                                        mws=(5.99, 10),
                                        depth=(70, -2))
        globe = CatalogAnalysis(cat_globe, name='globe',
                                params=params)
        globe.get_ratevar()
        globe.cat_var.get_stats()
        globe.cat_var.purge()
        globe.save()

    else:
        nz = CatalogAnalysis.load(paths.get('temporal', 'serial', 'new_zealand'))
        japan = CatalogAnalysis.load(paths.get('temporal', 'serial', 'japan'))
        california = CatalogAnalysis.load(paths.get('temporal', 'serial', 'california'))
        italy = CatalogAnalysis.load(paths.get('temporal', 'serial', 'italy'))
        globe = CatalogAnalysis.load(paths.get('temporal', 'serial', 'globe'))

    print(f'N_events \n\t New Zealand: {nz.catalog.get_number_events()} - Japan: {japan.catalog.get_number_events()}'
          f'- California: {california.catalog.get_number_events()} - GLobe: {globe.catalog.get_number_events()}')

    fig = plt.figure(figsize=(6,6))
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

def fig_etas_stats(figure_folder, nsims=20, savefig=True):


    n_max = 500

    nz = CatalogAnalysis.load(paths.get_path('temporal', 'serial', 'new_zealand'))
    etas = [CatalogAnalysis.load(paths.get_path('temporal', 'serial', f'etas_{i}'))
            for i in range(nsims)]

    print('etas loaded')
    for i in etas:
        i.cat_var.get_stats()
        i.cat_var.purge()

    print('stats calculated')
    etas_means = [np.mean([i.cat_var.stats['mean'][j] for i in etas])
                   for j in range(n_max)]
    etas_5 = [np.quantile([i.cat_var.stats['mean'][j] for i in etas], 0.025)
              for j in range(n_max)]
    etas_95 = [np.quantile([i.cat_var.stats['mean'][j] for i in etas], 0.975)
              for j in range(n_max)]

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


def ns_test(cat):

    time_y = cat.get_decimal_time()
    intervals_disc = np.linspace(0.01, 5, 50)

    N = []
    Count = []
    DT = []
    extent = time_y.max() - time_y.min()

    for t in intervals_disc:
        t_intervals = np.flip(time_y.max()+0.001 - np.arange(0, extent, t))

        disc = np.digitize(time_y, t_intervals)
        n, count = np.unique(disc, return_counts=True)

        windows = np.zeros(len(t_intervals) + 1)
        windows[n] = count
        windows[windows == 0] = np.nan

        N.append(n)
        new_intervals = np.insert(t_intervals, 0, time_y.min())
        new_intervals = np.append(new_intervals, time_y.max())
        DT.append(new_intervals)
        Count.append(windows)


    # plt.figure(figsize=(6, 4))
    H=[]
    th = []
    adf_array = []
    kpss_array = []
    dt_out =[]
    rates_out = []

    for i, j in enumerate(Count):
        dt_i = np.diff(DT[i])
        rates = j[1:-1].astype(float)/dt_i[1:-1].astype(float)
        rates[np.isnan(rates)] = 0
        adf_array.append(adfuller(rates,regression='ct'))
        kpss_array.append(kpss(rates, regression='ct'))
        dt_out.append(DT[i][2:-1])
        rates_out.append(rates)


    plt.plot(intervals_disc, [i[1] for i in adf_array], 'o', color='steelblue')
    plt.plot(intervals_disc, [i[1] for i in kpss_array], 'o', color='red')
    plt.hlines(0.05, 0, 5, color='black', linestyle='--')
    plt.show()

    return kpss_array, adf_array

if __name__ == '__main__':
    fig_folder = '.'

    a = fig_regions_stats(fig_folder, rerun=True, savefig=False)
    # fig_etas_stats(fig_folder, nsims=100, savefig=True)

    #
    # cat_nz = catalogs.filter_cat(catalogs.get_cat_nz(),
    #                              start_time=dt(1960, 1, 1),
    #                              end_time=dt(2016, 1, 1),
    #                              mws=(4.5, 10.0),
    #                              depth=(40, -2),
    #                              shapefile=paths.region_nz_test)

    # ns_test(cat_nz)

    # sim = 0
    # etas = catalogs.get_cat_etas(fn_store_simulation)

    #
    # cat_etas = catalogs.filter_cat(etas[0],
    #                                start_time=dt(2020, 1, 1),
    #                                end_time=dt(2080, 1, 1),
    #                               mws=(4.5, 10.0),
    #                               shapefile=paths.region_nz_test)
    #
    # a,b  = ns_test(etas[2])


