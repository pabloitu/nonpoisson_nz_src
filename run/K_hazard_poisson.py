from nonpoisson.temporal import Model, backwardPoisson, CatalogVariability
from nonpoisson.forecast import forecastModel, get_tvzpolygon
from nonpoisson.hazard import hazardModel, hazardResults
from nonpoisson.hazard import run_folder
import numpy as np
from nonpoisson import paths
from nonpoisson import catalogs
from nonpoisson.zonation import GeodeticModel
from datetime import datetime as dt
from matplotlib import pyplot as plt
from os.path import join
import seaborn as sns
def make_poisson_hazard(cities=None):

    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}
    bins = 3

    catalog = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(5.0, 10.0),
                                  depth=(40, -2),
                                  start_time=dt(1950, 1, 1), end_time=None,
                                  shapefile=paths.region_nz_test)



    hybrid = forecastModel.load('m')
    hybrid_haz = hazardModel(hybrid)
    hybrid_haz.set_secondary_attr()
    hybrid_haz.get_point_sources()
    hybrid_haz.grid_obs = cities
    hybrid_haz.write_model(imtl=imtl)


    poisson_urz = forecastModel.load(f'pua_{bins}')
    poisson_urz_haz = hazardModel(poisson_urz)
    poisson_urz_haz.set_secondary_attr()
    poisson_urz_haz.get_point_sources()
    if cities:
        poisson_urz_haz.grid_obs = cities
    poisson_urz_haz.write_model(imtl=imtl)

    poisson_fe = forecastModel.load(f'fe')
    poisson_fe_haz = hazardModel(poisson_fe)
    poisson_fe_haz.set_secondary_attr()
    poisson_fe_haz.get_point_sources()
    if cities:
        poisson_fe_haz.grid_obs = cities
    poisson_fe_haz.write_model(imtl=imtl)


def make_poisson_sens_hazard(cities=None):

    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}
    bins = 3

    catalog = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(5.0, 10.0),
                                  depth=(40, -2),
                                  start_time=dt(1950, 1, 1), end_time=None,
                                  shapefile=paths.region_nz_test)

    sens = ['4', '5', 'ss', 'tau_max']
    for val in sens:
        poisson_urz = forecastModel.load(f'pua_{val}')
        poisson_urz_haz = hazardModel(poisson_urz)
        poisson_urz_haz.set_secondary_attr()
        poisson_urz_haz.get_point_sources()
        if cities:
            poisson_urz_haz.grid_obs = cities
        poisson_urz_haz.write_model(imtl=imtl)


def run_models():

    models = ['m', 'pua_3', 'pua_4', 'pua_5', 'pua_ss', 'pua_tau_max', 'fe']

    for i in models:
        folder = paths.get_model('forecast', i)
        run_folder(folder)


def plot_results_cities(locations):
    # folder = paths.get_oqpath(folder)
    # calc_ids = [2, 1, 3]
    # folder_names = ['hybrid_m', 'fe_m',  'pua_3',]
    sns.set_style("darkgrid", {"ytick.left": True, 'xtick.bottom': True,
                               "axes.facecolor": ".9", "axes.edgecolor": "0",
                               'axes.linewidth': '1', 'font.family': 'Ubuntu'})

    model_titles = ['Multiplicative',  'Poisson URZ', 'Poisson FE',]
    colors = ['blue', 'red', 'purple']
    ls = ['-', '-', '--']
    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}

    hybrid = hazardResults(model_titles[0], paths.get_model('forecast', 'm'))
    hybrid.parse_db(imtl)
    hybrid.get_stats('hcurves', 'PGA')

    pua = hazardResults(model_titles[1], paths.get_model('forecast', 'pua_3'))
    pua.parse_db(imtl)
    pua.get_stats('hcurves', 'PGA')

    fem = hazardResults(model_titles[2], paths.get_model('forecast', 'fe'))
    fem.parse_db(imtl)
    fem.get_stats('hcurves', 'PGA')

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True, sharey=True)
    i = 0

    for location in locations:
        poi = getattr(paths, location)

        hybrid.plot_pointcurves('PGA', poi, ax=axes.ravel()[i], plot_args={'mean_c': colors[0], 'mean_s': ls[0], 'stats_lw': 2,
                                                                           'xlims': [1e-2, 1.5],
                                                                           'ylims': [1e-3, 2],
                                                                           'poes': False}, yrs=50)
        pua.plot_pointcurves('PGA', poi, ax=axes.ravel()[i], plot_args={'mean_c': colors[1], 'mean_s': ls[1], 'stats_lw': 2,
                                                                        'xlims': [1e-2, 1.5],
                                                                        'ylims': [1e-3, 2],
                                                                        'poes': False}, yrs=50)
        fem.plot_pointcurves('PGA', poi, ax=axes.ravel()[i], plot_args={'mean_c': colors[2], 'mean_s': ls[2], 'stats_lw': 2,
                                                                        'xlims': [1e-2, 1.5],
                                                                        'ylims': [1e-3, 2],
                                                                        'poes': True}, yrs=50)

        axes.ravel()[i].grid(visible=True, which='minor', color='white', linestyle='-', linewidth=0.8)
        axes.ravel()[i].grid(axis='both', visible=True, which='major', linewidth=2)
        axes.ravel()[i].set_title(location, fontsize=18)
        axes.ravel()[i].tick_params(axis='both', labelsize=17)

        if i == 0:
            axes.ravel()[i].legend(loc=1, fontsize=16)
        i += 1
    plt.subplots_adjust(hspace=0.16, wspace=0.03)
    fig.supxlabel(f'PGA $[g]$', fontsize=24)
    ylabel = 'Probability of one or more exceedances - %i years' % 50
    fig.supylabel(ylabel, fontsize=24)
    plt.tight_layout()
    plt.savefig(join(paths.ms2_figs['fig6'], f'poisson_curves.png'), dpi=300)
    plt.savefig(join(paths.ms2_figs['fig6'], f'fig6.png'), dpi=1200)
    plt.show()
    sns.reset_defaults()

# def plot_hazard(locations):
#     subfolders = ['m', 'pua_3', 'fe']
#     names = ['Hybrid', 'Poisson URZ', 'Poisson FE']
#     colors = ['blue', 'red', 'purple']
#     ls = ['-', '-', '--']
#     imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}
#     models = []
#
#     for n, f in zip(names, subfolders):
#         print('Model: ', f)
#         model = hazardResults(n, paths.get_model('forecast', f))
#         model.parse_db(imtl)
#         model.get_stats('hcurves', 'PGA')
#         models.append(model)
#
#     city_groups = [locations[:3], locations[3:6], locations[6:9]]
#
#     for city_ns, locs in enumerate(city_groups):
#         fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
#         for i, location in enumerate(locs):
#             poi = getattr(paths, location)
#             for j, model in enumerate(models):
#                 model.plot_pointcurves('PGA', poi, ax=axes.ravel()[i], plot_args={'mean_c': colors[j],
#                                                                                   'mean_s': ls[j], 'stats_lw': 2,
#                                                                                   'poes': j == 0}, yrs=50)
#             axes.ravel()[i].set_title(location, fontsize=15)
#             axes.ravel()[i].grid(visible=True, which='minor', color='white', linestyle='-', linewidth=0.5)
#
#         axes.ravel()[0].legend(loc=3, fontsize=14)
#         plt.subplots_adjust(hspace=0.16, wspace=0.03)
#
#         # fig.supxlabel(f'PGA $[g]$', fontsize=17)
#         ylabel = 'Probability of exceedance - %i years' % 50
#         # fig.supylabel(ylabel, fontsize=14)
#         plt.tight_layout()
#         # plt.savefig(join(paths.ms_paths['K'], f'figures_hazard/m_{city_ns}'),
#         #             dpi=300, facecolor=(0, 0, 0, 0))
#         plt.show()

def plot_sensibility(locations):
    # folder = paths.get_oqpath(folder)
    # calc_ids = [2, 1, 3]
    # folder_names = ['hybrid_m', 'fe_m',  'pua_3',]
    sns.set_style("darkgrid", {"ytick.left": True, 'xtick.bottom': True,
                               "axes.facecolor": ".9", "axes.edgecolor": "0",
                               'axes.linewidth': '1', 'font.family': 'Ubuntu'})

    model_titles = ['$J_2$ - 3 bins',  '$J_2$ - 4 bins', '$J_2$ - 5 bins',
                    '$\mathrm{SS}$ - 4 bins', '$\gamma_{\mathrm{max}}$ - 4 bins']

    colors = ['darkred', 'red', 'orange', 'green', 'lightblue']
    ls = ['--', '--', '--', '-.', '-.']
    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}

    pua3 = hazardResults(model_titles[0], paths.get_model('forecast', 'pua_3'))
    pua4 = hazardResults(model_titles[1], paths.get_model('forecast', 'pua_4'))
    pua5 = hazardResults(model_titles[2], paths.get_model('forecast', 'pua_5'))
    puass = hazardResults(model_titles[3], paths.get_model('forecast', 'pua_ss'))
    puagamma = hazardResults(model_titles[4], paths.get_model('forecast', 'pua_tau_max'))
    models = [pua3, pua4, pua5, puass, puagamma]

    for i in models:
        i.parse_db(imtl)
        i.get_stats('hcurves', 'PGA')

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    i = 0

    for location in locations:
        poi = getattr(paths, location)

        for j, mod in enumerate(models):
            mod.plot_pointcurves('PGA', poi, ax=axes.ravel()[i], plot_args={'mean_c': colors[j],
                                                                            'mean_s': ls[0], 'stats_lw': 2,
                                                                            'xlims': [1e-2, 1.5],
                                                                            'ylims': [1e-3, 2],
                                                                            'poes': False}, yrs=50)
        axes.ravel()[i].grid(visible=True, which='minor', color='white', linestyle='-', linewidth=0.8)
        axes.ravel()[i].grid(axis='both', visible=True, which='major', linewidth=2)
        axes.ravel()[i].set_title(location, fontsize=21)
        axes.ravel()[i].tick_params(axis='both', labelsize=17)
        if i == 0:
            axes.ravel()[i].legend(loc=3, fontsize=12, ncols=2)
        i += 1
    plt.subplots_adjust(hspace=0.16, wspace=0.03)
    fig.supxlabel(f'PGA $[g]$', fontsize=24)
    ylabel = 'Probability of one or more exceedances - %i years' % 50
    fig.supylabel(ylabel, fontsize=24)
    plt.tight_layout()
    plt.savefig(join(paths.ms2_figs['fig7'], f'poisson_sensibility.png'), dpi=300)
    plt.savefig(join(paths.ms2_figs['fig7'], f'fig7.png'), dpi=1200)
    plt.show()
    sns.reset_defaults()



def make_vti():

    subfolders = ['m', 'pua_3', 'fe',
                  'pua_3', 'pua_4', 'pua_5', 'pua_ss', 'pua_tau_max']


    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}

    for f in subfolders:
        folder = paths.get_model('forecast', f)
        haz_model = hazardResults(f, folder)
        haz_model.parse_db(imtl)
        haz_model.get_maps_from_curves(['PGA'], [0.1])
        haz_model.get_stats('hmaps', 'PGA')
        haz_model.model2vti(f'{f}_stats', 'hmaps_stats', ['PGA'], [0.1],
                            res_method='cubic', crs_f='EPSG:2193',
                            res=(2000, 2000), crop=True)


if __name__ == '__main__':


    # make_poisson_hazard()
    # make_poisson_sens_hazard()
    # run_models()
    cities = ['Auckland', 'Tauranga', 'Gisborne',
              'Napier', 'Wellington', 'Christchurch',
              'Queenstown', 'Dunedin', 'Invercargill'
              ]
    plot_results_cities(cities)
    cities = ['Auckland', 'Dunedin', 'Wellington', 'Christchurch',
              'Gisborne', 'Queenstown', 'Tauranga', 'Napier'
              ]
    plot_sensibility(cities)
    # plot_sens(cities)
    # make_vti()

