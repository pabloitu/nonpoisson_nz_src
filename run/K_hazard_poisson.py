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


def plot_hazard(locations):
    subfolders = ['m', 'pua_3', 'fe']
    names = ['Hybrid Multiplicative', 'Poisson URZ', 'Poisson FE']
    colors = ['blue', 'red', 'purple']
    ls = ['-', '-', '--']
    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}
    models = []

    for n, f in zip(names, subfolders):
        print('Model: ', f)
        model = hazardResults(n, paths.get_model('forecast', f))
        model.parse_db(imtl)
        model.get_stats('hcurves', 'PGA')
        models.append(model)

    city_groups = [locations[:3]]
    for city_ns, locs in enumerate(city_groups):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
        for i, location in enumerate(locs):
            poi = getattr(paths, location)
            for j, model in enumerate(models):
                model.plot_pointcurves('PGA', poi, ax=axes.ravel()[i], plot_args={'mean_c': colors[j],
                                                                                  'mean_s': ls[j], 'stats_lw': 2,
                                                                                  'poes': j == 0}, yrs=50)
            axes.ravel()[i].set_title(location, fontsize=15)
            axes.ravel()[i].grid(visible=True, which='minor', color='white', linestyle='-', linewidth=0.5)


        axes.ravel()[0].legend(loc=3, fontsize=14)
        plt.subplots_adjust(hspace=0.16, wspace=0.03)

        # fig.supxlabel(f'PGA $[g]$', fontsize=17)
        ylabel = 'Probability of exceedance - %i years' % 50
        # fig.supylabel(ylabel, fontsize=14)
        plt.tight_layout()
        # plt.savefig(join(paths.ms_paths['K'], f'figures_hazard/m_{city_ns}'),
        #             dpi=300, facecolor=(0, 0, 0, 0))
        plt.show()


def plot_sens(locations):
    subfolders = ['pua_3', 'pua_4', 'pua_5', 'pua_ss', 'pua_tau_max']
    names = ['Bins3', 'Bins4', 'Bins5', 'BinsSS', 'BinsGammamax']
    colors = ['blue', 'red', 'purple', 'green', 'yellow']
    ls = ['-', '-', '--', '--', '--']
    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}
    models = []

    for n, f in zip(names, subfolders):
        print('Model: ', f)
        model = hazardResults(n, paths.get_model('forecast', f))
        model.parse_db(imtl)
        model.get_stats('hcurves', 'PGA')
        models.append(model)

    city_groups = [locations[:3]]
    for city_ns, locs in enumerate(city_groups):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
        for i, location in enumerate(locs):
            poi = getattr(paths, location)
            for j, model in enumerate(models):
                model.plot_pointcurves('PGA', poi, ax=axes.ravel()[i], plot_args={'mean_c': colors[j],
                                                                                  'mean_s': ls[j], 'stats_lw': 2,
                                                                                  'poes': j == 0}, yrs=50)
            axes.ravel()[i].set_title(location, fontsize=15)
            axes.ravel()[i].grid(visible=True, which='minor', color='white', linestyle='-', linewidth=0.5)


        axes.ravel()[0].legend(loc=3, fontsize=14)
        plt.subplots_adjust(hspace=0.16, wspace=0.03)

        # fig.supxlabel(f'PGA $[g]$', fontsize=17)
        ylabel = 'Probability of exceedance - %i years' % 50
        # fig.supylabel(ylabel, fontsize=14)
        plt.tight_layout()
        # plt.savefig(join(paths.ms_paths['K'], f'figures_hazard/m_{city_ns}'),
        #             dpi=300, facecolor=(0, 0, 0, 0))
        plt.show()


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

    cities = ['Auckland', 'Dunedin', 'Wellington']
    make_poisson_hazard()
    make_poisson_sens_hazard()
    run_models()
    plot_hazard(cities)
    plot_sens(cities)
    make_vti()
