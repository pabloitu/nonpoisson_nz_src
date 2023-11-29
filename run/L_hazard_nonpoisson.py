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
import seaborn as sns
from os.path import join
def make_nonpoisson_hazard(cities=None):

    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}
    bins = 3

    catalog = catalogs.filter_cat(catalogs.get_cat_nz(), mws=(5.0, 10.0),
                                  depth=(40, -2),
                                  start_time=dt(1950, 1, 1), end_time=None,
                                  shapefile=paths.region_nz_test)

    negbinom_urz = forecastModel.load(f'npua')
    negbinom_urz_haz = hazardModel(negbinom_urz)
    negbinom_urz_haz.set_secondary_attr()
    negbinom_urz_haz.get_point_sources()
    if cities:
        negbinom_urz_haz.grid_obs = cities
    negbinom_urz_haz.write_model(imtl=imtl)

    poisson_fe = forecastModel.load(f'fe_low')
    poisson_fe_haz = hazardModel(poisson_fe)
    poisson_fe_haz.set_secondary_attr()
    poisson_fe_haz.get_point_sources()
    if cities:
        poisson_fe_haz.grid_obs = cities
    poisson_fe_haz.write_model(imtl=imtl)

    negbinom_fe = forecastModel.load(f'npfe')
    negbinom_fe_haz = hazardModel(negbinom_fe)
    negbinom_fe_haz.set_secondary_attr()
    negbinom_fe_haz.get_point_sources()
    if cities:
        negbinom_fe_haz.grid_obs = cities
    negbinom_fe_haz.write_model(imtl=imtl)


def run_models():

    models = ['npua', 'fe_low', 'npfe']

    for i in models:
        folder = paths.get_model('forecast', i)
        run_folder(folder)


def plot_hazard(locations):

    sns.set_style("darkgrid", {"ytick.left": True, 'xtick.bottom': True,
                               "axes.facecolor": ".9", "axes.edgecolor": "0",
                               'axes.linewidth': '1', 'font.family': 'Ubuntu'})

    subfolders = ['m', 'pua_3',  'fe', 'npua', 'npfe']
    names = ['Multiplicative', 'Poisson URZ', 'Poisson FE', 'Negbinom URZ','Negbinom FE']
    colors = ['blue', 'red', 'purple', 'green', 'orange']
    ls = ['-', '-', '-', '--', '--']
    imtl = {"PGA": list(np.logspace(-2, 0.2, 40))}
    models = []

    for n, f in zip(names, subfolders):
        print('Model: ', f)
        model = hazardResults(n, paths.get_model('forecast', f))
        model.parse_db(imtl)
        model.get_stats('hcurves', 'PGA')
        models.append(model)

    city_groups = [locations[:4], locations[4:]]
    for city_ns, locs in enumerate(city_groups):
        fig, axes = plt.subplots(1, 4, figsize=(12, 5), sharex=True, sharey=True)
        for i, location in enumerate(locs):
            poi = getattr(paths, location)
            for j, model in enumerate(models):
                model.plot_pointcurves('PGA', poi, ax=axes.ravel()[i], plot_args={'mean_c': colors[j],
                                                                                  'mean_s': ls[j], 'stats_lw': 2,
                                                                                  'xlims': [3e-2, 1e0],
                                                                                  'poes': j == 0}, yrs=50)
            axes.ravel()[i].set_title(location, fontsize=15)
            axes.ravel()[i].grid(visible=True, which='minor', color='white', linestyle='-', linewidth=0.5)
            axes.ravel()[i].tick_params(axis='both', labelsize=17)
        axes.ravel()[0].legend(loc=3, fontsize=14)
        plt.subplots_adjust(hspace=0.16, wspace=0.03)

        # fig.supxlabel(f'PGA $[g]$', fontsize=17)
        ylabel = 'Probability of exceedance - %i years' % 50
        axes.ravel()[0].set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        plt.savefig(join(paths.ms2_figs[f'fig{11 + city_ns}'], f'hazard_cities_{city_ns}.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(join(paths.ms2_figs[f'fig{11 + city_ns}'], f'fig{11 + city_ns}.png'),
                    dpi=1200, bbox_inches='tight')
        plt.show()
    sns.reset_defaults()

def make_vti():


    subfolders =  ['npua', 'fe_low', 'npfe', 'npfe_low']
    subfolders = ['npfe_low']
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

    make_nonpoisson_hazard()
    run_models()
    cities = ['Gisborne', 'Auckland', 'Wellington', 'Dunedin',
              'Napier', 'Tauranga', 'Queenstown', 'Invercargill'
              ]
    plot_hazard(cities)
    make_vti()

