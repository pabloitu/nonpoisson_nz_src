from datetime import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs

from nonpoisson import paths
from nonpoisson.catalogs import get_cat_nz, get_cat_japan,\
    get_cat_it, get_cat_global, get_cat_ca
from nonpoisson.catalogs import filter_cat, cat_oq2csep


default_plot_args = {
    'projection': cartopy.crs.Mercator(),
    'basemap': None,
    'region': True,
    'legend_loc': 2}


if __name__ == '__main__':

    # Get filtered catalogs
    cat_nz = cat_oq2csep(filter_cat(get_cat_nz(), start_time=dt(1964, 1, 1),
                         mws=[3.99, None], depth=[40, -2],
                         shapefile=paths.region_nz_collection))
    # california already filtered to the csep region from pycsep query
    cat_ca = cat_oq2csep(filter_cat(get_cat_ca(), start_time=dt(1981, 1, 1),
                        mws=[3.99, None], depth=[30, -2]))
    cat_jp = cat_oq2csep(filter_cat(get_cat_japan(),
                         start_time=dt(1985, 1, 1), end_time=dt(2011, 1, 1),
                         mws=[3.99, None], depth=[30, -2],
                         shapefile=paths.region_japan))
    cat_it = cat_oq2csep(filter_cat(get_cat_it(), start_time=dt(1960, 1, 1),
                         mws=[3.99, None], depth=[30, -2],
                         shapefile=paths.region_it))
    cat_globe = cat_oq2csep(filter_cat(get_cat_global(),
                                       start_time=dt(1990, 1, 1),
                            mws=[5.99, None], depth=[70, -2]))

    # Plot catalog figures: Stored in 'results/catalogs/fig'
    cat_nz.plot(plot_args={
        **default_plot_args,
        'projection': cartopy.crs.Mercator(central_longitude=179),
        'filename': paths.get('catalogs', 'fig', 'catalog_nz', ext=False)})
    cat_jp.plot(plot_args={
        **default_plot_args,
        'filename': paths.get('catalogs', 'fig', 'cat_jp', ext=False)})
    cat_it.plot(plot_args={
        **default_plot_args,
        'filename': paths.get('catalogs', 'fig', 'cat_it', ext=False)})
    cat_ca.plot(plot_args={
        **default_plot_args,
        'filename': paths.get('catalogs', 'fig', 'cat_ca', ext=False)})
    cat_globe.plot(plot_args={
        **default_plot_args,
        'filename': paths.get('catalogs', 'fig', 'cat_globe', ext=False)})
    plt.show()

    # Print number of events
    print(f'New Zealand: {cat_nz.get_number_of_events()}\n',
          f'Japan: {cat_jp.get_number_of_events()}\n',
          f'Italy: {cat_it.get_number_of_events()}\n',
          f'California: {cat_ca.get_number_of_events()}\n',
          f'Globe: {cat_globe.get_number_of_events()}\n',)

    # Save filtered catalog: Stored in 'results/catalogs/csv'
    cat_nz.write_ascii(paths.get('catalogs', 'csv', 'cat_nz'))
    cat_jp.write_ascii(paths.get('catalogs', 'csv', 'cat_jp'))
    cat_it.write_ascii(paths.get('catalogs', 'csv', 'cat_it'))
    cat_ca.write_ascii(paths.get('catalogs', 'csv', 'cat_ca'))
    cat_globe.write_ascii(paths.get('catalogs', 'csv', 'cat_globe'))
