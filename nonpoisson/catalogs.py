
import datetime
from datetime import datetime as dt

import fiona
import pandas
import numpy as np
from shapely.geometry import Polygon as Polygon_Shapely
from nonpoisson import paths

import csep
from csep.core import regions
from csep.utils import time_utils

from openquake.hazardlib.geo import Polygon as Polygon_OQ
from openquake.hazardlib.geo import Point
from openquake.hmtk.seismicity import catalogue, selector
from openquake.hazardlib.geo.utils import OrthographicProjection


def cat_oq2csep(cat_oq, region=None):
    """
    Converts Openquake catalogs into pyCSEP format
    """
    times = cat_oq.get_decimal_time()
    year = times.astype(int)
    rem = times - year
    base = [dt(y, 1, 1) for y in year]
    datetimes = [b + datetime.timedelta(seconds=(b.replace(year=b.year + 1) - b).total_seconds() * r) for b, r in
              zip(base, rem)]

    out = []
    for n, time_i in enumerate(datetimes):
        event_tuple = (n,
                       csep.utils.time_utils.datetime_to_utc_epoch(time_i),
                       cat_oq['latitude'][n],
                       cat_oq['longitude'][n],
                       cat_oq['depth'][n],
                       np.round(cat_oq['magnitude'][n], 1)
                       )
        out.append(event_tuple)

    cat = csep.catalogs.CSEPCatalog(data=out, region=region, name=cat_oq.name)

    return cat


def get_cat_nz(name='nz'):

    raw = pandas.read_csv(paths.cat_nz)
    lon = raw['lon'].to_numpy()
    lat = raw['lat'].to_numpy()
    depth = raw['depth'].to_numpy()
    mag = raw['mw'].to_numpy().astype(float)

    id = np.zeros(len(mag))
    year = raw['y'].to_numpy()
    month = raw['month'].to_numpy()
    day = raw['d'].to_numpy()
    hour = raw['h'].to_numpy()
    min = raw['minute'].to_numpy()
    sec = raw['s'].to_numpy()
    cat = catalogue.Catalogue()
    cat.load_from_array(['eventID', 'year', 'month', 'day', 'hour', 'minute',
                         'second', 'longitude', 'latitude', 'depth', 'magnitude'],
                        np.vstack((id, year, month, day, hour, min, sec, lon, lat, depth, mag)).T)
    cat.update_end_year()
    cat.update_start_year()
    cat.sort_catalogue_chronologically()
    cat.name = name
    return cat


def get_cat_nz_dc(name=None):

    raw = np.genfromtxt(paths.cat_nz_dc, skip_header=1, delimiter=',')
    lon = raw[:, 6]
    lat = raw[:, 7]
    depth = raw[:, 8]
    mag = raw[:, 9]

    id = np.zeros(len(mag))
    year = raw[:, 0]
    month = raw[:, 1]
    day = raw[:, 2]
    hour = raw[:, 3]
    min = raw[:, 4]
    sec = raw[:, 5]
    cat = catalogue.Catalogue()
    cat.load_from_array(['eventID', 'year', 'month', 'day', 'hour', 'minute',
                         'second', 'longitude', 'latitude', 'depth', 'magnitude'],
                        np.vstack((id, year, month, day, hour, min, sec, lon, lat, depth, mag)).T)
    cat.update_end_year()
    cat.update_start_year()
    cat.sort_catalogue_chronologically()
    if name is None:
        cat.name = 'nz_dc'
    else:
        cat.name = name
    return cat

def get_cat_ca(query=False, name='california'):
    # Magnitude bins properties

    def parse_csv():
        raw = pandas.read_csv(paths.cat_ca)
        lon = raw['longitude'].to_numpy()
        lat = raw['latitude'].to_numpy()
        depth = raw['depth'].to_numpy()
        mag = raw['magnitude'].to_numpy().astype(float)
        time_epoch = raw['origin_time'].to_numpy().astype(int)
        id = np.zeros(len(mag))

        datetime = [time_utils.epoch_time_to_utc_datetime(i) for i in time_epoch]
        year = [i.year for i in datetime]
        month = [i.month for i in datetime]
        day = [i.day for i in datetime]
        hour = [i.hour for i in datetime]
        min = [i.minute for i in datetime]
        sec = [i.second + i.microsecond*1e-6 for i in datetime]
        cat = catalogue.Catalogue()
        cat.load_from_array(['eventID', 'year', 'month', 'day', 'hour', 'minute',
                             'second', 'longitude', 'latitude', 'depth', 'magnitude'],
                            np.vstack((id, year, month, day, hour, min, sec, lon, lat, depth, mag)).T)
        cat.update_end_year()
        cat.update_start_year()
        cat.sort_catalogue_chronologically()
        cat.name = name
        return cat

    if query:
        min_mw = 4.
        max_mw = 9.0
        dmw = 0.1
        start_time = time_utils.strptime_to_utc_datetime("1960-01-01 11:57:35.0")
        end_time = time_utils.strptime_to_utc_datetime("2021-01-01 11:57:35.0")
        # Create space and magnitude regions. The forecast is already filtered in space and magnitude
        magnitudes = regions.magnitude_bins(min_mw, max_mw, dmw)
        region = regions.california_relm_region()

        # Bind region information to the forecast (this will be used for binning of the catalogs)
        space_magnitude_region = regions.create_space_magnitude_region(region, magnitudes)
        comcat_catalog = csep.query_comcat(start_time, end_time, min_magnitude=min_mw)
        comcat_catalog = comcat_catalog.filter_spatial(region)

        pandas_cat = comcat_catalog.to_dataframe()
        pandas_cat.to_csv(paths.cat_ca)
        return comcat_catalog

    catalog = parse_csv()

    return catalog


def get_cat_japan(name='japan'):
    # 0 lon, 1 lat, 2 year, 3 month, 4 day, 5 mw, 6 depth, 7 hour, 8 min, 9 sec
    data = np.genfromtxt(paths.cat_japan, delimiter=',')
    id = np.arange(data.shape[0])
    lon = data[:, 0]
    lat = data[:, 1]
    year = np.floor(data[:, 2])
    month = data[:, 3]
    day = data[:, 4]
    mag = data[:, 5]

    depth = data[:, 6]
    hour = data[:, 7]
    min = data[:, 8]
    sec = data[:, 9]
    cat = catalogue.Catalogue()
    cat.load_from_array(
        ['eventID', 'year', 'month', 'day', 'hour', 'minute', 'second', 'longitude', 'latitude', 'depth', 'magnitude'],
        np.vstack((id, year, month, day, hour, min, sec, lon, lat, depth, mag)).T)
    cat.update_end_year()
    cat.update_start_year()
    cat.name = name
    cat.sort_catalogue_chronologically()
    return cat


def get_cat_it(name='italy'):

    raw = pandas.read_csv(paths.cat_it)
    lon = raw['Lon'].to_numpy()
    lat = raw['Lat'].to_numpy()
    depth = raw['Depth'].to_numpy()
    mag = raw['Mw'].to_numpy().astype(float)

    id = np.arange(raw.shape[0])
    year = raw['Year'].to_numpy()
    month = raw['Mo'].to_numpy()
    day = raw['Da'].to_numpy()
    hour = raw['Ho'].to_numpy()
    min = raw['Mi'].to_numpy()
    sec = raw['Se'].to_numpy()
    cat = catalogue.Catalogue()
    cat.load_from_array(['eventID', 'year', 'month', 'day', 'hour', 'minute',
                         'second', 'longitude', 'latitude', 'depth', 'magnitude'],
                        np.vstack((id, year, month, day, hour, min, sec, lon, lat, depth, mag)).T)
    cat.update_end_year()
    cat.update_start_year()
    cat.sort_catalogue_chronologically()
    cat.name = name
    return cat


def get_cat_global(name='global'):
    # 0 lon, 1 lat, 2 year, 3 month, 4 day, 5 mw, 6 depth, 7 hour, 8 min, 9 sec
    data = pandas.read_csv(paths.cat_global)
    events_id = np.arange(data.shape[0])
    datetimes = [dt.strptime(i, '%Y/%m/%d %H:%M:%S.%f') for i in data['DateTime']]
    lon = data['Longitude'].to_numpy()
    lat = data['Latitude'].to_numpy()
    depth = data['Depth'].to_numpy()
    mag = data['Magnitude'].to_numpy().astype(float)
    year = [i.year for i in datetimes]

    month = [i.month for i in datetimes]
    day = [i.day for i in datetimes]
    hour = [i.hour for i in datetimes]
    min = [i.minute for i in datetimes]
    sec = [i.second for i in datetimes]

    cat = catalogue.Catalogue()
    cat.load_from_array(
        ['eventID', 'year', 'month', 'day', 'hour', 'minute', 'second', 'longitude', 'latitude', 'depth', 'magnitude'],
        np.vstack((events_id, year, month, day, hour, min, sec, lon, lat, depth, mag)).T)
    cat.update_end_year()
    cat.update_start_year()
    cat.name = name
    cat.sort_catalogue_chronologically()
    return cat


def get_cat_etas(path=None, name=None):

    raw = pandas.read_csv(path)
    lon = raw['latitude'].to_numpy()
    lat = raw['longitude'].to_numpy()
    depth = np.ones(lat.shape)
    mag = raw['magnitude'].to_numpy().astype(float)

    id = raw['catalog_id'].to_numpy()
    date = [datetime.datetime.strptime(i[:-3],'%Y-%m-%d %H:%M:%S.%f') for i in raw['time']]
    year = [i.year for i in date]
    month = [i.month for i in date]
    day = [i.day for i in date]
    # time = [datetime.time.fromisoformat(i) for i in raw['time']]
    hour = [i.hour for i in date]
    minute = [i.minute for i in date]
    second = [i.second for i in date]

    data = np.vstack((id, year, month, day, hour, minute, second, lon, lat, depth, mag)).T

    catalogs = []
    for i in np.unique(data[:, 0]).astype(int):

        cat = catalogue.Catalogue()
        cat.load_from_array(['eventID', 'year', 'month', 'day', 'hour', 'minute',
                             'second', 'longitude', 'latitude', 'depth', 'magnitude'],
                            data[data[:, 0] == i])
        cat.update_end_year()
        cat.update_start_year()
        cat.sort_catalogue_chronologically()
        if name is None:
            cat.name = f'etas_{i}'
        else:
            cat.name = f'{name}_{i}'
        catalogs.append(cat)

    return catalogs


def filter_cat(cat, mws=(3.99, 10.0), depth=(40, -2),
               start_time=dt(1964, 1, 1),
               end_time=None, shapefile=None, circle=False):

    filter = selector.CatalogueSelector(cat)
    new_cat = filter.within_depth_range(*depth)
    filter = selector.CatalogueSelector(new_cat)
    new_cat = filter.within_magnitude_range(*mws)
    filter = selector.CatalogueSelector(new_cat)
    new_cat = filter.within_time_period(start_time=start_time, end_time=end_time)
    if shapefile:
        polygon = fiona.open(shapefile)
        shell_sphe = polygon[0]['geometry']['coordinates'][0]
        holes_sphe = polygon[0]['geometry']['coordinates'][1:]
        proj = OrthographicProjection.from_lons_lats(np.array([i[0] for i in shell_sphe]),
                                                     np.array([i[1] for i in shell_sphe]))
        shapely_poly = Polygon_Shapely(shell=np.array(proj(*np.array(shell_sphe).T)).T,
                       holes=[np.array(proj(*np.array(i).T)).T for i in holes_sphe])
        oq_poly = Polygon_OQ._from_2d(shapely_poly, proj)
        filter = selector.CatalogueSelector(new_cat)
        new_cat = filter.within_polygon(oq_poly)
    if circle:

        filter = selector.CatalogueSelector(new_cat)
        point = Point(circle[0][0], circle[0][1], 0)
        new_cat = filter.circular_distance_from_point(point, circle[1], distance_type='epicentral')

    new_cat.update_end_year()
    new_cat.update_start_year()
    new_cat.sort_catalogue_chronologically()

    return new_cat

