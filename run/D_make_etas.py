import os
from os.path import join

import datetime
import logging
import fiona
import json
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from etas import set_up_logger
from etas.inversion import round_half_up, ETASParameterCalculation
from etas.inversion import read_shape_coords
from etas.simulation import generate_catalog

from nonpoisson import paths, catalogs

set_up_logger(level=logging.INFO)

# ETAS Paths
temporal_folder = paths.results_path['temporal']['dir']
etas_folder = join(temporal_folder, 'etas/')
os.makedirs(etas_folder, exist_ok=True)
region_path = join(etas_folder, 'nz_region.npy')
new_cat_fn = join(etas_folder, 'nz_cat_etas_format.csv')
params_fn = join(etas_folder, 'parameters_nz.json')
sim_id = 'nz'
fn_store = join(etas_folder, 'simulated_catalog.csv')


def invert_params():
    a = catalogs.get_cat_nz()
    cat_nz = catalogs.filter_cat(a, start_time=dt.datetime(1985, 1, 1),
                                 depth=(40, -2), mws=(3.5, 10.0),
                                 shapefile=paths.region_nz_test)

    with open(new_cat_fn, 'w') as file_:
        file_.write('id,latitude,longitude,time,magnitude\n')
        for n, i in enumerate(cat_nz):
            time_str = f'{i[2]}-{i[3]:02}-{i[4]:02}' \
                       f' {i[5]:02}:{i[6]:02}:{i[7]:02}'
            file_.write(f'{n},{i[10]},{i[9]},{time_str},{i[-3]}\n')

    region = fiona.open(paths.region_nz_test)
    nz_region = np.array(region[0]['geometry']['coordinates'][0])[:, [1, 0]]
    np.save(region_path, nz_region)

    theta_0 = {
        "log10_mu": -7,
        "log10_k0": -1.5,
        "a": 2,
        "log10_c": -2.,
        "omega": 0.,
        "log10_tau": 3.5,
        "log10_d": 0.5,
        "gamma": 1.,
        "rho": 0.8
    }

    inversion_meta = {
        "id": sim_id,
        "fn_catalog": new_cat_fn,
        "data_path": "",
        "auxiliary_start": dt.datetime(1985, 1, 1),
        "timewindow_start": dt.datetime(1991, 1, 1),
        "timewindow_end": dt.datetime(2020, 1, 1),
        "theta_0": theta_0,
        "mc": 3.6,
        'm_ref': 3.6,
        "delta_m": 0.1,
        "coppersmith_multiplier": 100,
        "shape_coords": region_path,
    }

    calculation = ETASParameterCalculation(inversion_meta)
    calculation.prepare()
    calculation.invert()
    calculation.store_results(data_path=etas_folder)


def simulate():
    np.random.seed(21)
    burn_date = datetime.datetime(1920, 1, 1)
    start_date = datetime.datetime(1980, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)
    min_magnitude = 4
    n_sims = 1000

    fn_parameters = params_fn
    with open(fn_parameters, 'r') as file_:
        simulation_config = json.load(file_)

    polygon = Polygon(read_shape_coords(simulation_config["shape_coords"]))
    bg_catalog = pd.read_csv(simulation_config["fn_ip"], index_col=0,
                             parse_dates=['time'],
                             dtype={'url': str, 'alert': str})
    bg_catalog = bg_catalog.query(
        f"magnitude>={min_magnitude}")
    bg_catalog = gpd.GeoDataFrame(bg_catalog,
                                  geometry=gpd.points_from_xy(
                                      bg_catalog.latitude,
                                      bg_catalog.longitude))
    bg_catalog = bg_catalog[bg_catalog.intersects(polygon)]

    for i in range(n_sims):
        synthetic = generate_catalog(
            polygon=polygon,
            timewindow_start=burn_date,
            timewindow_end=end_date,
            parameters=simulation_config["final_parameters"],
            mc=simulation_config["mc"],
            beta_main=simulation_config["beta"],
            delta_m=simulation_config["delta_m"],
            background_lats=bg_catalog['latitude'],
            background_lons=bg_catalog['longitude'],
            background_probs=bg_catalog['P_background'] * (
                    bg_catalog['zeta_plus_1'] /
                    bg_catalog['zeta_plus_1'].max()),
            gaussian_scale=0.1
        )
        synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
        synthetic.index.name = 'id'
        synthetic['catalog_id'] = i
        print("store catalog..")
        if i == 0:
            synthetic[["latitude", "longitude", "time", "magnitude",
                       "catalog_id"]].query(
                "time>=@start_date & magnitude>=@min_magnitude").to_csv(
                fn_store, header=True)
        else:
            synthetic[["latitude", "longitude", "time", "magnitude",
                       "catalog_id"]].query(
                "time>=@start_date & magnitude>=@min_magnitude").to_csv(
                fn_store, mode='a', header=False)
        print("\nDONE!")


if __name__ == '__main__':

    invert_params()
    simulate()
