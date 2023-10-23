import os
import logging

import etas.inversion
import numpy as np
import datetime as dt

import pandas as pd

from nonpoisson import paths, catalogs
from nonpoisson.temporal import CatalogAnalysis
from os.path import join
import fiona
import json
import matplotlib.pyplot as plt

from etas.simulation import ETASSimulation
from etas.inversion import ETASParameterCalculation
from etas import set_up_logger

set_up_logger(level=logging.DEBUG)



time_folder = paths.results_path['temporal']['dir']
etas_folder = join(time_folder, 'etas')
os.makedirs(etas_folder, exist_ok=True)

region_path = join(etas_folder, 'nz_region.npy')
new_cat_fn = join(etas_folder, 'nz_cat_etas_format.csv')
params_fn = join(etas_folder, 'params_etas_inv.json')
sim_id = 'nz'
fn_store_simulation = join(etas_folder, 'catalogs/simulated_catalog.csv')

def invert_params():

    a = catalogs.get_cat_nz()
    cat_nz = catalogs.filter_cat(a, start_time=dt.datetime(1985, 1, 1),
                                 depth=(40, -2), mws=(3.5, 10.0),
                                 shapefile=paths.region_nz_test)

    with open(new_cat_fn, 'w') as file_:
        file_.write('id,latitude,longitude,time,magnitude\n')
        for n, i in enumerate(cat_nz):
            time_str = f'{i[2]}-{i[3]:02}-{i[4]:02} {i[5]:02}:{i[6]:02}:{i[7]:02}'
            file_.write(f'{n},{i[10]},{i[9]},{time_str},{i[-3]}\n')

    region = fiona.open(paths.region_nz_test)
    nz_region = np.array(region[0]['geometry']['coordinates'][0])[:, [1,0]]
    np.save(region_path, nz_region)

    theta_0 = {
        'log10_mu': -5.8,
        'log10_k0': -2.6,
        'a': 1.8,
        'log10_c': -2.5,
        'omega': -0.02,
        'log10_tau': 3.5,
        'log10_d': -0.85,
        'gamma': 1.3,
        'rho': 0.66
    }

    inversion_meta = {
        "id": sim_id,
        "fn_catalog": new_cat_fn,
        "data_path": "",
        "auxiliary_start": dt.datetime(1985, 1, 1),
        "timewindow_start": dt.datetime(2012, 1, 1),
        "timewindow_end": dt.datetime(2022, 1, 1),
        "theta_0": theta_0,
        "mc": 3.6,
        "delta_m": 0.1,
        "coppersmith_multiplier": 100,
        "shape_coords": region_path,
    }

    calculation = ETASParameterCalculation(inversion_meta)
    calculation.prepare()
    calculation.invert()
    calculation.store_results(data_path=etas_folder)


def simulate():

    burn_date = "2000-01-01 00:00:00"
    start_date = "2020-01-01 00:00:00"
    years = 60
    n_sims = 2

    fn_parameters = join(etas_folder, f'parameters_{sim_id}.json')
    with open(fn_parameters, 'r') as file_:
        params = json.load(file_)
    params['auxiliary_start'] = burn_date
    params['timewindow_end'] = start_date
    with open(fn_parameters, 'w') as file_:
        json.dump(params, file_)

    with open(fn_parameters, 'r') as f:
        inversion_output = json.load(f)
    etas_inversion_reload = ETASParameterCalculation.load_calculation(
        inversion_output)

    # Initialize simulation
    simulation = ETASSimulation(etas_inversion_reload)
    simulation.prepare()
    simulation.catalog = None
    # Simulate and store one catalog
    simulation.simulate_to_csv(fn_store_simulation,
                               n_simulations=n_sims,
                               forecast_n_days=365*years)



if __name__ == '__main__':

    # invert_params()
    a = simulate()
    # get_rates()
    # a = get_ns_analysis()

