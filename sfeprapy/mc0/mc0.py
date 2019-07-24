# -*- coding: utf-8 -*-
import json
import multiprocessing as mp
import os
import time
import warnings
from tkinter import filedialog, Tk, StringVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from sfeprapy.func.fire_iso834 import fire as _fire_standard
from sfeprapy.mc0.mc0_func_gen import mc_inputs_generator
from sfeprapy.mc0.mc0_func_main import calc_time_equivalence_worker, calc_time_equivalence, y_results_summary


def select_path_input_csv():
    """get a list of dict()s representing different scenarios"""

    root = Tk()
    root.withdraw()
    folder_path = StringVar()

    path_input_file_csv = filedialog.askopenfile(
        title='Select Input File',
        parent=root,
        filetypes=[('SPREADSHEET', ['.csv', '.xlsx'])],
        mode='r'
    )

    folder_path.set(path_input_file_csv)

    root.update()

    try:
        path_input_file_csv = os.path.realpath(path_input_file_csv.name)
        return path_input_file_csv
    except AttributeError:
        raise FileNotFoundError('file not found.')


def mc_inputs_dict_to_df(dict_in: dict):
    """DESCRIPTION
    Generate sampled values for stochastic parameters based upon their distributions, this process includes static
    parameters where they are duplicated for a pre-defined number (i.e. number of simulations).

    PARAMETERS
    :param dict_in: dict, with keys defining parameter name and value defining their value.
    :return df_mc: DataFrame, sampled set as rows and parameters as columns.
    """

    df_mc = mc_inputs_generator(index=True, **dict_in)

    n_simulations = len(df_mc.index)
    beam_c = 0
    iso834_time = np.arange(0, 6 * 60 * 60, 30)
    iso834_temperature = _fire_standard(iso834_time, 273.15 + 20)

    df_mc['case_name'] = [dict_in['case_name']] * n_simulations
    df_mc['time_step'] = [dict_in['time_step']] * n_simulations
    df_mc['time_limiting'] = [dict_in['fire_tlim']] * n_simulations
    df_mc['window_height'] = [dict_in['room_window_height']] * n_simulations
    df_mc['window_width'] = [dict_in['room_window_width']] * n_simulations
    df_mc['room_breadth'] = [dict_in['room_breadth']] * n_simulations
    df_mc['room_depth'] = [dict_in['room_depth']] * n_simulations
    df_mc['room_height'] = [dict_in['room_height']] * n_simulations
    df_mc['room_wall_thermal_inertia'] = [dict_in['room_wall_thermal_inertia']] * n_simulations
    df_mc['fire_duration'] = [dict_in['time_duration']] * n_simulations
    df_mc['fire_mode'] = [dict_in['fire_mode']] * n_simulations
    df_mc['fire_t_alpha'] = [dict_in['fire_t_alpha']] * n_simulations
    df_mc['fire_gamma_fi_q'] = [dict_in['fire_gamma_fi_q']] * n_simulations
    df_mc['beam_c'] = [beam_c] * n_simulations
    df_mc['beam_cross_section_area'] = [dict_in['beam_cross_section_area']] * n_simulations
    df_mc['beam_temperature_goal'] = [dict_in['beam_temperature_goal']] * n_simulations
    df_mc['beam_loc_z'] = [dict_in['beam_loc_z']] * n_simulations
    df_mc['protection_k'] = [dict_in['beam_protection_k']] * n_simulations
    df_mc['protection_rho'] = [dict_in['beam_protection_rho']] * n_simulations
    df_mc['protection_c'] = [dict_in['beam_protection_c']] * n_simulations
    df_mc['protection_thickness'] = [dict_in['beam_protection_thickness']] * n_simulations
    df_mc['protection_protected_perimeter'] = [dict_in['beam_protection_protected_perimeter']] * n_simulations
    df_mc['iso834_time'] = [iso834_time] * n_simulations
    df_mc['iso834_temperature'] = [iso834_temperature] * n_simulations
    df_mc['probability_weight'] = [dict_in['probability_weight'] / n_simulations] * n_simulations
    df_mc['index'] = np.arange(0, n_simulations, 1)

    return df_mc


def main(path_input_master: str = None):

    # ==================================================================================================================
    # Parse inputs from files into list of dict()s
    # ==================================================================================================================

    if path_input_master is None:
        path_input_master = select_path_input_csv()
    path_input_master = os.path.realpath(path_input_master)
    path_work = os.path.dirname(path_input_master)

    if path_input_master.endswith('.csv'):
        df_input_params = pd.read_csv(path_input_master).set_index('PARAMETERS')
    elif path_input_master.endswith('.xlsx'):
        df_input_params = pd.read_excel(path_input_master).set_index('PARAMETERS')
    else:
        df_input_params = []
    dict_dict_input_params = df_input_params.to_dict()

    # load configuration parameters
    try:
        with open(os.path.join(path_work, 'config.json'), 'r') as f:
            dict_config_params = json.load(f)
    except (FileExistsError, FileNotFoundError) as _:
        dict_config_params_default = {
            'n_threads': 1,
            'output_fires': 0,
        }
        dict_config_params = dict_config_params_default

    # ==================================================================================================================
    # Main calculation
    # ==================================================================================================================

    dict_out = main_params(input_master=dict_dict_input_params, config_master=dict_config_params)

    for file_name, df_ in dict_out.items():
        if isinstance(df_, pd.DataFrame):
            df_.to_csv(os.path.join(path_work, file_name), index=False)
        else:
            warnings.warn('DataFrame object is expected, got {}.'.format(type(df_)))


def main_params(input_master: dict = None, config_master: dict = None):
    """DESCRIPTION
    Run Monte Carlo Simulation for mc0 (time equivalence approach 0), with inputs and outputs as variables, no system IO
    is involved.

    PARAMETERS
    :param input_master:    {case_name: {arg1: val, arg2: val, ...}}, where 'case_name' is case name (i.e. input csv
                            file's column header and arg# are stochastic/static parameters of the case.
    :param config_master:   dict, configuration definition.
    :return dict_out:       dict(DataFrame), concatenated output, mcs results and fires (if enquired).
    """

    # ==================================================================================================================
    # Parse configurations
    # ==================================================================================================================

    if config_master:
        n_threads = 1 if 'n_threads' not in config_master else config_master['n_threads']
        output_fires = 1 if 'output_fires' not in config_master else config_master['output_fires']
        output_summary = 1 if 'output_summary' not in config_master else config_master['output_summary']
    else:
        n_threads = 1
        output_fires = 0
        output_summary = 1

    # ==================================================================================================================
    # Spawn Monte Carlo parameters from dict() to DataFrame()s
    # ==================================================================================================================
    
    list_df_mc = []
    for case_name, params in input_master.items():
        params['case_name'] = case_name
        list_df_mc.append(mc_inputs_dict_to_df(params))
    df_mc_params = pd.concat(list_df_mc)

    # ==================================================================================================================
    # Main calculation
    # ==================================================================================================================

    list_mcs_out = list()
    dict_out = dict()

    for case_name in input_master.keys():

        df_mc_params_i = df_mc_params[df_mc_params['case_name']==case_name]

        list_dict_mc_params_i = df_mc_params_i.to_dict('records')

        print('{:<24.24}: {}'.format("CASE", case_name))
        print('{:<24.24}: {}'.format("NO. OF THREADS", n_threads))
        print('{:<24.24}: {}'.format("NO. OF SIMULATIONS", len(df_mc_params_i.index)))

        if n_threads == 1:
            results = [calc_time_equivalence(**dict_mc_params) for dict_mc_params in tqdm(list_dict_mc_params_i, ncols=60)]

        else:
            m = mp.Manager()
            q = m.Queue()
            p = mp.Pool(n_threads, maxtasksperchild=1000)
            jobs = p.map_async(calc_time_equivalence_worker, [(dict_, q) for dict_ in list_dict_mc_params_i])
            count_total_simulations = len(list_dict_mc_params_i)

            with tqdm(total=count_total_simulations, ncols=60) as pbar:
                while True:
                    if jobs.ready():
                        if count_total_simulations > pbar.n:
                            pbar.update(count_total_simulations - pbar.n)
                        break
                    else:
                        if q.qsize() - pbar.n > 0:
                            pbar.update(q.qsize() - pbar.n)
                        time.sleep(1)
                p.close()
                p.join()

            results = jobs.get()

        df_output = pd.DataFrame(results)
        df_output.sort_values('solver_time_equivalence_solved', inplace=True)  # sort base on time equivalence

        # export fires

        fire_time = df_output['fire_time']
        fire_temperature = df_output['fire_temperature']
        if output_fires:
            list_fire_temperature = fire_temperature.values
            dict_fire_temperature = {
                'temperature_{}'.format(v): list_fire_temperature[i] for i, v in enumerate(df_output['index'].values)
            }
            dict_fire_temperature['time'] = fire_time.iloc[0]
            df_fire_temperature = pd.DataFrame.from_dict(dict_fire_temperature)
            dict_out['.'.join([case_name, 'fires'])] = df_fire_temperature

        if output_summary:
            time.sleep(0.5)
            print(y_results_summary(df_output))

        df_output.pop('fire_time')
        df_output.pop('fire_temperature')

        list_mcs_out.append(df_output)

    dict_out['mcs.out'] = pd.concat(list_mcs_out)

    return dict_out


def _test_mc_params_1():
    _input_master_ = dict(
        example_case_1=dict(
            is_live=1,
            fire_mode=3,
            probability_weight=1,
            n_simulations=1000,
            time_step=30,
            time_duration=18000,
            fire_hrr_density_lbound=0.25,
            fire_hrr_density_ubound=0.25,
            fire_hrr_density_mean=0.25,
            fire_hrr_density_std=100,
            fire_qfd_std=126,
            fire_qfd_mean=420,
            fire_qfd_ubound=1500,
            fire_qfd_lbound=10,
            fire_spread_lbound=0.0035,
            fire_spread_ubound=0.019,
            fire_nft_mean=1050,
            fire_com_eff_lbound=0.75,
            fire_com_eff_ubound=0.999,
            fire_tlim=0.333,
            fire_t_alpha=300,
            fire_gamma_fi_q=1,
            room_breadth=16,
            room_depth=31.25,
            room_height=3,
            room_window_width=72,
            room_window_height=2.8,
            room_opening_fraction_std=0.2,
            room_opening_fraction_mean=0.2,
            room_opening_fraction_ubound=0.999,
            room_opening_fraction_lbound=0.001,
            room_opening_permanent_fraction=0,
            room_wall_thermal_inertia=720,
            beam_cross_section_area=0.017,
            beam_rho=7850,
            beam_temperature_goal=893,
            beam_protection_protected_perimeter=2.14,
            beam_protection_thickness=0,
            beam_protection_k=0.2,
            beam_protection_rho=800,
            beam_protection_c=1700,
            beam_loc_z=3,
            beam_loc_ratio_lbound=0.666,
            beam_loc_ratio_ubound=0.999,
        ),
        example_case_2 = dict(
            is_live=1,
            fire_mode=3,
            probability_weight=1,
            n_simulations=1000,
            time_step=30,
            time_duration=18000,
            fire_hrr_density_lbound=0.75,
            fire_hrr_density_ubound=0.75,
            fire_hrr_density_mean=0.75,
            fire_hrr_density_std=100,
            fire_qfd_std=126,
            fire_qfd_mean=420,
            fire_qfd_ubound=1500,
            fire_qfd_lbound=10,
            fire_spread_lbound=0.0035,
            fire_spread_ubound=0.019,
            fire_nft_mean=1050,
            fire_com_eff_lbound=0.75,
            fire_com_eff_ubound=0.999,
            fire_tlim=0.333,
            fire_t_alpha=300,
            fire_gamma_fi_q=1,
            room_breadth=16,
            room_depth=31.25,
            room_height=3,
            room_window_width=72,
            room_window_height=2.8,
            room_opening_fraction_std=0.2,
            room_opening_fraction_mean=0.2,
            room_opening_fraction_ubound=0.999,
            room_opening_fraction_lbound=0.001,
            room_opening_permanent_fraction=0,
            room_wall_thermal_inertia=720,
            beam_cross_section_area=0.017,
            beam_rho=7850,
            beam_temperature_goal=893,
            beam_protection_protected_perimeter=2.14,
            beam_protection_thickness=0,
            beam_protection_k=0.2,
            beam_protection_rho=800,
            beam_protection_c=1700,
            beam_loc_z=3,
            beam_loc_ratio_lbound=0.666,
            beam_loc_ratio_ubound=0.999,
        )
    )

    _config_master_ = dict(
        n_threads=6,
        output_fires=0,
        output_summary=1,
    )

    main_params(input_master=_input_master_, config_master=_config_master_)


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    main()
