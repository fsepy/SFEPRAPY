# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('agg')

import multiprocessing as mp
import os
import time
import numpy as np
import pandas as pd
import json
import warnings
from tkinter import filedialog, Tk, StringVar
from typing import Union

from tqdm import tqdm
from sfeprapy.func.fire_iso834 import fire as _fire_standard
from sfeprapy.mc0.mc0_func_main import calc_time_equivalence_worker, calc_time_equivalence, y_results_summary
from sfeprapy.mc0.mc0_func_post import extract_results
from sfeprapy.mc0.mc0_func_gen import mc_inputs_generator


def select_path_input_csv():

    # get a list of dict()s representing different scenarios

    root = Tk()
    root.withdraw()
    folder_path = StringVar()

    path_input_file_csv = filedialog.askopenfile(title='Select Input File',
                                                 parent=root,
                                                 filetypes=[('SPREADSHEET', ['.csv', '.xlsx'])],
                                                 mode='r')

    folder_path.set(path_input_file_csv)

    root.update()

    try:
        path_input_file_csv = os.path.realpath(path_input_file_csv.name)
        return path_input_file_csv
    except AttributeError:
        raise FileNotFoundError('file not found.')


def input_csv_to_jsons(path_input_file_csv):

    path_work = os.path.dirname(path_input_file_csv)

    # ========================================================
    # PROCESS AND SAVE *.JSON INPUT FILES FOR INDIVIDUAL CASES
    # ========================================================
    # csv to dict
    if path_input_file_csv.endswith('.csv'):
        df_input_params = pd.read_csv(path_input_file_csv).set_index('PARAMETERS')
    elif path_input_file_csv.endswith('.xlsx'):
        df_input_params = pd.read_excel(path_input_file_csv).set_index('PARAMETERS')
    else:
        df_input_params = []

    dict_dict_input_params = df_input_params.to_dict()

    # dict to json (list)
    list_path_input_jsons = []
    for key, val in dict_dict_input_params.items():
        path_input_file_json = os.path.join(path_work, 'temp', key + '.json')

        list_path_input_jsons.append(path_input_file_json)

        try:
            os.mkdir(os.path.dirname(path_input_file_json))
        except FileExistsError:
            pass

        with open(path_input_file_json, 'w') as f:
            json.dump(val, f)

    return path_work, list_path_input_jsons


def jsons_to_mc_input_df(list_path_json, save_csv=False):

    list_df_mc_params = []

    # finalise and save to *_in.csv
    # beam_c = Thermal().c()
    beam_c = 0
    iso834_time = np.arange(0, 6 * 60 * 60, 30)
    iso834_temperature = _fire_standard(iso834_time, 273.15 + 20)

    list_dict_params_from_json = []
    for path_json in list_path_json:
        with open(path_json, 'r') as f:
            list_dict_params_from_json.append(json.load(f))

    for i, dict_params_from_json in enumerate(list_dict_params_from_json):

        try:
            if dict_params_from_json['is_live'] == 0:
                list_df_mc_params.append(None)
                continue
        except KeyError:
            warnings.warn("Key 'is_line' does not exit. Proceeded by default.")

        df_mc_params = mc_inputs_generator(index=True, **dict_params_from_json)

        dict_params_from_json = list_dict_params_from_json[i]
        df_mc_params['time_step'] = [dict_params_from_json['time_step']] * len(df_mc_params.index)
        df_mc_params['time_limiting'] = [dict_params_from_json['fire_tlim']] * len(df_mc_params.index)
        df_mc_params['window_height'] = [dict_params_from_json['room_window_height']] * len(df_mc_params.index)
        df_mc_params['window_width'] = [dict_params_from_json['room_window_width']] * len(df_mc_params.index)
        df_mc_params['room_breadth'] = [dict_params_from_json['room_breadth']] * len(df_mc_params.index)
        df_mc_params['room_depth'] = [dict_params_from_json['room_depth']] * len(df_mc_params.index)
        df_mc_params['room_height'] = [dict_params_from_json['room_height']] * len(df_mc_params.index)
        df_mc_params['room_wall_thermal_inertia'] = [dict_params_from_json['room_wall_thermal_inertia']] * len(df_mc_params.index)
        # df_mc_params['fire_hrr_density'] = [dict_params_from_json['fire_hrr_density']] * len(df_mc_params.index)
        df_mc_params['fire_duration'] = [dict_params_from_json['time_duration']] * len(df_mc_params.index)
        df_mc_params['fire_mode'] = [dict_params_from_json['fire_mode']] * len(df_mc_params.index)
        df_mc_params['fire_t_alpha'] = [dict_params_from_json['fire_t_alpha']] * len(df_mc_params.index)
        df_mc_params['fire_gamma_fi_q'] = [dict_params_from_json['fire_gamma_fi_q']] * len(df_mc_params.index)
        df_mc_params['beam_c'] = [beam_c] * len(df_mc_params.index)
        df_mc_params['beam_cross_section_area'] = [dict_params_from_json['beam_cross_section_area']] * len(df_mc_params.index)
        df_mc_params['beam_temperature_goal'] = [dict_params_from_json['beam_temperature_goal']] * len(df_mc_params.index)
        df_mc_params['beam_loc_z'] = [dict_params_from_json['beam_loc_z']] * len(df_mc_params.index)
        df_mc_params['protection_k'] = [dict_params_from_json['beam_protection_k']] * len(df_mc_params.index)
        df_mc_params['protection_rho'] = [dict_params_from_json['beam_protection_rho']] * len(df_mc_params.index)
        df_mc_params['protection_c'] = [dict_params_from_json['beam_protection_c']] * len(df_mc_params.index)
        df_mc_params['protection_thickness'] = [dict_params_from_json['beam_protection_thickness']] * len(df_mc_params.index)
        df_mc_params['protection_protected_perimeter'] = [dict_params_from_json['beam_protection_protected_perimeter']] * len(df_mc_params.index)
        df_mc_params['iso834_time'] = [iso834_time] * len(df_mc_params.index)
        df_mc_params['iso834_temperature'] = [iso834_temperature] * len(df_mc_params.index)
        df_mc_params['probability_weight'] = [dict_params_from_json['probability_weight'] / len(df_mc_params.index)] * len(df_mc_params.index)
        df_mc_params['index'] = np.arange(0, len(df_mc_params.index), 1)

        list_df_mc_params.append(df_mc_params)

        if save_csv:
            fn_ = '{}_in.csv'.format(os.path.basename(list_path_json[i]).replace('.json', ''))
            pf_ = os.path.join(os.path.dirname(list_path_json[i]), fn_)

            try:
                df_mc_params.to_csv(pf_)
            except PermissionError:
                warnings.warn('WARNING! File save failed: {}'.format(pf_))

    return list_df_mc_params


def main(path_master_input_csv: Union[str, pd.DataFrame] = None):

    list_input_file_names = list()

    # ==================================================================================================================
    # Parse inputs from files into list of dict()s
    # ==================================================================================================================

    if path_master_input_csv is None:
        path_master_input_csv = select_path_input_csv()

    path_work, list_path_input_json = input_csv_to_jsons(path_master_input_csv)

    list_dict_input_jason = []
    for i in list_path_input_json:
        with open(i, 'r') as f:
            list_dict_input_jason.append(json.load(f))

    # load configuration parameters
    try:
        with open(os.path.join(path_work, 'config.json'), 'r') as f:
            dict_config_params = json.load(f)
    except FileExistsError:
        dict_config_params_default = {
            'n_threads': 1,
            'output_fires': 0,
        }
        dict_config_params = dict_config_params_default

    list_input_file_names = [os.path.basename(i).split('.')[0] for i in list_path_input_json]

    # ==================================================================================================================
    # Spawn Monte Carlo parameters from dict() to DataFrame()s
    # ==================================================================================================================

    list_df_mc_params = jsons_to_mc_input_df(list_path_json=list_path_input_json, save_csv=False)

    # convert DataFrame() to list of dict()s
    list_list_dict_mc_params = []
    for i, dict_mc_params in enumerate(list_df_mc_params):

        if dict_mc_params is None:
            list_list_dict_mc_params.append(None)
        else:
            list_dict_mc_params = []
            for j, w in dict_mc_params.iterrows():
                x = w.to_dict()
                list_dict_mc_params.append(x)
            list_list_dict_mc_params.append(list_dict_mc_params)

    # ==================================================================================================================
    # Main calculation
    # ==================================================================================================================

    list_path_out_csv = []

    for i, list_dict_mc_params in enumerate(list_list_dict_mc_params):

        if list_dict_mc_params is None:
            continue

        print('{:<24.24}: {}'.format("CASE", list_input_file_names[i]))
        print('{:<24.24}: {}'.format("NO. OF THREADS", dict_config_params['n_threads']))
        print('{:<24.24}: {}'.format("NO. OF SIMULATIONS", len(list_dict_mc_params)))

        if dict_config_params['n_threads'] == 1:
            results = []

            for dict_mc_params in tqdm(list_dict_mc_params):
                results.append(calc_time_equivalence(**dict_mc_params))
                time.sleep(0.001)

        else:
            m = mp.Manager()
            q = m.Queue()
            p = mp.Pool(int(dict_config_params['n_threads']), maxtasksperchild=1000)
            jobs = p.map_async(calc_time_equivalence_worker, [(dict_, q) for dict_ in list_dict_mc_params])
            count_total_simulations = len(list_dict_mc_params)

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

        # save to *.csv
        # results = np.array(results)
        list_path_out_csv.append(os.path.join(path_work, 'temp', '{}_out.csv'.format(list_input_file_names[i])))
        df_output = pd.DataFrame(results)
        # df_output.set_index("index", inplace=True)  # assign 'INDEX' column as DataFrame index
        df_output.sort_values('solver_time_equivalence_solved', inplace=True)  # sort base on time equivalence

        # export fires

        fire_time = df_output['fire_time']
        fire_temperature = df_output['fire_temperature']

        if 'output_fires' in dict_config_params:
            if dict_config_params["output_fires"] == 1:
                list_fire_temperature = fire_temperature.values
                dict_fire_temperature = {'temperature_{}'.format(v): list_fire_temperature[i] for i, v in enumerate(df_output['index'].values)}
                dict_fire_temperature['time'] = fire_time.iloc[0]
                df_fire_temperature = pd.DataFrame.from_dict(dict_fire_temperature)
                df_fire_temperature.to_csv(os.path.join(path_work, 'temp', list_input_file_names[i]+'_fires.csv'), index=False)

        if 'output_summary' in dict_config_params:
            if int(dict_config_params['output_summary']) > 0: 
                print(y_results_summary(df_output))

        df_output.pop('fire_time')
        df_output.pop('fire_temperature')
        df_output.to_csv(list_path_out_csv[-1], index=False)

    pp = list()
    for fn in os.listdir(os.path.join(path_work, 'temp')):
        if fn.endswith('_out.csv'):
            pp.append(os.path.join(path_work, 'temp', fn))

    dict_df_out = extract_results(*pp)
    for k, df in dict_df_out.items():
        df.to_csv(os.path.join(path_work, k+'.csv'), index=False)


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    main()
