# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('agg')

import multiprocessing as mp
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import copy
from tkinter import filedialog, Tk, StringVar
from scipy.interpolate import interp1d

from sfeprapy.dat.steel_carbon import Thermal
from sfeprapy.func.temperature_fires import parametric_eurocode1 as _fire_param
from sfeprapy.func.temperature_fires import standard_fire_iso834 as _fire_standard
from sfeprapy.func.tfm_alt import travelling_fire as _fire_travelling
from sfeprapy.time_equivalence_mc import calc_time_equiv_worker, mc_inputs_generator_core2, calc_time_equivalence

sns.set_style("ticks", {'axes.grid': True, })


def plot_dist(id_, df_input, path_input_file, headers):
    # headers = ['window_open_fraction', 'fire_load_density', 'fire_spread_speed', 'beam_position', 'temperature_max_near_field']
    # headers = ['WINDOW OPEN FRACTION [%]', 'FIRE LOAD DENSITY [MJ/m2]', 'FIRE SPREAD SPEED [m/s]', 'BEAM POSITION [m]', 'MAX. NEAR FIELD TEMPERATURE [C]']

    names = {'WINDOW OPEN FRACTION []': 'Ao',
             'FIRE LOAD DENSITY [MJ/m2]': 'qfd',
             'FIRE SPREAD SPEED [m/s]': 'spread',
             'BEAM POSITION [m]': 'beam_loc',
             'MAX. NEAR FIELD TEMPERATURE [C]': 'nft'}

    fig, ax = plt.subplots(figsize=(3.94, 2.76))  # (3.94, 2.76) for large and (2.5, 2) for small figure size

    for k, v in names.items():
        x = np.array(df_input[k].values, float)

        if k == 'MAX. NEAR FIELD TEMPERATURE [C]':
            x = x[x < 1200]

        sns.distplot(x, kde=False, rug=True, bins=50, ax=ax, norm_hist=True)

        # Normal plot parameters
        ax.set_ylabel('PDF')
        ax.set_xlabel(k)

        # Small simple plot parameters
        # ax.set_ylabel('')
        # ax.set_yticklabels([])
        # ax.set_xlabel('')

        plt.tight_layout()
        plt.savefig(
            os.path.join(os.path.dirname(path_input_file), '{} - dist - {}.png'.format(id_, v)),
            transparent=True,
            ppi=300
        )
        plt.cla()

    plt.clf()


def select_fires_teq(df_output):
    dict_fires = dict()

    # for i,v in enumerate(list_index_selected_fires) :
    args = None
    for i, v in df_output.iterrows():
        # get input arguments
        # args = df_input.loc[i].to_dict()
        args = v.to_dict()

        # get fire type
        # fire_type = int(df_output.loc[i]["FIRE TYPE [0:P, 1:T]"])
        fire_type = int(v["FIRE TYPE [0:P, 1:T]"])

        if fire_type == 0:  # parametric fire
            w, l, h = args["ROOM BREADTH [m]"], args["ROOM DEPTH [m]"], args["ROOM HEIGHT [m]"]
            inputs_parametric_fire = {
                "A_t": 2 * (w * l + w * h + h * l),
                "A_f": w * l,
                "A_v": args["WINDOW HEIGHT [m]"] * args["WINDOW WIDTH [m]"] * args["WINDOW OPEN FRACTION []"],
                "h_eq": args["WINDOW HEIGHT [m]"],
                "q_fd": args["FIRE LOAD DENSITY [MJ/m2]"] * 1e6,
                "lambda_": args["ROOM WALL THERMAL INERTIA [J/m2s1/2K]"] ** 2,
                # thermal inertia is used instead of k rho c.
                "rho": 1,  # see comment for lambda_
                "c": 1,  # see comment for lambda_
                "t_lim": args["TIME LIMITING []"],
                "time_end": args["FIRE DURATION [s]"],
                "time_step": args["TIME STEP [s]"],
                "time_start": args["TIME START [s]"],
                # "time_padding": (0, 0),
                "temperature_initial": 20 + 273.15,
            }
            tsec, temps = _fire_param(**inputs_parametric_fire)
        elif fire_type == 1:  # travelling fire
            inputs_travelling_fire = {
                "fire_load_density_MJm2": args["FIRE LOAD DENSITY [MJ/m2]"],
                "heat_release_rate_density_MWm2": args["FIRE HRR DENSITY [MW/m2]"],
                "length_compartment_m": args["ROOM DEPTH [m]"],
                "width_compartment_m": args["ROOM BREADTH [m]"],
                "fire_spread_rate_ms": args["FIRE SPREAD SPEED [m/s]"],
                "height_fuel_to_element_m": args["ROOM HEIGHT [m]"],
                "length_element_to_fire_origin_m": args["BEAM POSITION [m]"],
                "time_start_s": args["TIME START [s]"],
                "time_end_s": args["FIRE DURATION [s]"],
                "time_interval_s": args["TIME STEP [s]"],
                "nft_max_C": args["MAX. NEAR FIELD TEMPERATURE [C]"],
                "win_width_m": args["WINDOW WIDTH [m]"],
                "win_height_m": args["WINDOW HEIGHT [m]"],
                "open_fract": args["WINDOW OPEN FRACTION []"]
            }
            tsec, temps, hrr, r = _fire_travelling(**inputs_travelling_fire)
            temps += 273.15
        else:
            temps = 0
            print("FIRE TYPE UNKOWN.")

        dict_fires['FIRE {}'.format(str(i))] = temps - 273.15
        # plt.plot2(tsec/60., temps-273.15, alpha=.6)

    dict_fires["TIME [min]"] = np.arange(args["TIME START [s]"], args["FIRE DURATION [s]"], args["TIME STEP [s]"]) / 60.

    df_fires = pd.DataFrame(dict_fires)
    # list_names = ["TIME [min]"] + list_fire_name
    # df_fires = df_fires[list_names]

    # Save graphical plot to a .png file
    # ----------------------------------

    # Save numerical data to a .csv file
    # ----------------------------------
    # file_name = "{} - {}".format(os.path.basename(path_input_file).split('.')[0], __fn_fires_numerical)
    # df_fires.to_csv(os.path.join(os.path.dirname(path_input_file), file_name))
    return df_fires


def plot_figures(list_path_out_csv, plot_teq_xlim, reliability_target, path_work, **surplus_args):
    line_vertical, line_horizontal = 0, 0
    teq_fig, teq_ax = plt.subplots(figsize=(3.94, 2.76))
    teq_ax.set_xlim([0, plot_teq_xlim])

    for i, path_out_csv in enumerate(list_path_out_csv):
        df_result = pd.read_csv(path_out_csv)

        # Obtain variable of individual simulation case
        # ---------------------------------------------

        # path_input_file = list_path_input_file[i]
        # pref_ = list_pref[i]

        # Obtain some useful strings / directories / variables
        # ----------------------------------------------------

        id_ = os.path.basename(path_out_csv).replace('_out.csv', '')
        id_.replace('_out.csv', '')

        # Check 'deterministic'
        # ----------------------
        if len(df_result) <= 2:
            break

        # obtain x and y values for plot

        x = np.sort(df_result["TIME EQUIVALENCE [s]"].values) / 60.  # to minutes
        y = np.arange(1, len(x) + 1) / len(x)

        # plot the x, y

        plt.plot(x, y, label=id_)
        teq_ax.set_ylabel('Fractile')
        teq_ax.set_xlabel('Time [min]')
        teq_ax.set_xticks(ticks=np.arange(0, plot_teq_xlim + 0.001, 30))
        # plt.plot2(x, y, id_)
        # plt.format(**plt_format_)

        # plot horizontal and vertical lines (i.e. fractile and corresponding time euiqvalence period)

        if 0 < reliability_target < 1:
            line_horizontal = reliability_target
        elif reliability_target > 1:
            line_horizontal = 1 - 64.8 / reliability_target ** 2
        else:
            line_horizontal = 0

        if line_horizontal > 0:
            f_interp = interp1d(y, x)
            line_vertical = np.max((float(f_interp(line_horizontal)), line_vertical))

    if line_horizontal > 0:
        teq_ax.axhline(y=line_horizontal, c='grey')
        teq_ax.axvline(x=line_vertical, c='grey')
        teq_ax.text(
            x=line_vertical,
            y=teq_ax.get_ylim()[1],
            s="{:.0f}".format(line_vertical),
            va="bottom",
            ha="center",
            fontsize=9)

    teq_ax.legend().set_visible(True)
    teq_ax.legend(prop={'size': 7})
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_work, 't_eq.png'),
        transparent=True,
        bbox_inches='tight',
        dpi=300
    )

    plt.clf()
    plt.close()


def csv_to_jsons():
    # get a list of dict()s representing different scenarios
    root = Tk()
    root.withdraw()
    folder_path = StringVar()

    path_input_file_csv = filedialog.askopenfile(title='Select Input Files', filetypes=[('csv', ['.csv'])])
    folder_path.set(path_input_file_csv)
    root.update()

    try:
        path_input_file_csv = os.path.realpath(path_input_file_csv.name)
    except AttributeError:
        print("terminated because no files are selected.")
        return 0

    path_work = os.path.dirname(path_input_file_csv)

    # csv to dict
    df_input_params = pd.read_csv(path_input_file_csv).set_index('PARAMETERS')
    dict_dict_input_params = df_input_params.to_dict()

    # dict to json (list)
    list_path_input_jsons = []
    for key, val in dict_dict_input_params.items():
        path_input_file_json = os.path.join(path_work, 'temp', key + '.json')

        if not val['is_live']:
            continue

        list_path_input_jsons.append(path_input_file_json)
        try:
            os.mkdir(os.path.dirname(path_input_file_json))
        except FileExistsError:
            pass

        with open(path_input_file_json, 'w') as f:
            json.dump(val, f)

    return path_work, list_path_input_jsons


def run():
    # ==================================================================================================================
    # Parse inputs from files into list of dict()s
    # ==================================================================================================================

    # get a list of dict()s representing different scenarios
    # requires path_work, list_path_input_json
    # from tkinter import filedialog, Tk, StringVar
    # root = Tk()
    # root.withdraw()
    # folder_path = StringVar()
    #
    # list_path_input_json = filedialog.askopenfiles(title='Select Input Files', )
    # folder_path.set(list_path_input_json)
    # root.update()
    #
    # list_path_input_json = [os.path.realpath(i.name) for i in list_path_input_json]



    # load csv input file
    path_work, list_path_input_json = csv_to_jsons()

    # load configuration parameters
    try:
        with open(os.path.join(path_work, 'config.json'), 'r') as f:
            dict_config_params = json.load(f)
    except FileExistsError:
        dict_config_params_defualt = {
            'n_proc': 3,
            'plot_teq_xlim': 120,
            'reliability_target': 0.8,
        }
        dict_config_params = dict_config_params_defualt

    list_input_file_names = [os.path.basename(i).split('.')[0] for i in list_path_input_json]

    # plot all figures if no simulations to run
    if len(list_path_input_json) == 0:
        list_path_out_csv_all = []
        for f in os.listdir(os.path.join(path_work, 'temp')):
            if f.endswith("out.csv"):
                list_path_out_csv_all.append(os.path.join(path_work, 'temp', f))
        plot_figures(
            list_path_out_csv=list_path_out_csv_all,
            plot_teq_xlim=dict_config_params['plot_teq_xlim'],
            reliability_target=dict_config_params['reliability_target'],
            path_work=path_work
        )
        return 0

    # path_work = os.path.dirname(list_path_input_json[0])

    list_dict_input_files = []
    for i in list_path_input_json:
        with open(i, 'r') as f:
            list_dict_input_files.append(json.load(f))



    # ==================================================================================================================
    # Spawn Monte Carlo parameters from dict() to DataFrame()s
    # ==================================================================================================================

    list_df_mc_params = []

    for i, dict_mc_params in enumerate(list_dict_input_files):

        # obtain whether user defined MC parameters are provided
        try:
            is_mc_params_csv = dict_mc_params['path_mc_params_csv']
        except KeyError:
            is_mc_params_csv = False

        if not is_mc_params_csv:
            # make MC parameters
            # check if Monte Carlo input parameter *.csv exits
            file_name_ = '{}_in.csv'.format(list_input_file_names[i])
            path_input_file_ = os.path.join(path_work, file_name_)

            if os.path.exists(path_input_file_) and os.path.isfile(path_input_file_) and False:
                # todo: user defined *.csv MC input file
                df_mc_params = pd.read_csv(path_input_file_).set_index('index', drop=True)
                n_sim = len(df_mc_params.index)

            else:
                df_mc_params = mc_inputs_generator_core2(index=True, **dict_mc_params)

        else:
            # load MC parameters
            try:
                df_mc_params = pd.read_csv(dict_mc_params['path_mc_params_csv'])
            except FileNotFoundError:
                print('ERROR! *.csv file not found: {}'.format(dict_mc_params['path_mc_params_csv']))
                return -1

        list_df_mc_params.append(df_mc_params)

    # Finalise MC Parameters in DataFrame() and Save to *.csv

    # make *_in.csv input parameter files
    path_work_dump = os.path.join(path_work, 'temp')
    try:
        os.mkdir(path_work_dump)
    except FileExistsError:
        pass

    # finalise and save to *_in.csv
    beam_c = Thermal().c()
    iso834_time, iso834_temperature = _fire_standard(np.arange(0, 6 * 60 * 60, 1), 273.15 + 20)
    list_path_in_csv = []
    for i, df_mc_params in enumerate(list_df_mc_params):
        dict_mc_params_static = {
            'time_step': list_dict_input_files[i]['time_step'],
            'time_limiting': list_dict_input_files[i]['fire_tlim'],
            'window_height': list_dict_input_files[i]['room_window_height'],
            'window_width': list_dict_input_files[i]['room_window_width'],
            'room_breadth': list_dict_input_files[i]['room_breadth'],
            'room_depth': list_dict_input_files[i]['room_depth'],
            'room_height': list_dict_input_files[i]['room_height'],
            'room_wall_thermal_inertia': list_dict_input_files[i]['room_wall_thermal_inertia'],
            'fire_hrr_density': list_dict_input_files[i]['fire_hrr_density'],
            'fire_duration': list_dict_input_files[i]['time_duration'],
            'beam_rho': list_dict_input_files[i]['beam_rho'],
            'beam_c': beam_c,
            'beam_cross_section_area': list_dict_input_files[i]['beam_cross_section_area'],
            'beam_temperature_goal': list_dict_input_files[i]['beam_temperature_goal'],
            'beam_loc_z': list_dict_input_files[i]['beam_loc_z'],
            'protection_k': list_dict_input_files[i]['beam_protection_k'],
            'protection_rho': list_dict_input_files[i]['beam_protection_rho'],
            'protection_c': list_dict_input_files[i]['beam_protection_c'],
            'protection_thickness': list_dict_input_files[i]['beam_protection_thickness'],
            'protection_protected_perimeter': list_dict_input_files[i]['beam_protection_protected_perimeter'],
            'iso834_time': iso834_time,
            'iso834_temperature': iso834_temperature,
            'time_start': list_dict_input_files[i]['time_start'],
        }
        # dict_mc_params_static = {j:[dict_mc_params_static[j]]*len(df_mc_params.index) for j in dict_mc_params_static}
        for key, val in dict_mc_params_static.items():
            list_val = []
            for i_ in range(len(df_mc_params.index)):
                list_val.append(val)
            dict_mc_params_static[key] = list_val

        dict_mc_params = {j: df_mc_params[j].values for j in df_mc_params}
        dict_mc_params.update(dict_mc_params_static)
        dict_mc_params['index'] = df_mc_params.index.values

        df_mc_params = pd.DataFrame.from_dict(dict_mc_params).set_index('index')

        file_name_ = '{}_in.csv'.format(list_input_file_names[i])
        path_input_file_ = os.path.join(path_work_dump, file_name_)

        try:
            df_mc_params.to_csv(path_input_file_)
            list_path_in_csv.append(path_input_file_)
        except PermissionError:
            print('WARNING! File save failed: {}'.format(path_input_file_))
            pass

    # load *.csv
    list_df_mc_params = [pd.read_csv(i).set_index('index', drop=False) for i in list_path_in_csv]
    # replace str with objects, e.g. list()
    for i, v in enumerate(list_df_mc_params):
        dict_ = v.to_dict(orient='list')
        dict_['beam_c'] = [beam_c for i in range(len(v.index))]
        dict_['iso834_time'] = [iso834_time for i in range(len(v.index))]
        dict_['iso834_temperature'] = [iso834_temperature for i in range(len(v.index))]
        v = pd.DataFrame.from_dict(dict_).set_index('index')
        list_df_mc_params[i] = copy.deepcopy(v)

    # convert DataFrame() to list of dict()s
    list_list_dict_mc_params = []
    for i, dict_mc_params in enumerate(list_df_mc_params):
        list_dict_mc_params = []
        for j, w in dict_mc_params.iterrows():
            x = w.to_dict()
            x['index'] = j
            list_dict_mc_params.append(x)
        list_list_dict_mc_params.append(list_dict_mc_params)

    # ==================================================================================================================
    # Main calculation
    # ==================================================================================================================

    list_path_out_csv = []

    for i, list_dict_mc_params in enumerate(list_list_dict_mc_params):

        print("CASE:", list_input_file_names[i])
        print("NO. OF THREADS:", dict_config_params['n_proc'])
        print("NO. OF SIMULATION:", len(list_dict_mc_params))

        if dict_config_params['n_proc'] == 1:
            results = []

            for dict_mc_params in list_dict_mc_params:
                results.append(calc_time_equivalence(**dict_mc_params))

        else:
            time_simulation_start = time.perf_counter()
            m = mp.Manager()
            q = m.Queue()
            p = mp.Pool(int(dict_config_params['n_proc']), maxtasksperchild=1000)
            jobs = p.map_async(calc_time_equiv_worker, [(dict_, q) for dict_ in list_dict_mc_params])
            count_total_simulations = len(list_dict_mc_params)
            n_steps = 24  # length of the progress bar
            while True:
                if jobs.ready():
                    time_simulation_consumed = time.perf_counter() - time_simulation_start
                    print("{}{} {:.1f}s ".format('█' * round(n_steps), '-' * round(0), time_simulation_consumed))
                    break
                else:
                    p_ = q.qsize() / count_total_simulations * n_steps
                    print("{}{} {:03.1f}%".format('█' * int(round(p_)), '-' * int(n_steps - round(p_)),
                                                  p_ / n_steps * 100),
                          end='\r')
                    time.sleep(1)
            p.close()
            p.join()
            results = jobs.get()

        # save to *.csv
        results = np.array(results)
        list_path_out_csv.append(os.path.join(path_work, 'temp', '{}_out.csv'.format(list_input_file_names[i])))
        df_output = pd.DataFrame({'TIME STEP [s]': results[:, 0],
                                  'TIME START [s]': results[:, 1],
                                  'TIME LIMITING []': results[:, 2],
                                  'WINDOW HEIGHT [m]': results[:, 3],
                                  'WINDOW WIDTH [m]': results[:, 4],
                                  'WINDOW OPEN FRACTION []': results[:, 5],
                                  'ROOM BREADTH [m]': results[:, 6],
                                  'ROOM DEPTH [m]': results[:, 7],
                                  'ROOM HEIGHT [m]': results[:, 8],
                                  'ROOM WALL THERMAL INERTIA [J/m2s1/2K]': results[:, 9],
                                  'FIRE LOAD DENSITY [MJ/m2]': results[:, 10],
                                  'FIRE HRR DENSITY [MW/m2]': results[:, 11],
                                  'FIRE SPREAD SPEED [m/s]': results[:, 12],
                                  'FIRE DURATION [s]': results[:, 13],
                                  'BEAM POSITION [m]': results[:, 14],
                                  'BEAM RHO [kg/m3]': results[:, 15],
                                  'BEAM C [-]': results[:, 16],
                                  'BEAM CROSS-SECTION AREA [m2]': results[:, 17],
                                  'BEAM FAILURE TEMPERATURE [C]': results[:, 18],
                                  'PROTECTION K [W/m/K]': results[:, 19],
                                  'PROTECTION RHO [kg/m3]': results[:, 20],
                                  'PROTECTION C OBJECT []': results[:, 21],
                                  'PROTECTION THICKNESS [m]': results[:, 22],
                                  'PROTECTION PERIMETER [m]': results[:, 23],
                                  'ISO834 TIME ARRAY [s]': results[:, 24],
                                  'ISO834 TEMPERATURE ARRAY [K]': results[:, 25],
                                  'MAX. NEAR FIELD TEMPERATURE [C]': results[:, 26],
                                  'SEEK ITERATION LIMIT []': results[:, 27],
                                  'SEEK PROTECTION THICKNESS UPPER BOUND [m]': results[:, 28],
                                  'SEEK PROTECTION THICKNESS LOWER BOUND [m]': results[:, 29],
                                  'SEEK BEAM FAILURE TEMPERATURE TOLERANCE [K]': results[:, 30],
                                  'INDEX': results[:, 31],
                                  'TIME EQUIVALENCE [s]': results[:, 32],
                                  'SEEK STATUS [0:Fail, 1:Success]': results[:, 33],
                                  'FIRE TYPE [0:P, 1:T]': results[:, 34],
                                  'SOUGHT BEAM TEMPERATURE [K]': results[:, 35],
                                  'SOUGHT BEAM PROTECTION THICKNESS [m]': results[:, 36],
                                  'SOUGHT ITERATIONS []': results[:, 37],
                                  'BEAM TEMPERATURE TO FIXED PROTECTION THICKNESS [K]': results[:, 38],
                                  'FIRE TIME ARRAY [s]': results[:, 39],
                                  'FIRE TEMPERATURE ARRAY [K]': results[:, 40],
                                  'OPENING FACTOR [m0.5]': results[:, 41]
                                  })
        df_output.set_index("INDEX", inplace=True)  # assign 'INDEX' column as DataFrame index
        df_output.sort_values('TIME EQUIVALENCE [s]', inplace=True)  # sort base on time equivalence
        df_output.to_csv(list_path_out_csv[-1])

        # export fires
        df_fires = select_fires_teq(df_output)
        df_fires.to_csv(os.path.join(path_work, 'temp', '{}_fires.csv'.format(list_input_file_names[i])))

    # ==================================================================================================================
    # Postprocessing - plot figures
    # ==================================================================================================================

    # Requirements
    # list_path_out_csv: a list of numeral output file path in *.csv format

    list_path_out_csv_all = []
    for f in os.listdir(os.path.join(path_work, 'temp')):
        if f.endswith("out.csv"):
            list_path_out_csv_all.append(os.path.join(path_work, 'temp', f))

    plot_figures(
        list_path_out_csv=list_path_out_csv_all,
        plot_teq_xlim=dict_config_params['plot_teq_xlim'],
        reliability_target=dict_config_params['reliability_target'],
        path_work=path_work
    )


if __name__ == '__main__':
    run()
