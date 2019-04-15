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
import warnings
from tkinter import filedialog, Tk, StringVar
from scipy.interpolate import interp1d
from itertools import cycle
from collections import OrderedDict

from sfeprapy.dat.steel_carbon import Thermal
from sfeprapy.func.fire_parametric_ec import fire as _fire_param
from sfeprapy.func.fire_iso834 import fire as _fire_standard
from sfeprapy.func.fire_travelling import fire as _fire_travelling
from sfeprapy.mc1.mc_simulation_func2 import calc_time_equivalence_worker, calc_time_equivalence
from sfeprapy.mc1.mc_inputs_generator import mc_inputs_generator
from sfeprapy.func.fire_parametric_ec_din import fire as _fire_param_ger


class MonteCarloCase:

    # this dictionary object defines all user input variable names as key, and corresponding values:
    # 0 - user input, static;
    # 1 - user input, stochastic; and
    # 2 - derived input
    PARAM = dict(window_height=0, window_width=0, window_open_fraction=1, room_breadth=0, room_depth=0, room_height=0,
                 room_wall_thermal_inertia=0, fire_mode=0, fire_time_step=0, fire_iso834_time=2,
                 fire_iso834_temperature=2, fire_tlim=0, fire_load_density=1, fire_hrr_density=0,
                 fire_spread_speed=1, fire_nft_ubound=1, fire_duration=0, fire_t_alpha=0, fire_gamma_fi_q=0,
                 beam_position=1, beam_rho=0, beam_c=2, beam_cross_section_area=0, beam_loc_z=0, protection_k=0,
                 protection_rho=0, protection_c=0, protection_protected_perimeter=0, solver_temperature_target=0,
                 solver_time_duration=0, solver_thickness_lbound=0, solver_thickness_ubound=0, solver_tol=0,
                 solver_iteration_limit=0, index=2)

    def __init__(self, path_wd, name):
        self._path_wd = path_wd
        self._input_param = None
        self._name = name

        # flags
        self._is_live = False
        self._flag_output_numeric_results = False
        self._flag_output_numeric_fires = False
        self._flag_output_plot_teq = False
        self._flag_output_plot_dist = False

        # user defined preferences
        self._n_simulations = 1000
        self._n_threads = 2

        # generated data
        self._mc_param = None
        self._mc_results = None

        self.timer_mc = None

    @property
    def input_param(self):
        return self._input_param

    @property
    def mc_param(self):
        return self._mc_param

    @property
    def mc_param_list(self):
        return self._mc_param.to_dict('records')

    @property
    def n_threads(self):
        return self._n_threads

    @property
    def n_simulations(self):
        return int(self._n_simulations)

    @property
    def mc_results(self):
        return self._mc_results

    @input_param.setter
    def input_param(self, input_param):
        if not isinstance(input_param, dict):
            raise TypeError('Input parameters variable (input_param) should be a dict object.')
        for key, val in self.PARAM.items():
            if val != 0:
                continue
            if key not in input_param:
                raise ValueError('Could not find {} in input_param.'.format(key))

        self.n_threads = input_param['n_threads']
        self.n_simulations = input_param['n_simulations']
        self._input_param = input_param

    @mc_param.setter
    def mc_param(self, mc_param):
        if not isinstance(self, pd.DataFrame):
            raise TypeError('MC parameters variable (mc_param) should be a DataFrame object.')
        if not len(mc_param.index) == self.n_simulations:
            raise ValueError('Length mismatch, len(mc_param)={} != len(n_simulations)={}'.format(
                len(mc_param.index), self.n_simulations
            ))
        self._mc_param = mc_param

    @mc_results.setter
    def mc_results(self, mc_results):
        if not isinstance(mc_results, pd.DataFrame):
            raise TypeError('MC results variable (mc_results) should be a DataFrame object.')
        self._mc_results = mc_results

    @n_threads.setter
    def n_threads(self, n_threads):
        if n_threads is None:
            n_threads = 1
        if n_threads < 1:
            warnings.warn("Number of threads can only be a positive integer, your input value is {} and this is changed"
                          " to 1.".format(n_threads))
            n_threads = 1
        elif n_threads != int(n_threads):
            warnings.warn("Number of threads can only be a positive integer, your input value is {} and this is changed"
                          " to {}.".format(n_threads, int(n_threads)))
            n_threads = int(n_threads)
        self._n_threads = n_threads

    @n_simulations.setter
    def n_simulations(self, n_simulations):
        if n_simulations is None:
            n_simulations = int(10)
        if n_simulations < 1:
            warnings.warn("Number of threads can only be a positive integer, your input value is {} and this is changed"
                          " to 1.".format(n_simulations))
            n_simulations = int(1)
        elif n_simulations != int(n_simulations):
            warnings.warn("Number of threads can only be a positive integer, your input value is {} and this is changed"
                          " to {}.".format(n_simulations, int(n_simulations)))
            n_simulations = int(n_simulations)
        self._n_simulations = n_simulations

    def generate_mc_param(self):

        if self.input_param is None:
            raise ValueError('input_param is not defined.')

        input_param = self.input_param

        self.n_simulations = input_param['n_simulations']
        self.n_threads = input_param['n_threads']

        # finalise and save to *_in.csv
        iso834_time = np.arange(0, 6 * 60 * 60, 30)
        iso834_temperature = _fire_standard(iso834_time, 273.15 + 20)

        df_mc_params = mc_inputs_generator(index=True, **self.input_param)

        for key, val in self.PARAM.items():
            if val == 0:
                df_mc_params[key] = np.full(self.n_simulations, self.input_param[key])

        df_mc_params['fire_iso834_time'] = [iso834_time] * self.n_simulations
        df_mc_params['fire_iso834_temperature'] = [iso834_temperature] * self.n_simulations
        df_mc_params['index'] = np.arange(0, self.n_simulations, 1)

        df_mc_params.index.name = ['index']

        self._mc_param = df_mc_params

    def run_monte_carlo_simulation(self):
        if self.input_param is None:
            raise ValueError('input_param is not defined')
        if self.mc_param is None:
            raise ValueError('mc_param is not defined.')

        mc_param_list = self.mc_param_list
        n_threads = self.n_threads
        n_simulations = self.n_simulations

        if self.n_threads <= 1:
            results = []
            for dict_mc_params in mc_param_list:
                results.append(calc_time_equivalence(**dict_mc_params))

        else:
            time_simulation_start = time.perf_counter()
            m = mp.Manager()
            q = m.Queue()
            p = mp.Pool(n_threads, maxtasksperchild=1000)
            jobs = p.map_async(calc_time_equivalence_worker, [(dict_mc_param, q) for dict_mc_param in mc_param_list])
            n_steps = 24  # length of the progress bar
            while True:
                if jobs.ready():
                    time_simulation_consumed = time.perf_counter() - time_simulation_start
                    print("{}{} {:.1f}s ".format('█' * round(n_steps), '-' * round(0), time_simulation_consumed))
                    break
                else:
                    p_ = q.qsize() / n_simulations * n_steps
                    print("{}{} {:03.1f}%".format('█' * int(round(p_)), '-' * int(n_steps - round(p_)),
                                                  p_ / n_steps * 100),
                          end='\r')
                    time.sleep(1)
            p.close()
            p.join()
            results = jobs.get()

        df_output = pd.DataFrame(results)
        df_output.set_index("index", inplace=True)  # assign 'index' column as DataFrame index
        df_output.sort_values('solver_steel_temperature_solved', inplace=True)  # sort base on time equivalence

        self.mc_results = df_output

    def mc_results_to_csv(self, path=None):
        if self.mc_results is None:
            warnings.warn('No results to save.')
            return -1
        if path is None:
            path = os.path.join(self._path_wd, self._name+'.csv')
        try:
            self.mc_results.to_csv(path)
        except PermissionError:
            self.mc_results.to_csv(path+'1')

    def mc_results_get(self, name):
        return self.mc_results[name].values



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
        w, l, h = args["ROOM BREADTH [m]"], args["ROOM DEPTH [m]"], args["ROOM HEIGHT [m]"]
        t = np.arange(args["TIME START [s]"], args["FIRE DURATION [s]"], args["TIME STEP [s]"])

        # get fire type
        # fire_type = int(df_output.loc[i]["FIRE TYPE [0:P, 1:T]"])
        fire_type = int(v["FIRE TYPE [0:P, 1:T]"])

        if fire_type == 0:  # parametric fire

            inputs_parametric_fire = {
                "t": t,
                "A_t": 2 * (w * l + w * h + h * l),
                "A_f": w * l,
                "A_v": args["WINDOW HEIGHT [m]"] * args["WINDOW WIDTH [m]"] * args["WINDOW OPEN FRACTION []"],
                "h_eq": args["WINDOW HEIGHT [m]"],
                "q_fd": args["FIRE LOAD DENSITY [MJ/m2]"] * 1e6,
                "lambda_": args["ROOM WALL THERMAL INERTIA [J/m2s1/2K]"] ** 2,
                # thermal inertia (lambda_) is used instead of k rho c.
                "rho": 1,  # see comment for lambda_
                "c": 1,  # see comment for lambda_
                "t_lim": args["TIME LIMITING []"],
                # "time_end": args["FIRE DURATION [s]"],
                # "time_step": args["TIME STEP [s]"],
                # "time_start": args["TIME START [s]"],
                # "time_padding": (0, 0),
                "temperature_initial": 20 + 273.15,
            }
            temps = _fire_param(**inputs_parametric_fire)
        elif fire_type == 1:  # travelling fire
            inputs_travelling_fire = {
                "t": t,
                "fire_load_density_MJm2": args["FIRE LOAD DENSITY [MJ/m2]"],
                "heat_release_rate_density_MWm2": args["FIRE HRR DENSITY [MW/m2]"],
                "length_compartment_m": args["ROOM DEPTH [m]"],
                "width_compartment_m": args["ROOM BREADTH [m]"],
                "fire_spread_rate_ms": args["FIRE SPREAD SPEED [m/s]"],
                "height_fuel_to_element_m": args["ROOM HEIGHT [m]"],
                "length_element_to_fire_origin_m": args["BEAM POSITION [m]"],
                # "time_start_s": args["TIME START [s]"],
                # "time_end_s": args["FIRE DURATION [s]"],
                # "time_interval_s": args["TIME STEP [s]"],
                "nft_max_C": args["MAX. NEAR FIELD TEMPERATURE [C]"],
                "win_width_m": args["WINDOW WIDTH [m]"],
                "win_height_m": args["WINDOW HEIGHT [m]"],
                "open_fract": args["WINDOW OPEN FRACTION []"]
            }
            temps = _fire_travelling(**inputs_travelling_fire)
            temps += 273.15
        elif fire_type == 2:
            temps = _fire_param_ger(
                t_array_s=t,
                A_w_m2=args["WINDOW WIDTH [m]"] * args["WINDOW HEIGHT [m]"] * args["WINDOW OPEN FRACTION []"],
                h_w_m2=args["WINDOW HEIGHT [m]"],
                A_t_m2=2 * (w * l + w * h + h * l),
                A_f_m2=w * l,
                t_alpha_s=600,  # todo
                b_Jm2s05K=1500,  # todo
                q_x_d_MJm2=args["FIRE LOAD DENSITY [MJ/m2]"],
                gamma_fi_Q=1.0  # todo
            )
        else:
            temps = 0
            print("FIRE TYPE UNKOWN.")

        dict_fires['FIRE {}'.format(str(i))] = temps - 273.15

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


def plot_figures(list_path_out_csv, plot_teq_xlim, reliability_target, path_work, cycle_linestyles,
                 plot_figuresize=(3.94, 2.76),
                 figure_name='t_eq.png', apply_probability=False, figure_show_legend=True, **surplus_args):
    # PREFERENCES
    line_vertical, line_horizontal = 0, 0
    teq_fig, teq_ax = plt.subplots(figsize=plot_figuresize)
    teq_ax.set_xlim([0, plot_teq_xlim])
    linewidth_curves = 1
    linewidth_straight = 1

    for i, path_out_csv in enumerate(list_path_out_csv):
        df_result = pd.read_csv(path_out_csv)

        # Obtain variable of individual simulation case
        # ---------------------------------------------

        # path_input_file = list_path_input_file[i]
        # pref_ = list_pref[i]

        # Obtain some useful strings / directories / variables
        # ----------------------------------------------------

        id_ = os.path.basename(path_out_csv).replace('_out.csv', '').replace('_mergedout.csv', '')

        # Check 'deterministic'
        # ----------------------
        if len(df_result) <= 2:
            break

        # obtain x and y values for plot

        mask_teq_sort = np.asarray(df_result["TIME EQUIVALENCE [s]"].values).argsort()
        x = df_result["TIME EQUIVALENCE [s]"].values[mask_teq_sort] / 60.  # to minutes

        if 'PROBABILITY' in df_result and apply_probability:
            y = np.cumsum(df_result["PROBABILITY"].values[mask_teq_sort])
        else:
            y = np.arange(1, len(x) + 1) / len(x)

        # plot the x, y

        teq_ax.plot(x, y, label=id_, linestyle=next(cycle_linestyles), linewidth=linewidth_curves)
        teq_ax.set_ylabel('Fractile')
        teq_ax.set_xlabel('Time [min]')
        teq_ax.set_xticks(ticks=np.arange(0, plot_teq_xlim + 0.001, 30))

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
        teq_ax.axhline(y=line_horizontal, c='grey', linewidth=linewidth_straight)
        teq_ax.axvline(x=line_vertical, c='grey', linewidth=linewidth_straight)
        teq_ax.text(
            x=line_vertical,
            y=teq_ax.get_ylim()[1],
            s="{:.0f}".format(line_vertical),
            va="bottom",
            ha="center",
            fontsize=9)

    teq_ax.legend().set_visible(figure_show_legend)
    teq_ax.legend(prop={'size': 7})
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_work, figure_name),
        transparent=True,
        bbox_inches='tight',
        dpi=300
    )

    plt.clf()
    plt.close()


def select_path_input_csv():
    # get a list of dict()s representing different scenarios
    root = Tk()
    root.withdraw()
    folder_path = StringVar()

    path_input_file_csv = filedialog.askopenfile(title='Select Input Files', filetypes=[('csv', ['.csv'])])
    folder_path.set(path_input_file_csv)
    root.update()

    try:
        path_input_file_csv = os.path.realpath(path_input_file_csv.name)
        return path_input_file_csv
    except AttributeError:
        warnings.warn("terminated because no files are selected.")
        return -1


def csv_to_jsons(path_input_file_csv, dict_params_additional=None):
    path_work = os.path.dirname(path_input_file_csv)

    # ========================================================
    # PROCESS AND SAVE *.JSON INPUT FILES FOR INDIVIDUAL CASES
    # ========================================================

    # csv to dict
    df_input_params = pd.read_csv(path_input_file_csv).set_index('PARAMETERS')
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
            if dict_params_additional is not None:
                val.update(dict_params_additional)
            json.dump(val, f)

    return path_work, list_path_input_jsons


def jsons_to_mc_input_df(list_path_json, save_csv=False):
    list_df_mc_params = []

    # finalise and save to *_in.csv
    beam_c = Thermal().c()
    beam_c = 0
    iso834_time = np.arange(0, 6 * 60 * 60, 30)
    iso834_temperature = _fire_standard(iso834_time, 273.15 + 20)

    list_dict_params_from_json = []
    for path_json in list_path_json:
        with open(path_json, 'r') as f:
            list_dict_params_from_json.append(json.load(f))

    # full list of mc routin input parameters. 0 - static, 1 - stochastic, 2 - calculated
    dict_param_names = {

        "window_height": 0,
        "window_width": 0,
        "window_open_fraction": 1,

        "room_breadth": 0,
        "room_depth": 0,
        "room_height": 0,
        "room_wall_thermal_inertia": 0,

        "fire_mode": 0,
        "fire_time_step": 0,
        "fire_iso834_time": 2,
        "fire_iso834_temperature": 2,
        "fire_time_limiting": 0,
        "fire_load_density": 1,
        "fire_hrr_density": 0,
        "fire_spread_speed": 1,
        "fire_nft_ubound": 1,
        "fire_duration": 0,
        "fire_t_alpha": 0,
        "fire_gamma_fi_q": 0,

        "beam_position": 1,
        "beam_rho": 0,
        "beam_c": 2,
        "beam_cross_section_area": 0,
        "beam_loc_z": 0,

        "protection_k": 0,
        "protection_rho": 0,
        "protection_c": 0,
        "protection_protected_perimeter": 0,

        "solver_temperature_target": 0,
        "solver_time_duration": 0,
        "solver_thickness_lbound": 0,
        "solver_thickness_ubound": 0,
        "solver_tol": 0,
        "solver_iteration_limit": 0,

        "index": 2,
    }

    for i, dict_params_from_json in enumerate(list_dict_params_from_json):

        try:
            if dict_params_from_json['is_live'] == 0:
                list_df_mc_params.append(None)
                continue
        except KeyError:
            warnings.warn("Key 'is_line' does not exit. Proceeded by default.")

        df_mc_params = mc_inputs_generator(index=True, **dict_params_from_json)

        for k_, v_ in dict_param_names.items():
            if v_ != 0:
                continue
            if k_ not in dict_params_from_json:
                # if this happens, either take a look of dict_param_names or the *.csv file.
                raise ValueError('variable {} not found in *.csv input file.'.format(k_))

            df_mc_params[k_] = [dict_params_from_json[k_]] * len(df_mc_params)

        df_mc_params['fire_iso834_time'] = [iso834_time] * len(df_mc_params.index)
        df_mc_params['fire_iso834_temperature'] = [iso834_temperature] * len(df_mc_params.index)
        df_mc_params['beam_c'] = [beam_c] * len(df_mc_params)
        df_mc_params['index'] = df_mc_params.index

        df_mc_params.index.name = ['index']

        list_df_mc_params.append(df_mc_params)

        if save_csv:
            fn_ = '{}_in.csv'.format(os.path.basename(list_path_json[i]).replace('.json', ''))
            pf_ = os.path.join(os.path.dirname(list_path_json[i]), fn_)

            try:
                df_mc_params.to_csv(pf_)
            except PermissionError:
                warnings.warn('WARNING! File save failed: {}'.format(pf_))

    return list_df_mc_params


def post_processing_plot_figures(path_work, dict_config_params):
    # Requirements
    # list_path_out_csv: a list of numeral output file path in *.csv format

    # FIGURE FORMAT
    line_styles = OrderedDict(
        [
            ('solid', (0, ())),
            # ('loosely dotted', (0, (1, 10))),
            # ('dotted', (0, (1, 5))),
            # ('densely dotted', (0, (1, 1))),

            # ('loosely dashed', (0, (5, 10))),
            # ('dashed', (0, (5, 5))),
            # ('densely dashed', (0, (5, 1))),

            # ('loosely dashdotted', (0, (3, 10, 1, 10))),
            # ('dashdotted', (0, (3, 5, 1, 5))),
            # ('densely dashdotted', (0, (3, 1, 1, 1))),

            # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            # ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
            # ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
        ]
    )

    cycle_line_styles = cycle([v for v in line_styles.values()])

    list_path_out_csv_all = []
    for f in os.listdir(os.path.join(path_work, 'temp')):
        if f.endswith("_out.csv"):
            list_path_out_csv_all.append(os.path.join(path_work, 'temp', f))

    # Plot individual t_eq
    for path_out_csv in list_path_out_csv_all:
        plot_figures(
            list_path_out_csv=[path_out_csv],
            plot_teq_xlim=dict_config_params['plot_teq_xlim'],
            reliability_target=dict_config_params['reliability_target'],
            path_work=os.path.join(path_work, 'temp'),
            figure_name=os.path.basename(path_out_csv).replace('_out.csv', '.png'),
            cycle_linestyles=cycle_line_styles
        )

    # Plot individual t_eq into one figure
    plot_figures(
        list_path_out_csv=list_path_out_csv_all,
        plot_teq_xlim=dict_config_params['plot_teq_xlim'],
        reliability_target=dict_config_params['reliability_target'],
        path_work=path_work,
        figure_name='t_eq.png',
        plot_figuresize=dict_config_params['plot_figuresize'],
        cycle_linestyles=cycle_line_styles
    )

    # Plot combined t_eq with all results merged into one curve
    if len(list_path_out_csv_all) >= 0:
        list_pd_out_csv = []

        for path_out_csv in list_path_out_csv_all:
            df_ = pd.read_csv(path_out_csv, index_col='INDEX', dtype=np.float64)

            path_temp = os.path.dirname(path_out_csv)
            job_id = os.path.basename(path_out_csv).replace("_out.csv", "")

            with open(os.path.join(path_temp, "{}.json".format(job_id))) as f:
                dict_input = json.load(f)

            print(os.path.join(path_temp, "{}.json".format(job_id)), " - ", "probability_weight" in dict_input)

            if "probability_weight" in dict_input:  # apply probability weight if available

                df_["PROBABILITY"] = dict_input['probability_weight'] * (1 / len(df_.index))
                list_pd_out_csv.append(df_)

            # list_pd_out_csv.append(df_)

        pd_out_all = pd.concat(list_pd_out_csv, axis=0, ignore_index=True, sort=False)
        pd_out_all.sort_values("TIME EQUIVALENCE [s]", inplace=True)
        path_out_csv_merged = os.path.join(path_work, 'temp', '{}_mergedout.csv'.format(os.path.basename(path_work)))
        pd_out_all.to_csv(path_out_csv_merged)

        plot_figures(
            list_path_out_csv=[path_out_csv_merged],
            plot_teq_xlim=dict_config_params['plot_teq_xlim'],
            reliability_target=dict_config_params['reliability_target'],
            path_work=path_work,
            figure_name='t_eq_merged.png',
            apply_probability=True,
            figure_show_legend=False,
            plot_figuresize=dict_config_params['plot_figuresize'],
            cycle_linestyles=cycle_line_styles
        )


def individual_teq_to_csv(path_work):
    list_path_out_csv = []
    for f in os.listdir(os.path.join(path_work, 'temp')):
        if f.endswith("_out.csv"):
            list_path_out_csv.append(os.path.join(path_work, 'temp', f))

    for path_out_csv in list_path_out_csv:
        df_ = pd.read_csv(path_out_csv, index_col='INDEX', dtype=np.float16)
        name_ = os.path.basename(path_out_csv).replace("_out.csv", "")


def run():
    # ==================================================================================================================
    # Parse inputs from files into list of dict()s
    # ==================================================================================================================

    print('preparing recipes...')

    path_master_input_csv = select_path_input_csv()

    path_work, list_path_input_json = csv_to_jsons(path_master_input_csv)

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
            'n_proc': 2,
            'plot_teq_xlim': 180,
            'reliability_target': 0.8,
            'output_fires': 0,
            'plot_style': 'seaborn-paper',
            'plot_figuresize': (3.94, 2.76),
        }
        dict_config_params = dict_config_params_default

    try:
        plt.style.use(dict_config_params['plot_style'])
    except (KeyError, OSError) as e:
        plt.style.use('seaborn-paper')

    list_input_file_names = [os.path.basename(i).split('.')[0] for i in list_path_input_json]

    # ==================================================================================================================
    # Plot all figures if no simulations to run
    # ==================================================================================================================

    count_live_jobs = 0

    for i in list_path_input_json:
        with open(i, 'r') as f:
            try:
                count_live_jobs += int(json.load(f)['is_live'])
            except KeyError:
                pass

    if count_live_jobs == 0:
        print('seems we are getting a take away for dinner today, food is getting ready...')
        post_processing_plot_figures(path_work, dict_config_params)
        return 0

    # ==================================================================================================================
    # Spawn Monte Carlo parameters from dict() to DataFrame()s
    # ==================================================================================================================

    print('warming up oven and stoves...')

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

    print('cooking in progress...')

    list_path_out_csv = []

    for i, list_dict_mc_params in enumerate(list_list_dict_mc_params):

        if list_dict_mc_params is None:
            continue

        print("CASE:", list_input_file_names[i])

        if dict_config_params['n_proc'] == 1:
            results = []

            for dict_mc_params in list_dict_mc_params:
                results.append(calc_time_equivalence(**dict_mc_params))

        else:
            time_simulation_start = time.perf_counter()
            m = mp.Manager()
            q = m.Queue()
            p = mp.Pool(int(dict_config_params['n_proc']), maxtasksperchild=1000)
            jobs = p.map_async(calc_time_equivalence_worker, [(dict_, q) for dict_ in list_dict_mc_params])
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
        df_output = pd.DataFrame({
            "TEMPERATURE": results[0]
        })
        df_output.set_index("INDEX", inplace=True)  # assign 'INDEX' column as DataFrame index
        df_output.sort_values('TIME EQUIVALENCE [s]', inplace=True)  # sort base on time equivalence
        df_output.to_csv(list_path_out_csv[-1])

        # export fires
        if 'output_fires' in dict_config_params:
            if dict_config_params["output_fires"] == 0:
                continue

    # ==================================================================================================================
    # Postprocessing - plot figures
    # ==================================================================================================================

    print('getting food onto the table...')

    post_processing_plot_figures(path_work, dict_config_params)

    input('dinner is ready (press any key to continue)...')


def run2():

    print('preparing recipes...')

    path_master_input_csv = select_path_input_csv()

    with open(os.path.join(os.path.dirname(path_master_input_csv), 'config.json'), 'r') as f:
        dict_config = json.load(f)

    dict_params_additional = {}
    dict_params_additional['n_threads'] = dict_config['n_threads']

    path_work, list_path_input_json = csv_to_jsons(path_master_input_csv, dict_params_additional)

    list_mc_case = []
    for i in list_path_input_json:
        with open(i, 'r') as f:
            input_param = json.load(f)

        MCC = MonteCarloCase(
                path_wd=os.path.dirname(i),
                name=os.path.basename(i).replace('.json', '')
        )

        MCC.input_param = input_param

        MCC.generate_mc_param()

        list_mc_case.append(MCC)

    for i in list_mc_case:
        i.run_monte_carlo_simulation()
        i.mc_results_to_csv()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    run2()
