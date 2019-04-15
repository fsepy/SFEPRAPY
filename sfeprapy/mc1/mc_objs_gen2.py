import multiprocessing as mp
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import warnings
from tkinter import filedialog, Tk, StringVar
from scipy.interpolate import interp1d

from sfeprapy.func.fire_iso834 import fire as _fire_standard
from sfeprapy.mc1.mc_simulation_func2 import calc_time_equivalence_worker, calc_time_equivalence
from sfeprapy.mc1.mc_inputs_generator import mc_inputs_generator


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
                 solver_fire_duration=0, solver_thickness_lbound=0, solver_thickness_ubound=0, solver_tol=0,
                 solver_iteration_limit=0, index=2)

    def __init__(self, path_wd, name):
        self._path_wd = path_wd
        self._name = name
        try:
            os.mkdir(path_wd)
        except:
            pass

        self._input_param = None

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
    def name(self):
        return self._name

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
    def input_param(self, val: dict):
        if not isinstance(val, dict):
            raise TypeError('Input parameters variable (input_param) should be a dict object.')
        for key, val_ in self.PARAM.items():
            if val_ != 0:
                continue
            if key not in val:
                raise ValueError('Could not find {} in input_param.'.format(key))

        self.n_threads = val['n_threads']
        self.n_simulations = val['n_simulations']
        self._input_param = val

    @mc_param.setter
    def mc_param(self, val: pd.DataFrame):
        if not isinstance(self, pd.DataFrame):
            raise TypeError('MC parameters variable (mc_param) should be a DataFrame object.')
        if not len(val.index) == self.n_simulations:
            raise ValueError('Length mismatch, len(mc_param)={} != len(n_simulations)={}'.format(
                len(val.index), self.n_simulations
            ))
        self._mc_param = val

    @mc_results.setter
    def mc_results(self, val: pd.DataFrame):
        if not isinstance(val, pd.DataFrame):
            raise TypeError('MC results variable (mc_results) should be a DataFrame object.')
        self._mc_results = val

    @n_threads.setter
    def n_threads(self, val: int):
        if val is None:
            val = 1
        if val < 1:
            warnings.warn("Number of threads can only be a positive integer, your input value is {} and this is changed"
                          " to 1.".format(val))
            val = 1
        elif val != int(val):
            warnings.warn("Number of threads can only be a positive integer, your input value is {} and this is changed"
                          " to {}.".format(val, int(val)))
            n_threads = int(val)
        self._n_threads = val

    @n_simulations.setter
    def n_simulations(self, val: int):
        if val is None:
            val = int(10)
        if val < 1:
            warnings.warn("Number of threads can only be a positive integer, your input value is {} and this is changed"
                          " to 1.".format(val))
            val = int(1)
        elif val != int(val):
            warnings.warn("Number of threads can only be a positive integer, your input value is {} and this is changed"
                          " to {}.".format(val, int(val)))
            val = int(val)
        self._n_simulations = val

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
            path = os.path.join(self._path_wd, self._name + '.csv')
        try:
            self.mc_results.to_csv(path)
        except PermissionError:
            self.mc_results.to_csv(path + '1')

    def mc_results_get(self, name):
        return self.mc_results[name].values


class MonteCarlo:
    DEFAULT_TEMP_NAME = 'temp'
    DEFAULT_CONFIG_FILE_NAME = 'config.json'
    DEFAULT_FIGURE_NAME_TEQ_COMBINED = 'teq_combined.png'
    DEFAULT_CONFIG = dict(n_threads=1, reliability_target=0.8, output_fires=0, plot_teq_xlim=180,
                          plot_style="seaborn-paper", plot_figuresize=[3.94, 2.76])

    def __init__(self):
        self._path_master_csv = None

        self._dict_input_param = dict
        self._dict_input_wd = dict
        self._dict_input_case_name = dict

        self._dict_config = dict

        self._monte_carlo_cases = None

    @property
    def path_master_csv(self):
        return self._path_master_csv

    @property
    def dict_input_param(self):
        return self._dict_input_param

    @property
    def dict_input_wd(self):
        return self._dict_input_wd

    @property
    def dict_input_case_name(self):
        return self._dict_input_case_name

    @property
    def config(self, key=None):
        if key is None:
            return self._dict_config
        else:
            try:
                return self._dict_config[key]
            except KeyError:
                return None

    @property
    def monte_carlo_cases(self):
        return self._monte_carlo_cases

    @path_master_csv.setter
    def path_master_csv(self, val: str):
        val = os.path.realpath(val)
        if not os.path.isfile(val):
            raise FileNotFoundError('file {path_master_csv} not found.'.format(path_master_csv=val))
        self._path_master_csv = val

    @config.setter
    def config(self, val: dict):
        if not isinstance(val, dict):
            raise TypeError('config should be a dict object.')
        self._dict_config = val

    @monte_carlo_cases.setter
    def monte_carlo_cases(self, val: object):
        if not isinstance(self._monte_carlo_cases, list):
            self._monte_carlo_cases = [val]
        else:
            flag_exist = False
            for i, mc_case in enumerate(self._monte_carlo_cases):
                if mc_case.name == val.name:
                    self._monte_carlo_cases[i] = val
                    flag_exist = True

            if not flag_exist:
                self._monte_carlo_cases.append(val)

    def select_input_file(self):
        self.path_master_csv = self._gui_file_path()

        path_config = os.path.join(os.path.dirname(self.path_master_csv), self.DEFAULT_CONFIG_FILE_NAME)

        if os.path.isfile(path_config):
            with open(path_config, 'r') as f:
                self.config = json.load(f)
        else:
            warnings.warn('config.json not found, default config parameters are used.')
            self.config = self.DEFAULT_CONFIG

    def make_input_param(self):
        if self.path_master_csv is None:
            raise ValueError('csv file path (path_master_csv) not defined.')
        if self.config is None:
            raise ValueError('config is not defined.')

        path_input_file_csv = self.path_master_csv
        config = self.config

        # csv to dict
        df_input_params = pd.read_csv(path_input_file_csv).set_index('PARAMETERS')
        df_input_params = df_input_params.append(
            pd.Series({k: config['n_threads'] for k in df_input_params.columns.values}, name='n_threads'),
            ignore_index=False)

        dict_dict_input_params = df_input_params.to_dict()

        # dict to json (list)
        for case_name, input_param in dict_dict_input_params.items():
            MCC = MonteCarloCase(
                path_wd=os.path.join(os.path.dirname(path_input_file_csv), self.DEFAULT_TEMP_NAME),
                name=case_name
            )
            MCC.input_param = input_param
            MCC.generate_mc_param()

            self.monte_carlo_cases = MCC

    def make_mc_params(self):
        if self.monte_carlo_cases is None:
            raise ValueError('no monte carlo case has been set up.')

        for case in self.monte_carlo_cases:
            case.generate_mc_param()

    def run_mc(self):

        for case in self.monte_carlo_cases:
            case.run_monte_carlo_simulation()
            case.mc_results_to_csv()

    def plot_teq(self):
        plot_data = {}
        for case in self.monte_carlo_cases:
            plot_data[case.name] = dict(
                x=case.mc_results_get(''),
                y=np.linspace(0, 1, case.n_simulations)
            )

        self._plot_figure(
            path_save_figure=os.path.join(os.path.dirname(self.path_master_csv), self.DEFAULT_FIGURE_NAME_TEQ_COMBINED),
            data=plot_data,
            plot_xlim=(0, 1)
        )


    def _get_results_teq(self):

        for case in self.monte_carlo_cases:
            case.mc_results_get('solver_steel_temperature_solved')

    @staticmethod
    def _plot_figure(
            path_save_figure: str,
            data: dict,
            plot_xlim: tuple,
            plot_figuresize=(3.94, 2.76),
    ):

        teq_fig, teq_ax = plt.subplots(figsize=plot_figuresize)
        teq_ax.set_xlim([0, plot_xlim])

        for case_name, plot_data in data.items():
            teq_ax.plot(plot_data['x'], plot_data['y'], label=case_name, linewidth=1)
            teq_ax.set_ylabel('y-axis label')
            teq_ax.set_xlabel('x-axis label')
            teq_ax.set_xticks(ticks=np.arange(0, plot_xlim + 0.001, 30))

        teq_ax.legend().set_visible(True)
        teq_ax.legend(prop={'size': 7})
        plt.tight_layout()
        plt.savefig(
            path_save_figure,
            transparent=True,
            bbox_inches='tight',
            dpi=300
        )

        plt.clf()
        plt.close()

    @staticmethod
    def _gui_file_path():
        # get a list of dict()s representing different scenarios
        root = Tk()
        root.withdraw()
        folder_path = StringVar()

        path_input_file_csv = filedialog.askopenfile(title='Select Input File', filetypes=[('csv', ['.csv'])])
        folder_path.set(path_input_file_csv)
        root.update()

        try:
            path_input_file_csv = os.path.realpath(path_input_file_csv.name)
            return path_input_file_csv
        except AttributeError:
            raise FileNotFoundError('file not found.')


if __name__ == '__main__':
    MC = MonteCarlo()
    MC.select_input_file()
    MC.make_input_param()
    MC.make_mc_params()
    MC.run_mc()
