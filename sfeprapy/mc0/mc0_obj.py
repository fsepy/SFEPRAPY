import multiprocessing as mp
import os
import time
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import warnings
from tkinter import filedialog, Tk, StringVar
import scipy.stats as stats
from tqdm import tqdm

from sfeprapy.func.fire_iso834 import fire as _fire_standard


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
        fire_time = np.arange(0, input_param['fire_time_duration'] + input_param['fire_time_step'], input_param['fire_time_step'])
        iso834_time = np.arange(0, 6 * 60 * 60, input_param['fire_time_step'])
        iso834_temperature = _fire_standard(iso834_time, 273.15 + 20)

        df_mc_params = mc_inputs_generator(**self.input_param)

        for key, val in self.PARAM.items():
            if val == 0:
                df_mc_params[key] = np.full(self.n_simulations, self.input_param[key])

        df_mc_params['fire_iso834_time'] = [iso834_time] * self.n_simulations
        df_mc_params['fire_iso834_temperature'] = [iso834_temperature] * self.n_simulations
        df_mc_params['fire_time'] = [fire_time] * self.n_simulations
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
                results.append(grouped_a_b(**dict_mc_params))

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

    @staticmethod
    def _dist_gen_uniform(n_rv: int, lim1: float, lim2: float):

        if lim2 > lim1:
            lim1 += lim2
            lim2 = lim1 - lim2
            lim1 -= lim2

        sampled = np.linspace(lim1, lim2, n_rv + 1, dtype=float)
        sampled += (sampled[1] - sampled[0]) * 0.5
        sampled = sampled[0:-1]

        np.random.shuffle(sampled)
        return sampled

    @staticmethod
    def _dist_gen_gumbel_r(n_rv: int, miu: float, sigma: float, lim1: float, lim2: float):
        if lim1 < lim2:
            lim1 += lim2
            lim2 = lim1 - lim2
            lim1 -= lim2

        # parameters Gumbel W&S
        alpha = 1.282 / sigma
        u = miu - 0.5772 / alpha

        # parameters Gumbel scipy
        scale = 1 / alpha
        loc = u

        # Generate a linear spaced array inline with lower and upper boundary of log normal cumulative probability
        # density.
        sampled_cfd = np.linspace(
            stats.gumbel_r.cdf(x=lim2, loc=loc, scale=scale),
            stats.gumbel_r.cdf(x=lim1, loc=loc, scale=scale),
            n_rv
        )

        # Sample log normal distribution
        sampled = stats.gumbel_r.ppf(q=sampled_cfd, loc=loc, scale=scale)

        np.random.shuffle(sampled)
        return sampled

    @staticmethod
    def _dist_gen_lognorm_inv_mod(n_rv: int, miu: float, sigma: float, lim1: float, lim2: float, reserved: float = 0):
        """

        :param n_rv:
        :param miu:
        :param sigma:
        :param lim1:
        :param lim2:
        :param reserved:
        :return:
        """
        if lim1 < lim2:
            lim1 += lim2
            lim2 = lim1 - lim2
            lim1 -= lim2

        cov = sigma / miu
        sigma_ln = np.sqrt(np.log(1 + cov ** 2))
        miu_ln = np.log(miu) - 1 / 2 * sigma_ln ** 2

        loc = 0
        scale = np.exp(miu_ln)

        # Generate lim1 linear spaced array inline with lower and upper boundary of log normal cumulative probability
        # density.
        sampled_cfd = np.linspace(
            stats.lognorm.cdf(x=lim2, s=sigma_ln, loc=loc, scale=scale),
            stats.lognorm.cdf(x=lim1, s=sigma_ln, loc=loc, scale=scale),
            n_rv
        )

        # Sample log normal distribution
        sampled = stats.lognorm.ppf(q=sampled_cfd, s=sigma_ln, loc=loc, scale=scale)

        sampled = 1 - sampled

        sampled = [i * (1 - reserved) + reserved for i in sampled]

        np.random.shuffle(sampled)
        return sampled

    @staticmethod
    def _dist_gen_norm(n_rv: int, miu: float, sigma: float, lim1: float, lim2: float):
        if lim1 < lim2:
            lim1 += lim2
            lim2 = lim1 - lim2
            lim1 -= lim2
        # scale = (1.939 - (np.log(mean) * 0.266)) * mean

        sampled_cfd = np.linspace(
            stats.norm.cdf(x=lim2, loc=miu, scale=sigma),
            stats.norm.cdf(x=lim1, loc=miu, scale=sigma),
            n_rv
        )

        sampled = stats.norm.ppf(q=sampled_cfd, loc=miu, scale=sigma)

        np.random.shuffle(sampled)
        return sampled


class MCS:
    DEFAULT_TEMP_FOLDER_NAME = 'temp'
    DEFAULT_CONFIG_FILE_NAME = 'config.json'
    DEFAULT_CONFIG = dict(n_threads=1, output_fires=0)

    def __init__(self):
        self._path_wd = None

        self._df_master_input = None
        self._dict_config = None

        self._func_mcs_gen = None
        self._func_mcs_calc = None
        self._func_mcs_calc_mp = None

        self._df_mcs_out = None

    @property
    def path_wd(self):
        return self._path_wd

    @property
    def input(self) -> pd.DataFrame:
        return self._df_master_input

    @property
    def config(self):
        return self._dict_config

    @property
    def func_mcs_gen(self):
        return self._func_mcs_gen

    @property
    def func_mcs_calc_mp(self):
        return self._func_mcs_calc_mp

    @property
    def func_mcs_calc(self):
        return self._func_mcs_calc

    @property
    def mcs_out(self) -> pd.DataFrame:
        return self._df_mcs_out

    @path_wd.setter
    def path_wd(self, p_):
        assert os.path.isdir(p_)
        self._path_wd = p_

    @input.setter
    def input(self, df: pd.DataFrame):
        df.set_index('PARAMETERS', inplace=True)
        self._df_master_input = df

    @config.setter
    def config(self, dict_config: dict):
        self._dict_config = dict_config

    @func_mcs_gen.setter
    def func_mcs_gen(self, func_mcs_gen):
        self._func_mcs_gen = func_mcs_gen

    @func_mcs_calc.setter
    def func_mcs_calc(self, func_mcs_calc):
        self._func_mcs_calc = func_mcs_calc

    @func_mcs_calc_mp.setter
    def func_mcs_calc_mp(self, func_mcs_calc_mp):
        self._func_mcs_calc_mp = func_mcs_calc_mp

    @mcs_out.setter
    def mcs_out(self, df_out: pd.DataFrame):
        self._df_mcs_out = df_out

    def define_problem(self, data: Union[str, pd.DataFrame, dict] = None, config: dict = None, path_wd: str = None):

        # to get problem definition: try to parse from csv/xls/xlsx

        if data is None:
            fp = os.path.realpath(self._get_file_path_gui())
            self.path_wd = os.path.dirname(fp)
            if fp.endswith('.xlsx') or fp.endswith('.xls'):
                self.input = pd.read_excel(fp)
            elif fp.endswith('.csv'):
                self.input = pd.read_csv(fp)
            else:
                raise ValueError('Unknown input file format.')
        elif isinstance(data, str):
            fp = data
            self.path_wd = os.path.dirname(fp)
            if fp.endswith('.xlsx') or fp.endswith('.xls'):
                self.input = pd.read_excel(fp)
            elif fp.endswith('.csv'):
                self.input = pd.read_csv(fp)
            else:
                raise ValueError('Unknown input file format.')
        elif isinstance(data, pd.DataFrame):
            self.input = data
        elif isinstance(data, dict):
            self.input = pd.DataFrame.from_dict(data)

        # to get configuration: try to parse from cwd if there is any, otherwise chose default values

        if self.path_wd is None:
            self.config = self.DEFAULT_CONFIG
        elif os.path.isfile(os.path.join(self.path_wd, self.DEFAULT_CONFIG_FILE_NAME)):
            with open(os.path.join(self.path_wd, self.DEFAULT_CONFIG_FILE_NAME), 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.DEFAULT_CONFIG

    def define_stochastic_parameter_generator(self, func):
        self.func_mcs_gen = func

    def define_calculation_routine(self, func, func_mp=None):
        self.func_mcs_calc = func
        self.func_mcs_calc_mp = func_mp

    def run_mcs(self):
        # Check whether required parameters are defined
        err_msg = list()
        if self._df_master_input is None:
            err_msg.append('Problem definition is not defined.')
        if self._func_mcs_calc is None:
            err_msg.append('Monte Carlo Simulation calculation routine is not defined.')
        if self._func_mcs_gen is None:
            err_msg.append('Monte Carlo Simulation stochastic parameter generator is not defined.')
        if len(err_msg) > 0:
            raise ValueError(r'\n'.join(err_msg))

        # Prepare mcs parameter inputs
        x1 = self.input.to_dict()
        for case_name in list(x1.keys()):
            for param_name in list(x1[case_name].keys()):
                if ':' in param_name:
                    param_name_parent, param_name_sibling = param_name.split(':')
                    if param_name_parent not in x1[case_name]:
                        x1[case_name][param_name_parent] = dict()
                    x1[case_name][param_name_parent][param_name_sibling] = x1[case_name].pop(param_name)
        # to convert all "string numbers" to float data type
        for case_name in x1.keys():
            for i, v in x1[case_name].items():
                if isinstance(v, dict):
                    for ii, vv in v.items():
                        try:
                            x1[case_name][i][ii] = float(vv)
                        except:
                            pass
                else:
                    try:
                        x1[case_name][i] = float(v)
                    except:
                        pass

        # Generate mcs parameter samples
        x2 = {k: self.func_mcs_gen(v, int(v['n_simulations'])) for k, v in x1.items()}

        # Run mcs simulation
        x3 = {k: self._mcs_mp(self.func_mcs_calc, self.func_mcs_calc_mp, x=v, n_threads=self.config['n_threads']) for k, v in x2.items()}

        self.mcs_out = pd.concat(x3)

        if self.path_wd:
            self.mcs_out.to_csv(os.path.join(self.path_wd, 'mcs_out.csv'), index=False)

    @staticmethod
    def _mcs_mp(func, func_mp, x: pd.DataFrame, n_threads: int):
        list_mcs_in = x.to_dict(orient='records')

        print('{:<24.24}: {}'.format("CASE", list_mcs_in[0]['case_name']))
        print('{:<24.24}: {}'.format("NO. OF THREADS", n_threads))
        print('{:<24.24}: {}'.format("NO. OF SIMULATIONS", len(x.index)))

        if n_threads == 1:
            mcs_out = list()
            for i in tqdm(list_mcs_in, ncols=60):
                mcs_out.append(func(**i))
        else:
            import multiprocessing as mp
            m, p = mp.Manager(), mp.Pool(n_threads, maxtasksperchild=1000)
            q = m.Queue()
            jobs = p.map_async(func_mp, [(dict_, q) for dict_ in list_mcs_in])
            n_simulations = len(list_mcs_in)
            with tqdm(total=n_simulations, ncols=60) as pbar:
                while True:
                    if jobs.ready():
                        if n_simulations > pbar.n:
                            pbar.update(n_simulations - pbar.n)
                        break
                    else:
                        if q.qsize() - pbar.n > 0:
                            pbar.update(q.qsize() - pbar.n)
                        time.sleep(1)
                p.close()
                p.join()
                mcs_out = jobs.get()

        df_mcs_out = pd.DataFrame(mcs_out)
        try:
            df_mcs_out.drop('fire_temperature', inplace=True, axis=1)
        except KeyError:
            pass
        df_mcs_out.sort_values('solver_time_equivalence_solved', inplace=True)  # sort base on time equivalence

        return df_mcs_out

    @staticmethod
    def _get_file_path_gui():
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


def test():

    # test gui version
    def test_gui():
        from sfeprapy.mc0.mc0_func_main_2 import teq_main as calc
        from sfeprapy.mc0.mc0_func_main_2 import teq_main_wrapper as calc_mp
        from sfeprapy.func.mcs_gen import main as gen
        mcs = MCS()
        mcs.define_problem()
        mcs.define_stochastic_parameter_generator(gen)
        mcs.define_calculation_routine(calc, calc_mp)
        mcs.run_mcs()

    # test non-gui version
    def test_arg_dict():
        import sfeprapy.mc0 as mc0
        from sfeprapy.mc0.mc0_func_main_2 import teq_main as calc
        from sfeprapy.mc0.mc0_func_main_2 import teq_main_wrapper as calc_mp
        from sfeprapy.func.mcs_gen import main as gen
        mcs = MCS()
        mcs.define_problem(data=mc0.EXAMPLE_INPUT_DICT, config=mc0.EXAMPLE_CONFIG_DICT)
        mcs.define_stochastic_parameter_generator(gen)
        mcs.define_calculation_routine(calc, calc_mp)
        mcs.run_mcs()

    # test_gui()
    test_arg_dict()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    test()
