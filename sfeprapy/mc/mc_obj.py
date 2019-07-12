import os
import types
import warnings

import numpy as np
import pandas as pd
import stats
from scipy.interpolate import interp1d

from sfeprapy.mc1.mc1_func_gen import mc_inputs_generator


class MonteCarloSimulation:

    def __init__(
            self,
            path_wd: str,
            name: str,
            func_mcs: (types.FunctionType, types.BuiltinFunctionType),
            func_gen: (types.FunctionType, types.BuiltinFunctionType),
            n_simulations: int = 1000,
            n_threads: int = 2
    ):
        # ==============================================================================================================
        # REQUIRED PARAMETERS
        # ==============================================================================================================
        self._path_wd = path_wd
        self._name = name

        # ==============================================================================================================
        # REQUIRED PARAMETERS 2
        # ==============================================================================================================
        self.n_simulations = n_simulations
        self.n_threads = n_threads
        self._is_live = False
        self._flag_output_numeric_results = False
        self._flag_output_numeric_fires = False
        self._flag_output_plot_teq = False
        self._flag_output_plot_dist = False
        self._func_gen = func_gen  # random inputs processing function
        self._func_mcs = func_mcs  # monte carlo simulation (main_args) procedure

        # ==============================================================================================================
        # PLACEHOLDERS
        # ==============================================================================================================
        self._input_param = None
        self._mc_param = None
        self._mc_results = None

        self.timer_mc = None

        # ==============================================================================================================
        # MISC
        # ==============================================================================================================
        try:
            os.mkdir(path_wd)  # make work directory
        except FileExistsError:
            pass

    @property
    def is_live(self):
        return self._is_live

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
        return self._mc_param.to_dict('index')

    @property
    def n_threads(self):
        return self._n_threads

    @property
    def n_simulations(self):
        return int(self._n_simulations)

    @is_live.setter
    def is_live(self, val: bool):
        self._is_live = val

    @input_param.setter
    def input_param(self, val: dict):
        if not isinstance(val, dict):
            raise TypeError('Input parameters variable (input_param) should be a dict object.')

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

        self._n_threads = int(val)

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

    def in2sc(self, **additional_kws):
        # User input to stochastic randomised parameters

        if self.input_param is None:
            raise ValueError('input_param is not defined.')

        input_param: dict = self.input_param

        # finalise and save to *_in.csv

        dict_input_param = self.input_param
        df_mc_params = mc_inputs_generator(**dict_input_param)

        for k, v in input_param.items():
            df_mc_params[k] = [v] * self.n_simulations
        for k, v in additional_kws.items():
            df_mc_params[k] = [v] * self.n_simulations

        self._mc_param = df_mc_params

    def mcs(self):

        if self.input_param is None:
            raise ValueError('input_param is not defined')
        if self.mc_param is None:
            raise ValueError('mc_param is not defined.')

        if self._is_live:
            df_output = self._func_mcs(self._mc_param)
        elif os.path.exists(os.path.join(self._path_wd, self._name + '.csv')):
            # df_output = pd.DataFrame.from_csv(os.path.join(self._path_wd, self._name + '.csv'))
            df_output = pd.read_csv(os.path.join(self._path_wd, self.name + '.csv'))
        else:
            df_output = None

        self._mc_results = df_output

    def res2csv(self, path=None):

        if self._mc_results is None:
            warnings.warn('No results to save.')
            return -1

        if path is None:
            path = os.path.join(self._path_wd, self._name + '.csv')

        name_suffix = 0
        while True:
            try:
                if name_suffix == 0:
                    self._mc_results.to_csv(path)
                    break
                else:
                    self._mc_results.to_csv(path + str(int(name_suffix)))
                    break
            except PermissionError:
                name_suffix += 1

    def res2cdf(self, name: str, cdf_loc: (list, np.ndarray)):
        if self._mc_results is None:
            warnings.warn('No results are extracted.')
            return -1

        x = self._mc_results[name].values
        x = np.sort(x)
        x = np.insert(x, 0, 0)
        y = np.linspace(0, 1, len(x))
        func = interp1d(y, x)

        return func(cdf_loc)

    def mc_results_get(self, name: str):
        return self._mc_results[name].values


def d(dist_type: str, lbound: float, ubound: float, n: int = 100, shuffle: bool = False, **dist_kws):
    """

    :param dist_type: distribution type, can be any distribution types in scipy.stats or customised
    distribution 'custom_qfd_car_park', 'custom_window_breakage'.
    :param lbound: lower limit of the sampled values.
    :param ubound: upper limit of the sampled values.
    :param n: number of samples to be produced.
    :param shuffle: True or False, shuffle samples or not.
    :param dist_kws:
    :return:
    """

    if dist_type == 'custom_qfd_car_park':
        samples = 0
    elif dist_type == 'custom_window_breakage':
        samples = 0
    else:
        cdf_q = np.linspace(
            getattr(stats, dist_type).cdf(x=lbound, **dist_kws),
            getattr(stats, dist_type).cdf(x=ubound, **dist_kws),
            n
        )

        samples = getattr(stats, dist_type).ppf(q=cdf_q, **dist_kws)
        samples[samples == np.inf] = ubound
        samples[samples == -np.inf] = lbound

    if shuffle:
        np.random.shuffle(samples)

    return samples


if __name__ == '__main__':

    # fire_time = np.arange(0, 10800+5, 5)
    # iso834_temperature = _fire_standard(fire_time, 273.15 + 20)
    # df_mc_params['fire_iso834_time'] = [iso834_time] * self.n_simulations
    # df_mc_params['fire_iso834_temperature'] = [iso834_temperature] * self.n_simulations
    # df_mc_params['fire_time'] = [fire_time] * self.n_simulations
    # df_mc_params['index'] = np.arange(0, self.n_simulations, 1)

    input_param = dict(
        is_live=1,
        probability_weight=0.2,
        fire_mode=3,
        simulations=1000,
        time_step=30,
        time_duration=18000,
        fire_hrr_density=dict(
            dist_type='norm',
            lbound=0.249,
            ubound=0.251,
            mean=0.25,
            std=100),
        fire_qfd=dict(
            ubound=1500,
            lbound=100,
            mean=420,
            std=126),
        fire_spread=dict(
            dist_type='uniform',
            lbound=0.0035,
            ubound=0.0190),
        fire_com_eff=dict(
            dist_type='uniform',
            lbound=0.8,
            ubound=0.9999),
        fire_nft=dict(
            dist_type='norm',
            mean=1050,
            std=1.5),
        beam_loc_length_ratio=dict(
            lbound=0.6,
            ubound=0.9),
        fire_tlim=0.333,
        fire_t_alpha=300,
        fire_gamma_fi_q=1,
        room_breadth=15.8,
        room_depth=31.6,
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
        beam_loc_z=2.8
    )

    pref_param = dict(
        path_wd=os.path.abspath(__file__),
        name='Test Case A',
        func_mcs=None,
        func_gen=None,
        n_simulations=1000,
        n_threads=2
    )

    # MC = MonteCarlo()
    # MC.select_input_file()
    # MC.make_input_param()
    # MC.make_mc_params()
    # MC.run_mc()
