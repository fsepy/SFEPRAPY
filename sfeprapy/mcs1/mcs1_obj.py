import json
import os
import types
import warnings
from tkinter import filedialog, Tk, StringVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sfeprapy.func.fire_iso834 import fire as _fire_standard
from sfeprapy.mcs1.mcs1_func_gen import mc_inputs_generator
from sfeprapy.mcs1.mcs1_func_main import main as func_main


class MonteCarloCase:
    # this dictionary object defines all user input variable names as key, and corresponding values:
    # 0 - user input, static;
    # 1 - user input, stochastic; and
    # 2 - derived input
    PARAM = dict(
        window_height=0,
        window_width=0,
        window_open_fraction=1,
        room_breadth=0,
        room_depth=0,
        room_height=0,
        room_wall_thermal_inertia=0,
        fire_mode=0,
        fire_time_step=0,
        fire_iso834_time=2,
        fire_iso834_temperature=2,
        fire_tlim=0,
        fire_load_density=1,
        fire_hrr_density_ubound=0,
        fire_hrr_density_lbound=0,
        fire_hrr_density_mean=0,
        fire_hrr_density_std=0,
        fire_spread_speed=1,
        fire_nft_ubound=1,
        fire_duration=0,
        fire_t_alpha=0,
        fire_gamma_fi_q=0,
        beam_position=1,
        beam_rho=0,
        beam_c=2,
        beam_cross_section_area=0,
        beam_loc_z=0,
        protection_k=0,
        protection_rho=0,
        protection_c=0,
        protection_protected_perimeter=0,
        solver_temperature_target=0,
        solver_fire_duration=0,
        solver_thickness_lbound=0,
        solver_thickness_ubound=0,
        solver_tolerance=0,
        solver_iteration_limit=0,
        index=2,
    )

    def __init__(
        self,
        path_wd: str,
        name: str,
        func: (types.FunctionType, types.BuiltinFunctionType),
    ):
        self._path_wd = path_wd
        self._name = name
        self._func = func

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

        # user preferences
        self._n_simulations = 1000
        self._n_threads = 2

        self._func_gen = None
        self._func_mcs = None

        # generated data
        self._status = 0
        self._mc_param = None
        self._mc_results = None

        self.timer_mc = None

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
        return self._mc_param.to_dict("index")

    @property
    def n_threads(self):
        return self._n_threads

    @property
    def n_simulations(self):
        return int(self._n_simulations)

    @property
    def mc_results(self):
        return self._mc_results

    @is_live.setter
    def is_live(self, val: bool):
        self._is_live = val

    @input_param.setter
    def input_param(self, val: dict):
        if not isinstance(val, dict):
            raise TypeError(
                "Input parameters variable (input_param) should be a dict object."
            )
        for key, val_ in self.PARAM.items():
            if val_ == 0:
                if key not in val:
                    raise ValueError("Could not find {} in input_param.".format(key))

        self.n_threads = val["n_threads"]
        self.n_simulations = val["n_simulations"]
        self._input_param = val

    @mc_param.setter
    def mc_param(self, val: pd.DataFrame):
        if not isinstance(self, pd.DataFrame):
            raise TypeError(
                "MC parameters variable (mc_param) should be a DataFrame object."
            )
        if not len(val.index) == self.n_simulations:
            raise ValueError(
                "Length mismatch, len(mc_param)={} != len(n_simulations)={}".format(
                    len(val.index), self.n_simulations
                )
            )

        self._mc_param = val

    @mc_results.setter
    def mc_results(self, val: pd.DataFrame):
        if not isinstance(val, pd.DataFrame):
            raise TypeError(
                "MC results variable (mc_results) should be a DataFrame object."
            )
        self._mc_results = val

    @n_threads.setter
    def n_threads(self, val: int):
        if val is None:
            val = 1
        if val < 1:
            warnings.warn(
                "Number of threads can only be a positive integer, your input value is {} and this is changed"
                " to 1.".format(val)
            )
            val = 1
        elif val != int(val):
            warnings.warn(
                "Number of threads can only be a positive integer, your input value is {} and this is changed"
                " to {}.".format(val, int(val))
            )

        self._n_threads = int(val)

    @n_simulations.setter
    def n_simulations(self, val: int):

        if val is None:
            val = int(10)
        if val < 1:
            warnings.warn(
                "Number of threads can only be a positive integer, your input value is {} and this is changed"
                " to 1.".format(val)
            )
            val = int(1)
        elif val != int(val):
            warnings.warn(
                "Number of threads can only be a positive integer, your input value is {} and this is changed"
                " to {}.".format(val, int(val))
            )
            val = int(val)

        self._n_simulations = val

    def in2sc(self):
        # User input to stochastic randomised parameters

        if self.input_param is None:
            raise ValueError("input_param is not defined.")

        input_param = self.input_param

        self.n_simulations = input_param["n_simulations"]
        self.n_threads = input_param["n_threads"]

        # finalise and save to *_in.csv
        fire_time = np.arange(
            0,
            input_param["fire_time_duration"] + input_param["fire_time_step"],
            input_param["fire_time_step"],
        )
        iso834_time = np.arange(0, 6 * 60 * 60, 30)
        iso834_temperature = _fire_standard(iso834_time, 273.15 + 20)

        dict_input_param = self.input_param
        df_mc_params = mc_inputs_generator(**dict_input_param)

        for key, val in self.PARAM.items():
            if val == 0:
                df_mc_params[key] = np.full(self.n_simulations, self.input_param[key])

        df_mc_params["fire_iso834_time"] = [iso834_time] * self.n_simulations
        df_mc_params["fire_iso834_temperature"] = [
            iso834_temperature
        ] * self.n_simulations
        df_mc_params["fire_time"] = [fire_time] * self.n_simulations
        df_mc_params["index"] = np.arange(0, self.n_simulations, 1)

        df_mc_params.index.name = [
            "index"
        ]  # todo is this correct? shouldn't it be a str?

        self._mc_param = df_mc_params

    def mc_sim(self):

        if self.input_param is None:
            raise ValueError("input_param is not defined")
        if self.mc_param is None:
            raise ValueError("mc_param is not defined.")

        if self._is_live:
            df_output = self._func(self._mc_param)
        elif os.path.exists(os.path.join(self._path_wd, self._name + ".csv")):
            # df_output = pd.DataFrame.from_csv(os.path.join(self._path_wd, self._name + '.csv'))
            df_output = pd.read_csv(os.path.join(self._path_wd, self.name + ".csv"))
        else:
            df_output = None

        self._mc_results = df_output

    def res2csv(self, path=None):

        if self.mc_results is None:
            warnings.warn("No results to save.")
            return -1

        if path is None:
            path = os.path.join(self._path_wd, self._name + ".csv")

        name_suffix = 0
        while True:
            try:
                if name_suffix == 0:
                    self.mc_results.to_csv(path)
                    break
                else:
                    self.mc_results.to_csv(path + str(int(name_suffix)))
                    break
            except PermissionError:
                name_suffix += 1

    def res2cdf(self, name: str, cdf_loc: (list, np.ndarray)):
        if self.mc_results is None:
            warnings.warn("No results are extracted.")
            return -1

        x = self.mc_results[name].values
        x = np.sort(x)
        x = np.insert(x, 0, 0)
        y = np.linspace(0, 1, len(x))
        func = interp1d(y, x)

        return func(cdf_loc)

    def mc_results_get(self, name: str):
        return self.mc_results[name].values


class MonteCarlo:
    DEFAULT_TEMP_NAME = "temp"
    DEFAULT_CONFIG_FILE_NAME = "config.json"
    DEFAULT_FIGURE_NAME_TEQ_COMBINED = "teq_combined.png"
    DEFAULT_CONFIG = dict(
        n_threads=1,
        reliability_target=0.8,
        output_fires=0,
        plot_teq_xlim=180,
        plot_style="seaborn-paper",
        plot_figuresize=[3.94, 2.76],
    )

    def __init__(self):
        self._path_master_csv = None

        self._dict_input_param = dict()
        self._dict_input_wd = dict()
        self._dict_input_case_name = dict()

        self._dict_config = dict()

        self._monte_carlo_cases = list()

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
            raise FileNotFoundError(
                "file {path_master_csv} not found.".format(path_master_csv=val)
            )
        self._path_master_csv = val

    @config.setter
    def config(self, val: dict):
        if not isinstance(val, dict):
            raise TypeError("config should be a dict object.")
        self._dict_config = val

    @monte_carlo_cases.setter
    def monte_carlo_cases(self, val: MonteCarloCase):
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

    def select_input_file(self, path_master_csv: str = None):

        if path_master_csv is None:
            self.path_master_csv = self._gui_file_path()
        else:
            self.path_master_csv = os.path.realpath(
                path_master_csv.replace('"', "").replace("'", "")
            )

        path_config = os.path.join(
            os.path.dirname(self.path_master_csv), self.DEFAULT_CONFIG_FILE_NAME
        )

        if os.path.isfile(path_config):
            with open(path_config, "r") as f:
                self.config = json.load(f)
        else:
            warnings.warn("config.json not found, default config parameters are used.")
            self.config = self.DEFAULT_CONFIG

    def make_input_param(self):
        if self.path_master_csv is None:
            raise ValueError("csv file path (path_master_csv) not defined.")
        if self.config is None:
            raise ValueError("config is not defined.")

        path_input_file_csv = self.path_master_csv
        config = self.config

        # csv to dict
        if path_input_file_csv.endswith(".csv"):
            df_input_params = pd.read_csv(path_input_file_csv).set_index("PARAMETERS")
        elif path_input_file_csv.endswith(".xlsx"):
            df_input_params = pd.read_excel(path_input_file_csv).set_index("PARAMETERS")
        else:
            df_input_params = []

        df_input_params = df_input_params.append(
            pd.Series(
                {k: config["n_threads"] for k in df_input_params.columns.values},
                name="n_threads",
            )
        )

        dict_dict_input_params = df_input_params.to_dict()

        # dict to json (list)
        for case_name, input_param in dict_dict_input_params.items():
            MCC = MonteCarloCase(
                path_wd=os.path.join(
                    os.path.dirname(path_input_file_csv), self.DEFAULT_TEMP_NAME
                ),
                name=case_name,
                func=func_main,
            )
            MCC.input_param = input_param
            MCC.is_live = int(input_param["is_live"]) == 1
            MCC.in2sc()

            self.monte_carlo_cases = MCC

    def make_mc_params(self):
        if self.monte_carlo_cases is None:
            raise ValueError("no monte carlo case has been set up.")

        for case in self.monte_carlo_cases:
            case.in2sc()

    def run_mc(self):

        for case in self.monte_carlo_cases:
            case.mc_sim()
            case.res2csv()

    def out_combined_ky(self):

        dict_res = dict()

        for case in self.monte_carlo_cases:
            if case.is_live:
                dict_res[case.name] = case.mc_results_get("strength_reduction_factor")

        p = os.path.join(os.path.dirname(self.path_master_csv), "ky.csv")
        df_ = pd.DataFrame.from_dict(dict_res)
        df_.to_csv(p)

    def out_combined_T(self):

        dict_res = dict()

        for case in self.monte_carlo_cases:
            if case.is_live:
                dict_res[case.name] = case.mc_results_get(
                    "solver_steel_temperature_solved"
                )

        p = os.path.join(os.path.dirname(self.path_master_csv), "T.csv")
        df_ = pd.DataFrame.from_dict(dict_res)
        df_.to_csv(p)

    def _plot_teq(self):
        plot_data = {}
        for case in self.monte_carlo_cases:
            plot_data[case.name] = dict(
                x=case.mc_results_get(""), y=np.linspace(0, 1, case.n_simulations)
            )

        self._plot_figure(
            path_save_figure=os.path.join(
                os.path.dirname(self.path_master_csv),
                self.DEFAULT_FIGURE_NAME_TEQ_COMBINED,
            ),
            data=plot_data,
            plot_xlim=(0, 1),
        )

    def _get_results_teq(self):

        for case in self.monte_carlo_cases:
            case.mc_results_get("solver_steel_temperature_solved")

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
            teq_ax.plot(plot_data["x"], plot_data["y"], label=case_name, linewidth=1)
            teq_ax.set_ylabel("y-axis label")
            teq_ax.set_xlabel("x-axis label")
            teq_ax.set_xticks(ticks=np.arange(0, plot_xlim + 0.001, 30))

        teq_ax.legend().set_visible(True)
        teq_ax.legend(prop={"size": 7})
        plt.tight_layout()
        plt.savefig(path_save_figure, transparent=True, bbox_inches="tight", dpi=300)

        plt.clf()
        plt.close()

    @staticmethod
    def _gui_file_path():
        # get a list of dict()s representing different scenarios
        root = Tk()
        root.withdraw()
        folder_path = StringVar()

        path_input_file_csv = filedialog.askopenfile(
            title="Select Input File",
            parent=root,
            filetypes=[("SPREADSHEET", [".csv", ".xlsx"])],
            mode="r",
        )
        folder_path.set(path_input_file_csv)
        root.update()

        try:
            path_input_file_csv = os.path.realpath(path_input_file_csv.name)
            return path_input_file_csv
        except AttributeError:
            raise FileNotFoundError("file not found.")


if __name__ == "__main__":
    MC = MonteCarlo()
    MC.select_input_file()
    MC.make_input_param()
    MC.make_mc_params()
    MC.run_mc()
