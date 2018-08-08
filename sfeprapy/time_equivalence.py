import os, copy, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from pickle import load as pload
from pickle import dump as pdump
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("ticks", {'axes.grid': True,})

from sfeprapy.func.temperature_fires import standard_fire_iso834 as _fire_standard
from sfeprapy.func.temperature_fires import parametric_eurocode1 as _fire_param
from sfeprapy.func.tfm_alt import travelling_fire as _fire_travelling
from sfeprapy.time_equivalence_core import calc_time_equiv_worker, mc_inputs_generator

# try:
#     from sfeprapy.func.temperature_fires import standard_fire_iso834 as _fire_standard
#     from sfeprapy.func.temperature_fires import parametric_eurocode1 as _fire_param
#     from sfeprapy.func.tfm_alt import travelling_fire as _fire_travelling
#     from sfeprapy.time_equivalence_core import calc_time_equiv_worker, mc_inputs_generator
# except ModuleNotFoundError:
#     from .func.temperature_fires import standard_fire_iso834 as _fire_standard
#     from .func.temperature_fires import parametric_eurocode1 as _fire_param
#     from .func.tfm_alt import travelling_fire as _fire_travelling
#     from .time_equivalence_core import calc_time_equiv_worker, mc_inputs_generator

# OUTPUT STRING FORMAT
# ====================

__strformat_1_1 = "{:25}{}"
__strformat_1_1_1 = "{:25}{:3}{}"

# GLOBAL SETTINGS
# ===============

# Global variables
# -------

__id = ""
__dir_work = ""
__fn_output = "res.p"
__fn_output_numerical = "res.csv"
__fn_plot_te = "plot_teq.png"
__fn_plot_te_all = "plot_teq_all.png"

__fn_plot_temp = "plot_temp.png"

__fn_selected_plot = "selected.png"
__fn_selected_numerical = "selected.csv"


def init_global_variables(_id, dir_work):
    global __id, __dir_work
    __id = _id
    __dir_work = dir_work


def step0_parse_input_files(dir_work):
    """

    :param dir_work:
    :return:
    """

    list_files = []
    for f in os.listdir(dir_work):
        if f.endswith('.txt'):
            list_files.append(os.path.join(dir_work, f))

    list_input_files = []
    for f_ in list_files:
        with open(f_, "r") as f__:
            l = f__.readline()
        if l.find("# MC INPUT FILE") > -1:
            list_input_files.append(f_)

    return list_input_files


def step1_inputs_maker(path_input_file):
    """

    :param path_input_file:
    :return:
    """

    file_name = os.path.basename(path_input_file)
    dir_work = os.path.dirname(path_input_file)
    __id = file_name.split(".")[0]

    fire = _fire_standard(np.arange(0, 3*60*60, 1), 273.15+20)
    inputs_extra = {"iso834_time": fire[0],
                    "iso834_temperature": fire[1],}

    df_input, dict_pref = mc_inputs_generator(dict_extra_variables_to_add=inputs_extra, dir_file=path_input_file)

    return df_input, dict_pref


def step2_calc(df_input, dict_pref, progress_print_interval=5):
    plot_dist('test', df_input, '')
    # LOCAL SETTINGS
    # ==============

    # To limit memory usage when multiprocessing is employed, a maximum number of tasks is defined for a single process.
    # Therefore a process can not preserve data over this limit.

    mp_maxtasksperchild = 1000

    # Load kwargs

    dict_input_kwargs = df_input.to_dict(orient="index")
    list_kwargs = []
    for key, val in dict_input_kwargs.items():
        val["index"] = key
        list_kwargs.append(val)

    # Load settings

    dict_settings = dict_pref
    n_proc = dict_settings["n_proc"]

    # Check number of processes are to be used

    n_proc = os.cpu_count() if int(n_proc) < 1 else int(n_proc)

    # SIMULATION START

    print(__strformat_1_1.format("Input file:", __id))
    print(__strformat_1_1.format("Total simulations:", len(list_kwargs)))
    print(__strformat_1_1.format("Number of threads:", n_proc))

    time_count_simulation = time.perf_counter()
    m = mp.Manager()
    q = m.Queue()
    p = mp.Pool(n_proc, maxtasksperchild=mp_maxtasksperchild)
    jobs = p.map_async(calc_time_equiv_worker, [(kwargs, q) for kwargs in list_kwargs])
    count_total_simulations = len(list_kwargs)
    n_steps = 24  #
    while progress_print_interval:
        if jobs.ready():
            time_count_simulation = time.perf_counter() - time_count_simulation
            print("{}{} {:03.0f}% ({:.1f}s)".format('#'*round(n_steps), '-'*round(0), 100., time_count_simulation))
            break
        else:
            p_ = q.qsize() / count_total_simulations * n_steps
            print("{}{} {:03.1f}%".format('#'*int(round(p_)), '-'*int(n_steps-round(p_)), p_/n_steps*100), end='\r')
            time.sleep(1)
    p.close()
    p.join()
    results = jobs.get()

    # format outputs

    results = np.array(results, dtype=float)
    df_output = pd.DataFrame({"TIME EQUIVALENCE [min]": results[:, 0]/60.,
                               "SEEK STATUS [bool]": results[:, 1],
                               "WINDOW OPEN FRACTION [%]": results[:, 2],
                               "FIRE LOAD DENSITY [MJ/m2]": results[:, 3],
                               "FIRE SPREAD SPEED [m/s]": results[:, 4],
                               "BEAM POSITION [m]": results[:, 5],
                               "MAX. NEAR FIELD TEMPERATURE [C]": results[:, 6],
                               "FIRE TYPE [0:P., 1:T.]": results[:, 7],
                               "PEAK STEEL TEMPERATURE TO GOAL SEEK [C]": results[:, 8]-273.15,
                               "PROTECTION THICKNESS [m]": results[:, 9],
                               "SEEK ITERATIONS [-]": results[:, 10],
                               "PEAK STEEL TEMPERATURE TO FIXED PROTECTION [C]": np.sort(results[:, 11])-273.15,
                               "INDEX": results[:, 12]})
    df_output = df_output[["TIME EQUIVALENCE [min]", "PEAK STEEL TEMPERATURE TO GOAL SEEK [C]", "PROTECTION THICKNESS [m]", "SEEK STATUS [bool]", "SEEK ITERATIONS [-]", "WINDOW OPEN FRACTION [%]", "FIRE LOAD DENSITY [MJ/m2]", "FIRE SPREAD SPEED [m/s]", "BEAM POSITION [m]", "MAX. NEAR FIELD TEMPERATURE [C]", "FIRE TYPE [0:P., 1:T.]", "PEAK STEEL TEMPERATURE TO FIXED PROTECTION [C]", "INDEX"]]
    df_output.set_index("INDEX", inplace=True)

    path_results_file = os.path.join(__dir_work, "{} - {}".format(__id, __fn_output))
    pdump(df_output, open(path_results_file, "wb"))

    return df_output


def plot_dist(id_, df_input, headers):
    headers = ['window_open_fraction', 'fire_load_density', 'fire_spread_speed', 'beam_position', 'temperature_max_near_field']

    fig, ax = plt.subplots(figsize=(2.5, 2))

    for each_header in headers:
        x = df_input[each_header].values

        if each_header == 'temperature_max_near_field':
            x = x[x<1200]
        # if each_header == 'window_open_fraction':
        #     x = trunc_lognorm_cfd(0,1,10000,0.2,0,np.exp(0.2))

        sns.distplot(x, kde=False, rug=True, bins=50, ax=ax, norm_hist=True)
        ax.set(ylabel='', yticklabels=[])
        # ax.set_xticklabels([])
        # ax.set_y_label

        plt.savefig(os.path.join(__dir_work, '{} - dist - {}.png'.format(id_, each_header)), ppi=300)
        plt.cla()

    plt.clf()


def select_fires_teq(df_input, dict_pref, df_output):

    # INPUT ARGUMENTS VALIDATION
    # ==========================

    if dict_pref["select_fires_teq"] <= 0 or dict_pref["select_fires_teq_tol"] <= 0:
        return 0

    # Load settings

    percentile = dict_pref["select_fires_teq"]
    tolerance = dict_pref["select_fires_teq_tol"]

    df_output.sort_values(by=["TIME EQUIVALENCE [min]"], inplace=True)

    # Convert 'percentile_ubound' and 'percentile_lbound' to integers according to actual range i.e. 'index=1000'

    index_max = int(max(df_output.index.values))

    percentile_selected_bounds = np.array([percentile - abs(tolerance), percentile + abs(tolerance)])
    if percentile_selected_bounds[0] < 0: percentile_selected_bounds[0] = 0
    if percentile_selected_bounds[1] > 1: percentile_selected_bounds[1] = 1

    index_selected_bounds = np.round(percentile_selected_bounds * index_max, 0).astype(dtype=int)
    if index_selected_bounds[0] == index_selected_bounds[1]: return 0

    indices_selected = np.arange(index_selected_bounds[0], index_selected_bounds[1], 1, dtype=int)

    list_index_selected_fires = df_output.iloc[indices_selected].index.values

    # iterate through all selected fires, store time and temperature

    # plt = Scatter2D()
    dict_fires = {}
    list_fire_name = ["TEMPERATURE {} (INDEX {}) [C]".format(str(i), str(int(v))) for i, v in enumerate(list_index_selected_fires)]

    for i,v in enumerate(list_index_selected_fires):
        # get input arguments
        args = df_input.loc[i].to_dict()

        # get fire type
        fire_type = int(df_output.loc[i]["FIRE TYPE [0:P., 1:T.]"])

        if fire_type == 0:  # parametric fire
            w, l, h = args["room_breadth"], args["room_depth"], args["room_height"]
            inputs_parametric_fire = {
                "A_t": 2*(w*l+w*h+h*l),
                "A_f": w*l,
                "A_v": args["window_height"] * args["window_width"] * args["window_open_fraction"],
                "h_eq": args["window_height"],
                "q_fd": args["fire_load_density"] * 1e6,
                "lambda_": args["room_wall_thermal_inertia"] ** 2,  # thermal inertia is used instead of k rho c.
                "rho": 1,  # see comment for lambda_
                "c": 1,  # see comment for lambda_
                "t_lim": args["time_limiting"],
                "time_end": args["fire_duration"],
                "time_step": args["time_step"],
                "time_start": args["time_start"],
                # "time_padding": (0, 0),
                "temperature_initial": 20 + 273.15,
            }
            tsec, temps = _fire_param(**inputs_parametric_fire)
        elif fire_type == 1:  # travelling fire
            inputs_travelling_fire = {
                "fire_load_density_MJm2": args["fire_load_density"],
                "heat_release_rate_density_MWm2": args["fire_hrr_density"],
                "length_compartment_m": args["room_depth"],
                "width_compartment_m": args["room_breadth"],
                "fire_spread_rate_ms": args["fire_spread_speed"],
                "height_fuel_to_element_m": args["room_height"],
                "length_element_to_fire_origin_m": args["beam_position"],
                "time_start_s": args["time_start"],
                "time_end_s": args["fire_duration"],
                "time_interval_s": args["time_step"],
                "nft_max_C": args["temperature_max_near_field"],
                "win_width_m": args["window_width"],
                "win_height_m": args["window_height"],
                "open_fract": args["window_open_fraction"]
            }
            tsec, temps, hrr, r = _fire_travelling(**inputs_travelling_fire)
            temps += 273.15
        else:
            print("FIRE TYPE UNKOWN.")

        dict_fires[list_fire_name[i]] = temps - 273.15
        # plt.plot2(tsec/60., temps-273.15, alpha=.6)

    dict_fires["TIME [min]"] = np.arange(args["time_start"], args["fire_duration"], args["time_step"]) / 60.

    df_fires = pd.DataFrame(dict_fires)
    list_names = ["TIME [min]"] + list_fire_name
    df_fires = df_fires[list_names]

    # Save graphical plot to a .png file
    # ----------------------------------

    # Save numerical data to a .csv file
    # ----------------------------------
    file_name = "{} - {}".format(__id, __fn_selected_numerical)
    df_fires.to_csv(os.path.join(__dir_work, file_name))


def step3_calc_post(list_dir_file, list_input, list_pref, list_output):

    # OBTAIN OUTPUT FILE PATHS
    # ========================

    # SAVE NUMERICAL VALUES IN A .CSV FILE
    # ====================================

    for i, output_ in enumerate(list_output):

        # Obtain variable of individual simulation case
        # ---------------------------------------------

        dir_file_ = list_dir_file[i]
        input_ = list_input[i]
        pref_ = list_pref[i]

        # Obtain some useful strings / directories / variables
        # ----------------------------------------------------

        id_ = os.path.basename(dir_file_).split('.')[0]
        dir_work_ = os.path.dirname(dir_file_)
        # plt_format = {"figure_size_scale": 0.5,
        #               "axis_lim_y1": (0, 1),
        #               "axis_lim_x": (0, 120),
        #               "legend_is_shown": True,
        #               "marker_size": 0,
        #               "mark_every": 200,
        #               "axis_grid_show": True}

        # update global varibles for file name and word path directory
        init_global_variables(id_, dir_work_)

        # Save numerical values (inputs and outputs, i.e. everything)
        # -----------------------------------------------------------

        fn_ = os.path.join(dir_work_, " - ".join([id_, __fn_output_numerical]))
        pd.concat([input_, output_], axis=1).to_csv(fn_)

        # Plot distribution
        # -----------------
        plot_dist(id_, input_, [])

        # Plot and save time equivalence for individual output
        # ----------------------------------------------------

        # obtain x and y values for plot
        x = np.sort(output_["TIME EQUIVALENCE [min]"].values)
        y = np.arange(1, len(x) + 1) / len(x)


    # PLOT TIME EQUIVALENCE FOR ALL OUTPUT
    # ====================================
    line_vertical, line_horizontal = 0, 0
    teq_fig, teq_ax = plt.subplots(figsize=(3.94, 2.76))
    for i, output_ in enumerate(list_output):

        # Obtain variable of individual simulation case
        # ---------------------------------------------

        dir_file_ = list_dir_file[i]

        # Obtain some useful strings / directories / variables
        # ----------------------------------------------------

        id_ = os.path.basename(dir_file_).split('.')[0]
        dir_work_ = os.path.dirname(dir_file_)

        # obtain x and y values for plot

        x = np.sort(output_["TIME EQUIVALENCE [min]"].values)
        y = np.arange(1, len(x) + 1) / len(x)

        # plot the x, y

        plt.plot(x, y, label=id_)
        teq_ax.set_ylabel('Fractile')
        teq_ax.set_xlabel('Time [min]')
        teq_ax.set_xticks(ticks=np.arange(0, 180.1, 30))
        # plt.plot2(x, y, id_)
        # plt.format(**plt_format_)
        
        # plot horizontal and vertical lines (i.e. fractile and corresponding time euiqvalence period)

        if 0 < pref_["building_height"] < 1:
            line_horizontal = pref_["building_height"]
        elif pref_["building_height"] > 1:
            line_horizontal = 1 - 64.8 / pref_["building_height"] ** 2
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
        os.path.join(__dir_work, "{} - {}".format(os.path.basename(__dir_work), __fn_plot_te)),
        bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()
        #
    # plt.format(**plt_format_)
    #
    # plt.plot_vertical_line(x=x__)
    # plt.plot_horizontal_line(y=line_horizontal)
    # plt.axes_primary.text(x=x_end, y=line_horizontal, s="{:.4f}".format(line_horizontal), va="center", ha="left", fontsize=6)
    #
    # plt.save_figure2(os.path.join(dir_work_, "{} - {}".format(os.path.basename(dir_work_), __fn_plot_te)))
    #
    # plt.self_delete()


# def step4_results_visulisation(dict_pref, df_output):
#
#     # # Load settings
#
#     dict_settings = dict_pref
#     height_building = dict_settings["building_height"]
#
#     # Load the dataframe obj file
#
#     df_results = df_output
#
#     # Define horizontal line(s) to plot
#
#     if height_building > 0:
#         y__ = 1 - 64.8 / height_building ** 2
#     else:
#         y__ = 0.5
#
#     # Obtain time equivalence, in minutes, as x-axis values
#
#     x = np.sort(df_results["TIME EQUIVALENCE [min]"].values * 60.)
#     y = np.arange(1, len(x) + 1) / len(x)
#     f_interp = interp1d(y, x)
#     if height_building > 0:
#         y_line = 1 - 64.8 / height_building ** 2
#         x_line = f_interp(y_line)
#     else:
#         x_line = y_line = 0
#
#     plt = Scatter2D()
#     plt_format = {"figure_size_scale": __plot_scale,
#                   "axis_lim_y1": (0, 1),
#                   "axis_lim_x": __plot_x_limit,
#                   "legend_is_shown": False,
#                   "axis_label_x": "Time Equivalence [min]",
#                   "axis_label_y1": "Fractile",
#                   "marker_size": __plot_mark_size,
#                   "mark_every": __plot_mark_every}
#     plt.plot2(x/60, y, "Simulation results")
#
#     plt.format(**plt_format)
#
#     if height_building > 0:
#         x_end = plt.axes_primary.get_xlim()[1]
#         y_end = plt.axes_primary.get_ylim()[1]
#
#         x_line_ = x_line/60.
#         y_line_ = y_line
#         plt.plot_vertical_line(x=x_line_)
#         plt.axes_primary.text(x=x_line_, y=y_end, s="{:.0f}".format(x_line_), va="bottom", ha="center", fontsize=6)
#         plt.plot_horizontal_line(y=y_line_)
#         plt.axes_primary.text(x=x_end, y=y_line_, s="{:.4f}".format(y_line_), va="center", ha="left", fontsize=4)
#
#     file_path = os.path.join(__dir_work, "{} - {}".format(__id, __fn_plot_te))
#     plt.save_figure2(file_path)
#     plt.self_delete()


# def step5_results_visulisation_all(list_id, list_pref, list_output):
#
#     # INPUT VALIDATION
#
#     # list_id, list_pref, list_output must be lists with the same length and each contains at least 1 item.
#
#     if len(list_pref) <= 1:
#         return 0
#     elif len(list_pref) != len(list_output):
#         return 0
#
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # Plot a Graph for All Data
#     # ------------------------------------------------------------------------------------------------------------------
#
#     # instantiate plotting object
#     plt = Scatter2D()
#
#     # format parameters for figure
#     plt_format = {"figure_size_scale": __plot_scale,
#                   "axis_lim_y1": (0, 1),
#                   "axis_lim_x": __plot_x_limit,
#                   "legend_is_shown": True,
#                   "axis_label_x": "Time Equivalence [min]",
#                   "axis_label_y1": "Fractile",
#                   "marker_size": __plot_mark_size,
#                   "mark_every": __plot_mark_every}
#
#     # format parameters for additional texts which indicate the x_line and y_line values
#     # plt_format_text = {"fontsize": 6, "bbox": dict(boxstyle="square", fc="w", ec="b")}
#
#     # container for x_line and y_line
#     x_line, y_line = [], []
#     height_building = 0
#
#     # iterate through all result files and plot lines accordingly
#     for i, dict_pref in enumerate(list_pref):
#         df_output = list_output[i]
#
#         # Load settings
#
#         dict_settings = dict_pref
#         height_building = dict_settings["building_height"]
#
#         df_results = df_output
#
#         # obtain values: x, y, x_line (vertical line) and y_line (horizontal line)
#
#         x = df_results["TIME EQUIVALENCE [min]"].values
#         x = np.sort(x)
#         y = np.arange(1, len(x) + 1) / len(x)
#         f_interp = interp1d(y, x)
#         if height_building == 0:
#             y_line_ = 0
#         else:
#             y_line_ = 1 - 64.8 / height_building ** 2
#             x_line_ = f_interp(y_line_)
#
#         # plot line f(x)
#
#         plt.plot2(x, y, list_id[i])
#         plt.format(**plt_format)
#
#         # obtain x_line and y_line for later use
#
#         if height_building > 0:
#             x_line.append(round(float(x_line_), 0))
#             y_line.append(round(float(y_line_), 4))
#
#     plt.format(**plt_format)
#
#     if height_building > 0:
#         x_line = set(x_line)
#         y_line = set(y_line)
#         x_end = plt.axes_primary.get_xlim()[1]
#         y_end = plt.axes_primary.get_ylim()[1]
#
#         x_line = [max(x_line)]
#
#         for x_line_ in x_line:
#             plt.plot_vertical_line(x=x_line_)
#             plt.add_text(x=x_line_, y=y_end, s="{:.0f}".format(x_line_), va="bottom", ha="center", fontsize=6)
#
#         for y_line_ in y_line:
#             plt.plot_horizontal_line(y=y_line_)
#             plt.add_text(x=x_end, y=y_line_, s="{:.4f}".format(y_line_), va="center", ha="left", fontsize=4)
#
#     file_name = "{} - {}".format(os.path.basename(__dir_work), __fn_plot_te_all)
#     file_path = os.path.join(__dir_work, file_name)
#     plt.save_figure2(path_file=file_path)
#     plt.self_delete()


# def step6_results_visualization_temperature(df_input, dict_pref, df_output):
#
#     if np.sum(df_input["protection_thickness"].values) <= 0:
#         return 0
#
#     dict_settings = dict_pref
#     height_building = dict_settings["building_height"]
#
#     # Load the dataframe obj file
#
#     df_results = df_output
#
#     # Define horizontal line(s) to plot
#
#     if height_building > 0:
#         y__ = 1 - 64.8 / height_building ** 2
#     else:
#         y__ = 0.5
#
#     # Obtain time equivalence, in minutes, as x-axis values
#
#     x = df_results["PEAK STEEL TEMPERATURE TO FIXED PROTECTION [C]"].values
#     x = np.sort(x)
#     y = np.arange(1, len(x) + 1) / len(x)
#
#     plt = Scatter2D()
#
#     plt_format = {"figure_size_scale": 0.7,
#                   "axis_lim_y1": (0, 1),
#                   # "axis_lim_x": __plot_x_limit,
#                   "legend_is_shown": False,
#                   "axis_label_x": "Peak Steel Temperature [$^\circ$C]",
#                   "axis_label_y1": "Fractile",
#                   "marker_size": __plot_mark_size,
#                   "mark_every": __plot_mark_every}
#     plt.plot2(x, y, "Simulation results")
#
#     plt.format(**plt_format)
#
#     file_name = "{} - {}".format(__id, __fn_selected_plot)
#     file_path = os.path.join(__dir_work, file_name)
#     plt.save_figure2(file_path)


# def step7_select_fires_teq(df_input, dict_pref, df_output):
#
#     if dict_pref["select_fires_teq"] < 0:
#         return 0
#
#     # Load settings
#
#     dict_settings = dict_pref
#     height_building = dict_settings["building_height"]
#     percentile = dict_settings["select_fires_teq"]
#     tolerance = dict_settings["select_fires_teq_tol"]
#
#     # Load results and input arguments
#     df_results = df_output
#     df_input_arguments = df_input
#     df_results.sort_values(by=["TIME EQUIVALENCE [min]"], inplace=True)
#
#     if tolerance < 0:
#         tolerance = 0
#
#     # Convert 'percentile_ubound' and 'percentile_lbound' to integers according to actual range i.e. 'index=1000'
#
#     index_max = int(max(df_results.index.values))
#
#     percentile_ubound = percentile + abs(tolerance)
#     percentile_lbound = percentile - abs(tolerance)
#
#     percentile_ubound *= index_max
#     percentile_lbound *= index_max
#
#     percentile_ubound = int(round(percentile_ubound, 0))
#     percentile_lbound = int(round(percentile_lbound, 0))
#
#     if percentile_lbound <= percentile_ubound:
#         range_selected = np.arange(percentile_lbound, percentile_ubound+1, 1)
#     else:
#         print("NO FIRES ARE SELECTED.")
#         return 0
#
#     list_index_selected_fires = df_results.iloc[range_selected].index.values
#     df_results_selected = df_results.iloc[range_selected]
#
#     # iterate through all selected fires, store time and temperature
#
#     plt = Scatter2D()
#     dict_fires = {}
#     list_fire_name = ["TEMPERATURE {} [C]".format(str(i)) for i,v in enumerate(list_index_selected_fires)]
#     for i,v in enumerate(list_index_selected_fires):
#         # get input arguments
#         args = df_input_arguments.loc[i].to_dict()
#
#         # get fire type
#         fire_type = int(df_results.loc[i]["FIRE TYPE [0:P., 1:T.]"])
#
#         if fire_type == 0:  # parametric fire
#             w, l, h = args["room_breadth"], args["room_depth"], args["room_height"]
#             inputs_parametric_fire = {
#                 "A_t": 2*(w*l+w*h+h*l),
#                 "A_f": w*l,
#                 "A_v": args["window_height"] * args["window_width"] * args["window_open_fraction"],
#                 "h_eq": args["window_height"],
#                 "q_fd": args["fire_load_density"] * 1e6,
#                 "lambda_": args["room_wall_thermal_inertia"] ** 2,  # thermal inertia is used instead of k rho c.
#                 "rho": 1,  # see comment for lambda_
#                 "c": 1,  # see comment for lambda_
#                 "t_lim": args["time_limiting"],
#                 "time_end": args["fire_duration"],
#                 "time_step": args["time_step"],
#                 "time_start": args["time_start"],
#                 # "time_padding": (0, 0),
#                 "temperature_initial": 20 + 273.15,
#             }
#             tsec, temps = _fire_param(**inputs_parametric_fire)
#         elif fire_type == 1:  # travelling fire
#             inputs_travelling_fire = {
#                 "fire_load_density_MJm2": args["fire_load_density"],
#                 "heat_release_rate_density_MWm2": args["fire_hrr_density"],
#                 "length_compartment_m": args["room_depth"],
#                 "width_compartment_m": args["room_breadth"],
#                 "fire_spread_rate_ms": args["fire_spread_speed"],
#                 "height_fuel_to_element_m": args["room_height"],
#                 "length_element_to_fire_origin_m": args["beam_position"],
#                 "time_start_s": args["time_start"],
#                 "time_end_s": args["fire_duration"],
#                 "time_interval_s": args["time_step"],
#                 "nft_max_C": args["temperature_max_near_field"],
#                 "win_width_m": args["window_width"],
#                 "win_height_m": args["window_height"],
#                 "open_fract": args["window_open_fraction"]
#             }
#             tsec, temps, hrr, r = _fire_travelling(**inputs_travelling_fire)
#             temps += 273.15
#         else:
#             print("FIRE TYPE UNKOWN.")
#
#         dict_fires[list_fire_name[i]] = temps-273.15
#         plt.plot2(tsec/60., temps-273.15, alpha=.6)
#
#     dict_fires["TIME [min]"] = np.arange(args["time_start"],args["fire_duration"],args["time_step"]) / 60.
#
#     df_fires = pd.DataFrame(dict_fires)
#     list_names = ["TIME [min]"] + list_fire_name
#     df_fires = df_fires[list_names]
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # Save graphical plot to a .png file
#     # ------------------------------------------------------------------------------------------------------------------
#     # format parameters for figure
#     plt_format = {
#         "figure_size_scale": 0.4,
#         "axis_lim_y1": (0, 1400),
#         "axis_lim_x": __plot_x_limit,
#         "legend_is_shown": False,
#         "axis_label_x": "Time [min]",
#         "axis_label_y1": "Gas Temperature [$^\circ$C]",
#         "marker_size": __plot_mark_size,
#         "mark_every": __plot_mark_every,
#         "axis_xtick_major_loc": np.arange(0, 181, 20),
#         "line_colours": [(0, 0, 0)]
#     }
#     plt.format(**plt_format)
#     file_name = "{} - {}".format(__id, __fn_selected_plot)
#     file_path = os.path.join(__dir_work, file_name)
#
#     plt.save_figure2(path_file=file_path)
#     # saveprint(os.path.basename(file_path))
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # Save numerical data to a .csv file
#     # ------------------------------------------------------------------------------------------------------------------
#     file_name = "{} - {}".format(__id, __fn_selected_numerical)
#     df_fires.to_csv(os.path.join(__dir_work, file_name))
#     # saveprint(file_name)


def run(project_full_paths=list()):

    _strfmt_1_1 = "{:25}{}"
    __fn_output = "res.p"

    if len(project_full_paths) == 0:
        while True:
            project_full_path = input("Work directory: ")

            if project_full_path == "" and len(project_full_paths) != 0:
                break

            project_full_path.replace('"', '')
            project_full_path.replace("'", '')
            project_full_path = os.path.abspath(os.path.realpath(project_full_path))

            project_full_paths.append(project_full_path)

    # MAIN BODY
    # =========

    for project_full_path in project_full_paths:
        list_files = step0_parse_input_files(dir_work=project_full_path)
        list_files.sort()
        list_input, list_output, list_pref, list_id = [], [], [], []
        print(_strfmt_1_1.format("Work directory:", project_full_path))
        for f in list_files:
            id_ = os.path.basename(f).split(".")[0]
            init_global_variables(id_, project_full_path)

            # Step 1: make a list of key word arguments for function inputs
            # -------------------------------------------------------------

            df_input, dict_pref = step1_inputs_maker(f)

            ff = os.path.join(project_full_path, " - ".join([id_, __fn_output]))
            if os.path.isfile(ff):
                df_output = pload(open(ff, "rb"))
            else:

                # Step 2: perform main time equivalence calculation
                # -------------------------------------------------

                df_output = step2_calc(df_input, dict_pref)

            list_input.append(df_input)
            list_id.append(id_)
            list_pref.append(dict_pref)
            list_output.append(df_output)

        # Step 3: select fire curves and output numerical and graphical files
        # -------------------------------------------------------------------

        step3_calc_post(list_files, list_input, list_pref, list_output)


if __name__ == '__main__':
    # run(['/Users/fuyans/Desktop/untitled'])
    run()
