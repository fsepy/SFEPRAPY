import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from pickle import load as pload
from pickle import dump as pdump
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt

from sfeprapy.func.temperature_fires import standard_fire_iso834 as _fire_standard
from sfeprapy.func.temperature_fires import parametric_eurocode1 as _fire_param
from sfeprapy.func.tfm_alt import travelling_fire as _fire_travelling
from sfeprapy.time_equivalence_mc import calc_time_equiv_worker, mc_inputs_generator

sns.set_style("ticks", {'axes.grid': True})

# OUTPUT STRING FORMAT
# ====================

__strformat_1_1 = "{:25}{}"
__strformat_1_1_1 = "{:25}{:3}{}"

# GLOBAL SETTINGS
# ===============

# Global variables
# -------

__app_pref = {
    'problem id': None,
    'problem folder directory': None,
    'result p name': 'res.p',
    'result csv name': 'res.csv',
    'result png name': 'teq_all.png',
    'result fire csv name': 'fires.csv'
}

# __id = ""
# __dir_work = ""
__fn_output = "res.p"
__fn_output_numerical = "res.csv"
__fn_plot_te = "teq.png"
__fn_plot_te_all = "teq_all.png"

# __fn_plot_temp = "plot_temp.png"

__fn_selected_plot = "fires.png"
__fn_fires_numerical = "fires.csv"


# def init_global_variables(_id, dir_work):
#     global __id, __dir_work
#     __id = _id
#     __dir_work = dir_work


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

    fire = _fire_standard(np.arange(0, 3 * 60 * 60, 1), 273.15 + 20)
    inputs_extra = {"iso834_time": fire[0],
                    "iso834_temperature": fire[1], }

    df_input, dict_pref = mc_inputs_generator(dict_extra_variables_to_add=inputs_extra, dir_file=path_input_file)

    return df_input, dict_pref


def step2_calc(df_input, dict_pref, path_input_file, progress_print_interval=5):
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

    print(__strformat_1_1.format("Input file:", os.path.basename(path_input_file)))
    print(__strformat_1_1.format("Total simulations:", len(list_kwargs)))
    print(__strformat_1_1.format("Number of threads:", n_proc))

    time_simulation_start = time.perf_counter()
    m = mp.Manager()
    q = m.Queue()
    p = mp.Pool(n_proc, maxtasksperchild=mp_maxtasksperchild)
    jobs = p.map_async(calc_time_equiv_worker, [(kwargs, q) for kwargs in list_kwargs])
    count_total_simulations = len(list_kwargs)
    n_steps = 24  # length of the progress bar
    while progress_print_interval:
        if jobs.ready():
            time_simulation_consumed = time.perf_counter() - time_simulation_start
            print("{}{} {:.1f}s".format('█' * round(n_steps), '-' * round(0), time_simulation_consumed))
            break
        else:
            p_ = q.qsize() / count_total_simulations * n_steps
            print("{}{} {:03.1f}%".format('█' * int(round(p_)), '-' * int(n_steps - round(p_)), p_ / n_steps * 100),
                  end='\r')
            time.sleep(1)
    p.close()
    p.join()
    results = jobs.get()

    # format outputs

    results = np.array(results)

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

    path_results_file = os.path.join(
        os.path.dirname(path_input_file),
        "{} - {}".format(
            os.path.basename(path_input_file).split('.')[0],
            __fn_output
        )
    )
    pdump(df_output, open(path_results_file, "wb"))

    return df_output


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
        ax.set_xlabel('x')

        # Small simple plot parameters
        # ax.set_ylabel('')
        # ax.set_yticklabels([])
        # ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(path_input_file), '{} - dist - {}.png'.format(id_, v)), ppi=300, transparent=True)
        plt.cla()

    plt.clf()


def select_fires_teq(df_output, path_input_file):

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
                "lambda_": args["ROOM WALL THERMAL INERTIA [J/m2s1/2K]"] ** 2,  # thermal inertia is used instead of k rho c.
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
    file_name = "{} - {}".format(os.path.basename(path_input_file).split('.')[0], __fn_fires_numerical)
    df_fires.to_csv(os.path.join(os.path.dirname(path_input_file), file_name))


def step3_calc_post(list_path_input_file, list_pref, list_output):

    # SAVE NUMERICAL VALUES IN A .CSV FILE
    # ====================================

    for i, output_ in enumerate(list_output):
        # Obtain variable of individual simulation case
        # ---------------------------------------------

        path_input_file = list_path_input_file[i]

        # Obtain some useful strings / directories / variables
        # ----------------------------------------------------

        id_ = os.path.basename(path_input_file).split('.')[0]
        dir_work_ = os.path.dirname(path_input_file)

        # update global variables for file name and word path directory
        # init_global_variables(id_, dir_work_)

        # Save selected fires
        select_fires_teq(df_output=output_, path_input_file=path_input_file)

        # Save numerical values (inputs and outputs, i.e. everything)
        # -----------------------------------------------------------

        fn_ = os.path.join(dir_work_, " - ".join([id_, __fn_output_numerical]))

        output_.to_csv(fn_)

        # Plot distribution
        # -----------------
        if list_pref[i]['simulations'] > 2:
            plot_dist(id_, output_, path_input_file=path_input_file, headers=[])

        # Plot and save time equivalence for individual output
        # ----------------------------------------------------

        # obtain x and y values for plot
        x = np.sort(output_["TIME EQUIVALENCE [s]"].values) / 60.
        y = np.arange(1, len(x) + 1) / len(x)

    # PLOT TIME EQUIVALENCE FOR ALL OUTPUT
    # ====================================

    # do not plot if 'deterministic'

    line_vertical, line_horizontal = 0, 0
    teq_fig, teq_ax = plt.subplots(figsize=(3.94, 2.76))
    teq_ax.set_xlim([0, 120])
    for i, output_ in enumerate(list_output):

        # output_ = output_[output_['SEEK STATUS [0:Fail, 1:Success]'] == 1]

        # Obtain variable of individual simulation case
        # ---------------------------------------------

        path_input_file = list_path_input_file[i]
        pref_ = list_pref[i]

        # Obtain some useful strings / directories / variables
        # ----------------------------------------------------

        id_ = os.path.basename(path_input_file).split('.')[0]
        dir_work_ = os.path.dirname(path_input_file)

        # Check 'deterministic'
        # ----------------------
        if pref_['simulations'] <= 2:
            continue

        # obtain x and y values for plot

        x = np.sort(output_["TIME EQUIVALENCE [s]"].values) / 60.  # to minutes
        y = np.arange(1, len(x) + 1) / len(x)

        # plot the x, y

        plt.plot(x, y, label=id_)
        teq_ax.set_ylabel('Fractile')
        teq_ax.set_xlabel('Time [min]')
        teq_ax.set_xticks(ticks=np.arange(0, 180.1, 30))
        # plt.plot2(x, y, id_)
        # plt.format(**plt_format_)

        # plot horizontal and vertical lines (i.e. fractile and corresponding time euiqvalence period)

        if 0 < pref_["reliability_target"] < 1:
            line_horizontal = pref_["reliability_target"]
        elif pref_["reliability_target"] > 1:
            line_horizontal = 1 - 64.8 / pref_["reliability_target"] ** 2
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
        os.path.join(os.path.dirname(list_path_input_file[0]), "{} - {}".format(os.path.basename(os.path.dirname(list_path_input_file[0])), __fn_plot_te)),
        bbox_inches='tight', dpi=300, transparent=True
    )

    plt.clf()
    plt.close()

__run_count = 0


# DEPRECIATED 20181114. MC portal migrated to /sfeprapy/mc.__main__
# def run(project_full_paths=list()):
#     _strfmt_1_1 = "{:25}{}"
#     __fn_output = "res.p"
#
#     global __run_count
#     __run_count += 1
#
#     if len(project_full_paths) == 0 or __run_count > 1:
#         from tkinter import filedialog, Tk, StringVar
#         root = Tk()
#         root.withdraw()
#         folder_path = StringVar()
#         while True:
#             file_name = filedialog.askdirectory(title='Select problem definitions folder')
#             if not file_name: break
#             folder_path.set(file_name)
#             project_full_paths.append(file_name)
#
#     # MAIN BODY
#     # =========
#
#     for project_full_path in project_full_paths:
#         list_path_input_file = step0_parse_input_files(dir_work=project_full_path)
#         list_path_input_file.sort()
#         list_input, list_output, list_pref, list_id = [], [], [], []
#         print(_strfmt_1_1.format("Work directory:", project_full_path))
#         for path_input_file in list_path_input_file:
#             id_ = os.path.basename(path_input_file).split(".")[0]
#
#             # Step 1: make a list of key word arguments for function inputs
#             # -------------------------------------------------------------
#
#             df_input, dict_pref = step1_inputs_maker(path_input_file)
#
#             ff = os.path.join(project_full_path, " - ".join([id_, __fn_output]))
#             if os.path.isfile(ff):
#                 df_output = pload(open(ff, "rb"))
#             else:
#
#                 # Step 2: perform main time equivalence calculation
#                 # -------------------------------------------------
#
#                 df_output = step2_calc(df_input, dict_pref, path_input_file)
#
#             list_input.append(df_input)
#             list_id.append(id_)
#             list_pref.append(dict_pref)
#             list_output.append(df_output)
#
#         # Step 3: select fire curves and output numerical and graphical files
#         # -------------------------------------------------------------------
#
#         step3_calc_post(list_path_input_file, list_pref, list_output)
#
#     input("Press Enter to finish")


if __name__ == '__main__':
    # run(['/Users/fuyans/Desktop/untitled'])
    # run()
    pass
