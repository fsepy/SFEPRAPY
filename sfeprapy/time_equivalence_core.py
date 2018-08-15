# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats.distributions import norm, gumbel_r
from pandas import DataFrame as df

from sfeprapy.func.tfm_alt import travelling_fire as _fire_travelling
from sfeprapy.dat.steel_carbon import Thermal
from sfeprapy.func.temperature_steel_section import protected_steel_eurocode as _steel_temperature
from sfeprapy.func.temperature_fires import parametric_eurocode1 as _fire_param
from sfeprapy.func.kwargs_from_text import kwargs_from_text


def lognorm_parameters_true_to_inv(miu, sigma):
    """
    NAME: lognorm_parameters_true_to_inv
    VERSION: 0.0.1
    AUTHOR: Yan Fu, Ruben Coline
    DESCRIPTION:
    Converts the mean and standard deviation of distribution from x to ln(x).

    PARAMETERS:
    :param miu: True mean of the x
    :param sigma: True standard deviation of x
    :return: (miu, sigma) where miu and sigma are based on ln(x)

    USAGE:
    >>> print(lognorm_parameters_true_to_inv(0.2, 0.2))
    >>> (-1.9560115027140728, 0.8325546111576977)
    """
    cov = sigma / miu

    sigma_ln = np.sqrt(np.log(1 + cov ** 2))
    miu_ln = np.log(miu) - 1 / 2 * sigma_ln ** 2

    return miu_ln, sigma_ln

# if __name__ == '__main__':
#     print(lognorm_parameters_true_to_inv(0.2, 0.2))


def lognorm_trunc_ppf(a, b, n_rv, sigma, loc, scale, cdf_y=None):
    """
    NAME: lognorm_trunc_ppf
    VERSION: 0.0.1
    AUTHOR: Yan Fu
    DATE: 3 Aug 2018
    DESCRIPTION:
    Truncated log normal distribution cumulative function density. Truncate and normalise  a log normal distribution
    function for a given boundary (a, b).

    PARAMETERS:
    :param a: float, Lower boundary
    :param b: float, Upper boundary
    :param n_rv: integer, total number of
    :param sigma: float, standard deviation of log normal distribution
    :param loc: float, location of log normal distribution
    :param scale: float, scale of the log normal distribution
    :param cdf_y: array (1 dimension) or None. A set of numbers represent cumulative probability. If None the function
    will return the sampled values
    :return: array (1 dimension), set of numbers represent sampled values of truncated log normal distribution inline
    with 'cfd_y'

    USAGE:
    >>> import numpy as np
    >>> a = 0                   # lower boundary
    >>> b = 1                   # upper boundary
    >>> n_rv = 5                # number of random variables
    >>> sigma = 0.2             # standard deviation
    >>> miu = 0.2               # mean

    # Convert true mean and sigma to ln(x) based mean and sigma
    >>> miu, sigma = lognorm_parameters_true_to_inv(miu, sigma)

    >>> loc = np.exp(miu)       # distribution mean is 0.2 (i.e. loc = np.exp(miu))
    >>> scale = 1               # default
    >>> result = lognorm_trunc_ppf(a, b, n_rv, sigma, loc, scale)
    >>> print(result)
    [0.14142136 0.49653783 0.65783987 0.81969479 1.        ]
    """

    # Generate a linear spaced array inline with lower and upper boundary of log normal cumulative probability density.
    sampled_cfd = np.linspace(
        stats.lognorm.cdf(x=a, s=sigma, loc=loc, scale=scale),
        stats.lognorm.cdf(x=b, s=sigma, loc=loc, scale=scale),
        n_rv
    )

    # Sample log normal distribution
    sampled = stats.lognorm.ppf(q=sampled_cfd, s=sigma, loc=loc, scale=scale)

    # Work out cumulative probability function from 'sampled', output in forms of x y.
    # Interpolate x and y are processed to be capable to cope with two extreme values. y[0] (cumulative probability,
    # initial boundary) is manually set to 0.
    x = np.linspace(a, b, int(n_rv), endpoint=False)
    x += (x[1] - x[0]) / 2
    x[-1] -= (x[1] - x[0]) / 2
    x = np.append([0], x)
    y = np.array([np.sum(sampled <= i) for i in x]) / len(sampled)
    y[0] = 0

    # Interpolate
    f = interp1d(y, x, bounds_error=False, fill_value=(np.min(y), np.max(y)))

    if cdf_y is None:
        return sampled
    else:
        return f(cdf_y)


def gumbel_parameter_converter(miu, sigma):
    # parameters Gumbel W&S
    alpha = 1.282 / sigma
    u = miu - 0.5772 / alpha

    # parameters Gumbel scipy
    scale = 1 / alpha
    loc = u

    return loc, scale


def gumbel_r_trunc_ppf(a, b, n_rv, loc, scale, cdf_y=None):
    """
    NAME:
    VERSION:
    AUTHOR:
    DATE:
    DESCRIPTION:

    PARAMETERS:
    :param a:
    :param b:
    :param n_rv:
    :param loc:
    :param scale:
    :param cdf_y:
    :return:

    USAGE:

    """

    # Generate a linear spaced array inline with lower and upper boundary of log normal cumulative probability density.
    sampled_cfd = np.linspace(
        stats.gumbel_r.cdf(x=a, loc=loc, scale=scale),
        stats.gumbel_r.cdf(x=b, loc=loc, scale=scale),
        n_rv
    )

    # Following three lines are used to check the validity of the distribution
    # print("511 (0.80): {:.4f}".format(stats.gumbel_r.cdf(x=510, loc=loc, scale=scale)))
    # print("584 (0.90): {:.4f}".format(stats.gumbel_r.cdf(x=584, loc=loc, scale=scale)))
    # print("655 (0.95): {:.4f}".format(stats.gumbel_r.cdf(x=655, loc=loc, scale=scale)))

    # Sample log normal distribution
    sampled = stats.gumbel_r.ppf(q=sampled_cfd, loc=loc, scale=scale)

    # Work out cumulative probability function from 'sampled', output in forms of x y.
    # Interpolate x and y are processed to be capable to cope with two extreme values. y[0] (cumulative probability,
    # initial boundary) is manually set to 0.
    x = np.linspace(a, b, int(n_rv)+1, endpoint=False)
    x += (x[1] - x[0]) / 2
    x = x[0:-2]
    # x[-1] -= (x[1] - x[0]) / 2
    # x = np.append([0], x)
    y = np.array([np.sum(sampled <= i) for i in x]) / len(sampled)
    # y[0] = 0

    # Interpolate
    f = interp1d(y, x, bounds_error=False, fill_value=(np.min(y), np.max(y)))

    if cdf_y is None:
        return sampled
    else:
        return f(cdf_y)


# if __name__ == '__main__':
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     loc, scale = gumbel_parameter_converter(600, 180)
#     sns.distplot(gumbel_r_trunc_ppf(0, 1800, 10000, loc, scale))
#     plt.show()


def latin_hypercube_sampling(num_samples, num_arguments=1, sample_lbound=0, sample_ubound=1):
    """
    NAME: latin_hypercube_sampling
    AUTHOR: Yan Fu
    VERSION: 0.1
    DATE: 3 Aug 2018
    DESCRIPTION:
    Latin Hypercube Sampling, generates an nxm array where m equal to 'num_arguments' and n equal to 'num_samples'.
    Current version only adopts 'centered' sampling mode (each sampled cell value is centered).

    PARAMETERS:
    :param num_samples: Number of samples (i.e. rows)
    :param num_arguments: Number of arguments (i.e. columns)
    :param sample_lbound: Lower sampling boundary
    :param sample_ubound: Upper sampling boundary
    :return: An array with shape (num_samples, num_arguments)

    EXAMPLE:
    >>> result = latin_hypercube_sampling(num_samples=10, num_arguments=3, sample_lbound=0, sample_ubound=0.001)
    This example yields an array with shape of (100, 3), with each column filled 100 linear spaced numbers (shuffled)
    from 1 to 0.001 (i.e. shuffled [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]).
    An example output:
    result = [
        [0.85 0.95 0.55]
        [0.55 0.45 0.05]
        [0.25 0.25 0.15]
        [0.35 0.85 0.75]
        [0.45 0.65 0.85]
        [0.75 0.05 0.65]
        [0.05 0.35 0.45]
        [0.15 0.15 0.35]
        [0.65 0.55 0.25]
        [0.95 0.75 0.95]
    ]

    """

    if sample_lbound > sample_ubound:
        sample_ubound += sample_lbound
        sample_lbound = sample_ubound - sample_lbound
        sample_ubound = sample_ubound - sample_lbound

    # Generate sorted integers with correct shape
    mat_random_num = np.linspace(sample_lbound, sample_ubound, num_samples + 1, dtype=float)
    mat_random_num += (mat_random_num[1] - mat_random_num[0]) * 0.5
    mat_random_num = mat_random_num[0:-1]
    mat_random_num = np.reshape(mat_random_num, (len(mat_random_num), 1))
    mat_random_nums = mat_random_num * np.ones((1, num_arguments))

    # np.random.shuffle(mat_random_nums)

    for i in range(np.shape(mat_random_nums)[1]):
        np.random.shuffle(mat_random_nums[:, i])

    if num_arguments == 1:
        mat_random_nums = mat_random_nums.flatten()

    return mat_random_nums


def calc_time_equiv_worker(arg):
    kwargs, q = arg
    result = calc_time_equivalence(**kwargs)
    q.put("index: {}".format(kwargs["index"]))
    return result


def calc_time_equivalence(
        time_step,
        time_start,
        time_limiting,
        window_height,
        window_width,
        window_open_fraction,
        room_breadth,
        room_depth,
        room_height,
        room_wall_thermal_inertia,
        fire_load_density,
        fire_hrr_density,
        fire_spread_speed,
        fire_duration,
        beam_position,
        beam_rho,
        beam_c,
        beam_cross_section_area,
        beam_temperature_goal,
        protection_k,
        protection_rho,
        protection_c,
        protection_thickness,
        protection_protected_perimeter,
        iso834_time,
        iso834_temperature,
        nft_ubound=1200,
        seek_ubound_iter=20,
        seek_ubound=0.1,
        seek_lbound=0.0001,
        seek_tol_y=1.,
        index=-1,
        is_return_dict=False,
        **kwargs
):
    """
    NAME:
    VERSION:
    AUTHOR:
    DATE:
    DESCRIPTION:

    PARAMETERS:
    :param time_step:
    :param time_start:
    :param time_limiting:
    :param window_height:
    :param window_width:
    :param window_open_fraction:
    :param room_breadth:
    :param room_depth:
    :param room_height:
    :param room_wall_thermal_inertia:
    :param fire_load_density:
    :param fire_hrr_density:
    :param fire_spread_speed:
    :param fire_duration:
    :param beam_position:
    :param beam_rho:
    :param beam_c:
    :param beam_cross_section_area:
    :param beam_temperature_goal:
    :param protection_k:
    :param protection_rho:
    :param protection_c:
    :param protection_thickness:
    :param protection_protected_perimeter:
    :param iso834_time:
    :param iso834_temperature:
    :param nft_ubound:
    :param seek_ubound_iter:
    :param seek_ubound:
    :param seek_lbound:
    :param seek_tol_y:
    :param index:
    :param is_return_dict:
    :param kwargs:
    :return:

    EXAMPLE:
    """

    #   Check on applicable fire curve
    window_area = window_height * window_width * window_open_fraction
    room_floor_area = room_breadth * room_depth
    room_area = (2 * room_floor_area) + ((room_breadth + room_depth) * 2 * room_height)

    #   Opening factor - is it within EC limits?
    opening_factor = window_area * np.sqrt(window_height) / room_area

    #   Spread speed - Does the fire spread to involve the full compartment?
    sp_time = max([room_depth, room_breadth]) / fire_spread_speed
    burnout_m2 = max([fire_load_density / fire_hrr_density, 900.])
    # print("room depth:", room_depth)
    # print("fire speed:", fire_spread_speed)
    # print("sp_time:", sp_time)
    # print("burnout_m2:", burnout_m2)
    # print("opening factor:", opening_factor)
    # burnout_m2 = -1
    if sp_time < burnout_m2 and 0.02 < opening_factor <= 0.2:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire
        fire_time, fire_temp = _fire_param(**{"A_t": room_area,
                                              "A_f": room_floor_area,
                                              "A_v": window_area,
                                              "h_eq": window_height,
                                              "q_fd": fire_load_density * 1e6,
                                              "lambda_": room_wall_thermal_inertia ** 2,
                                              "rho": 1,
                                              "c": 1,
                                              "t_lim": time_limiting,
                                              "time_end": fire_duration,
                                              "time_step": time_step,
                                              "time_start": time_start,
                                              # "time_padding": (0, 0),
                                              "temperature_initial": 20 + 273.15, })

        fire_type = 0  # parametric fire

    else:  # Otherwise, it is a travelling fire
        #   Get travelling fire curve
        fire_time, fire_temp, heat_release, distance_to_element = _fire_travelling(
            fire_load_density,
            fire_hrr_density,
            room_depth,
            room_breadth,
            fire_spread_speed,
            room_height,
            beam_position,
            time_start,
            fire_duration,
            time_step,
            nft_ubound,
            window_width,
            window_height,
            window_open_fraction,
        )
        fire_temp += 273.15
        fire_type = 1  # travelling fire

    # Solve heat transfer using EC3 correlations
    # SI UNITS FOR INPUTS!
    inputs_steel_heat_transfer = {"time": fire_time,
                                  "temperature_ambient": fire_temp,
                                  "rho_steel": beam_rho,
                                  "c_steel_T": beam_c,
                                  "area_steel_section": beam_cross_section_area,
                                  "k_protection": protection_k,
                                  "rho_protection": protection_rho,
                                  "c_protection": protection_c,
                                  "thickness_protection": protection_thickness,
                                  "perimeter_protected": protection_protected_perimeter,
                                  "is_terminate_peak": True}

    # Find maximum steel temperature for the static protection layer thickness
    if protection_thickness > 0:
        temperature_steel_ubound = np.max(_steel_temperature(**inputs_steel_heat_transfer)[1])
    else:
        temperature_steel_ubound = -1

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS
    # ====================================================================

    seek_count_iter = 0  # count how many iterations for  the seeking process
    seek_status = False  # flag used to indicate when the seeking is successful

    # Default values
    time_fire_resistance = -1
    sought_temperature_steel_ubound = -1
    sought_protection_thickness = -1

    if beam_temperature_goal > 0:  # check seeking temperature, opt out if less than 0
        # Seeking process
        while seek_count_iter < seek_ubound_iter and seek_status is False:
            seek_count_iter += 1
            sought_protection_thickness = np.average([seek_ubound, seek_lbound])
            inputs_steel_heat_transfer["thickness_protection"] = sought_protection_thickness
            t_, T_, d_ = _steel_temperature(**inputs_steel_heat_transfer)
            sought_temperature_steel_ubound = np.max(T_)
            y_diff_seek = sought_temperature_steel_ubound - beam_temperature_goal
            if abs(y_diff_seek) <= seek_tol_y:
                seek_status = True
            elif sought_temperature_steel_ubound > beam_temperature_goal:  # steel too hot, increase protect thickness
                seek_lbound = sought_protection_thickness
            else:  # steel is too cold, increase intrumescent paint thickness
                seek_ubound = sought_protection_thickness

        # BEAM FIRE RESISTANCE PERIOD IN ISO 834
        # ======================================

        # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.
        inputs_steel_heat_transfer["time"] = iso834_time
        inputs_steel_heat_transfer["temperature_ambient"] = iso834_temperature
        time_, temperature_steel, data_all = _steel_temperature(**inputs_steel_heat_transfer)

        # re-arrange time and steel temperature, which will be used later, to prevent interpolation boundary error. If
        # boundaries are breached, the corresponding min or max of time will be returned, i.e. 0 and 2 hrs
        time_ = np.concatenate((np.array([0]), time_, np.array([time_[-1]])))
        temperature_steel = np.concatenate((np.array([-1]), temperature_steel, np.array([1e12])))

        # perform interpolation for teq based on acutal steel temperature and iso steel temperature
        interp_ = interp1d(temperature_steel, time_, kind="linear", bounds_error=False, fill_value=-1)
        time_fire_resistance = interp_(beam_temperature_goal)

    if is_return_dict:
        return {
            "time_fire_resistance": time_fire_resistance,
            "seek_status": seek_status,
            "window_open_fraction": window_open_fraction,
            "fire_load_density": fire_load_density,
            "fire_spread_speed": fire_spread_speed,
            "beam_position": beam_position,
            "nft_ubound": nft_ubound,
            "fire_type": fire_type,
            "sought_temperature_steel_ubound": sought_temperature_steel_ubound,
            "sought_protection_thickness": sought_protection_thickness,
            "seek_count_iter": seek_count_iter,
            "temperature_steel_ubound": temperature_steel_ubound,
            "index": index
        }
    else:
        return time_fire_resistance, seek_status, window_open_fraction, fire_load_density, fire_spread_speed, beam_position, nft_ubound, fire_type, sought_temperature_steel_ubound, sought_protection_thickness, seek_count_iter, temperature_steel_ubound, index


def mc_inputs_generator(dict_extra_variables_to_add=None, dir_file=str):
    steel_prop = Thermal()

    #   Handy interim functions

    linear_distribution = lambda min, max, prob: ((max - min) * prob) + min

    # DEPRECIATED (4 Aug 2018)
    # def linear_distribution(min, max, prob):
    #     return ((max - min) * prob) + min

    # ------------------------------------------------------------------------------------------------------------------
    #   Define the inputs from file
    # ------------------------------------------------------------------------------------------------------------------

    # dict_setting_vars = dict()

    # Read input variables from external text file
    with open(str(dir_file), "r") as file_inputs:
        string_inputs = file_inputs.read()
    dict_vars_0 = kwargs_from_text(string_inputs)
    if dict_extra_variables_to_add:
        dict_vars_0.update(dict_extra_variables_to_add)
    dict_vars_0["beam_c"] = steel_prop.c()

    # dict_vars_0_ is used for making the returned DataFrame, hence it passes necessary variables only
    dict_vars_0_ = copy.copy(dict_vars_0)
    simulations = dict_vars_0["simulations"]

    # Variable group definition
    list_setting_vars = ["simulations", "steel_temp_failure", "n_proc", "building_height", "select_fires_teq",
                         "select_fires_teq_tol"]
    list_interim_vars = ["qfd_std", "qfd_mean", "qfd_ubound", "qfd_lbound", "opening_fraction_lbound", "opening_fraction_ubound", 'opening_fraction_std', 'opening_fraction_mean',
                         "beam_loc_ratio_lbound",
                         "beam_loc_ratio_ubound", "com_eff_lbound", "com_eff_ubound", "spread_lbound", "spread_ubound", "nft_average"]

    # Extract separated
    df_pref = {k: None for k in list_setting_vars}
    dict_dist_vars = {key: None for key in list_interim_vars}
    for key in df_pref:
        if key in dict_vars_0:
            df_pref[key] = dict_vars_0[key]
            del dict_vars_0[key]
    for key in dict_dist_vars:
        if key in dict_vars_0:
            dict_dist_vars[key] = dict_vars_0[key]
            del dict_vars_0[key]

    for key in list(dict_vars_0_.keys()):
        if key in list_setting_vars:
            del dict_vars_0_[key]
        elif key in list_interim_vars:
            del dict_vars_0_[key]

    for key in dict_vars_0_: dict_vars_0_[key] = [dict_vars_0_[key]] * simulations

    # ------------------------------------------------------------------------------------------------------------------
    # Distribution variables
    # ------------------------------------------------------------------------------------------------------------------

    # lhs_mat = lhs(n=6, samples=simulations, criterion=dict_setting_vars["lhs_criterion"])
    lhs_mat = latin_hypercube_sampling(num_samples=simulations, num_arguments=6)

    # Near field standard deviation
    avg_nft = dict_dist_vars["nft_average"]  # TFM near field temperature - Norm distribution - mean [C]
    std_nft = (1.939 - (np.log(avg_nft) * 0.266)) * avg_nft

    # Convert LHS probabilities to distribution invariants
    # todo: check why this is not used
    com_eff_lbound = dict_dist_vars["com_eff_lbound"]  # Min combustion efficiency [-]  - Linear dist
    com_eff_ubound = dict_dist_vars["com_eff_ubound"]  # Max combustion efficiency [-]  - Linear dist
    comb_lhs = linear_distribution(com_eff_lbound, com_eff_ubound, lhs_mat[:, 0])
    # comb_lhs = np.linspace(com_eff_lbound, com_eff_ubound, simulations+1)
    # comb_lhs += (comb_lhs[1] - comb_lhs[0])
    # comb_lhs = comb_lhs[0:-2]
    # np.random.shuffle(comb_lhs)

    # Fuel load density
    # -----------------
    qfd_std = dict_dist_vars["qfd_std"]  # Fire load density - Gumbel distribution - standard dev [MJ/sq.m]
    qfd_mean = dict_dist_vars["qfd_mean"]  # Fire load density - Gumbel distribution - mean [MJ/sq.m]
    qfd_ubound = dict_dist_vars["qfd_ubound"]  # Fire load density - Gumbel distribution - upper limit [MJ/sq.m]
    qfd_lbound = dict_dist_vars["qfd_lbound"]  # Fire load density - Gumbel distribution - lower limit [MJ/sq.m]
    qfd_loc, qfd_scale = gumbel_parameter_converter(qfd_mean, qfd_std)
    qfd_lhs = gumbel_r_trunc_ppf(qfd_lbound, qfd_ubound, simulations, qfd_loc, qfd_scale) * comb_lhs
    np.random.shuffle(qfd_lhs)
    # DEPRECIATED 14 Aug 2018 - Following codes calculate gumbel parameters for qfd
    # qfd_scale = qfd_std * (6 ** 0.5) / np.pi
    # qfd_loc = qfd_mean - (0.57722 * qfd_scale)
    # qfd_dist = gumbel_r(loc=qfd_loc, scale=qfd_scale)
    # qfd_p_l, qfd_p_u = qfd_dist.cdf(qfd_lbound), qfd_dist.cdf(qfd_ubound)
    # qfd_lhs = gumbel_r(loc=qfd_loc, scale=qfd_scale).ppf(lhs_mat[:, 1]) * comb_lhs

    # Opening fraction factor (glazing fall-out fraction)
    # ---------------------------------------------------
    opening_fraction_std = dict_dist_vars['opening_fraction_std']  # Glazing fall-out fraction - log normal distribution - standard deviation
    opening_fraction_mean = dict_dist_vars['opening_fraction_mean']  # Glazing fall-out fraction - log normal distribution - mean
    opening_fraction_lbound = dict_dist_vars['opening_fraction_lbound']  # Minimum glazing fall-out fraction
    opening_fraction_ubound = dict_dist_vars['opening_fraction_ubound']  # Maximum glazing fall-out fraction
    opening_fraction_mean, opening_fraction_std = lognorm_parameters_true_to_inv(opening_fraction_mean, opening_fraction_std)
    glaz_lhs = 1 - lognorm_trunc_ppf(opening_fraction_lbound, opening_fraction_ubound, simulations, opening_fraction_std, 0, np.exp(opening_fraction_mean))
    np.random.shuffle(glaz_lhs)
    # DEPRECIATED 14 Aug 2018 - Following two lines calculates linear distributed opening fraction
    # glaz_lhs = linear_distribution(glaz_lbound, glaz_ubound, lhs_mat[:, 2])
    # glaz_lhs = 1-lognorm_trunc_ppf(0, 1, 100, 0.2, 0, np.exp(0.2), lhs_mat[:, 2])

    # Beam location
    # -------------
    beam_lbound = dict_dist_vars["beam_loc_ratio_lbound"]  # Min beam location relative to compartment length for TFM [-]  - Linear dist
    beam_ubound = dict_dist_vars["beam_loc_ratio_ubound"]  # Max beam location relative to compartment length for TFM [-]  - Linear dist
    beam_lhs = linear_distribution(beam_lbound, beam_ubound, lhs_mat[:, 3]) * dict_vars_0["room_depth"]
    # beam_lhs = np.linspace(beam_lbound, beam_ubound, simulations+1)
    # beam_lhs += (beam_lhs[1] - beam_lhs[0])
    # beam_lhs = beam_lhs[0:-2] * dict_vars_0["room_depth"]
    # np.random.shuffle(beam_lhs)

    # Fire spread speed (travelling fire)
    # -----------------------------------
    spread_lbound = dict_dist_vars["spread_lbound"]  # Min spread rate for TFM [m/s]  - Linear dist
    spread_ubound = dict_dist_vars["spread_ubound"]  # Max spread rate for TFM [m/s]  - Linear dist
    spread_lhs = linear_distribution(spread_lbound, spread_ubound, lhs_mat[:, 4])
    # spread_lhs = np.linspace(spread_lbound, spread_ubound, simulations+1)
    # spread_lhs += (spread_lhs[1] - spread_lhs[0])
    # spread_lhs = spread_lhs[0:-2]
    # np.random.shuffle(spread_lhs)

    # Near field temperature (travelling fire)
    # ----------------------------------------
    nft_lhs = norm(loc=avg_nft, scale=std_nft).ppf(lhs_mat[:, 5])
    nft_lhs[nft_lhs > 1200] = 1200  # todo: reference?

    # ------------------------------------------------------------------------------------------------------------------
    # Create input kwargs for mc calculation
    # ------------------------------------------------------------------------------------------------------------------

    list_kwargs = []
    for i in range(0, simulations):
        if qfd_lbound > qfd_lhs[i] > qfd_ubound:  # Fire load density is outside limits
            continue
        x_ = dict_vars_0.copy()
        x_.update({"window_open_fraction": glaz_lhs[i],
                   "fire_load_density": qfd_lhs[i],
                   "fire_spread_speed": spread_lhs[i],
                   "beam_position": beam_lhs[i],
                   "nft_ubound": nft_lhs[i],
                   "index": i}, )
        list_kwargs.append(x_)

    dict_vars_0_["window_open_fraction"] = glaz_lhs
    dict_vars_0_["fire_load_density"] = qfd_lhs
    dict_vars_0_["fire_spread_speed"] = spread_lhs
    dict_vars_0_["beam_position"] = beam_lhs
    dict_vars_0_["nft_ubound"] = nft_lhs
    dict_vars_0_["index"] = np.arange(0, simulations, 1, int)

    df_input = df(dict_vars_0_)
    df_input.set_index("index", inplace=True)

    return df_input, df_pref
