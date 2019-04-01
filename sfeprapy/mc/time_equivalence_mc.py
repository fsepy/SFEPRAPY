# -*- coding: utf-8 -*-
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats.distributions import norm
from pandas import DataFrame as df
import numpy as np

from sfeprapy.func.fire_travelling import fire as _fire_travelling
from sfeprapy.func.temperature_steel_section import protected_steel_eurocode as _steel_temperature
from sfeprapy.func.temperature_steel_section import protected_steel_eurocode_max_temperature as _steel_temperature_max
from sfeprapy.func.fire_parametric_ec import fire as _fire_param
from sfeprapy.func.fire_parametric_ec_din import fire as _fire_param_ger


def lognorm_parameters_true_to_inv(miu, sigma):
    """
    NAME: lognorm_parameters_true_to_inv
    VERSION: 0.0.1
    AUTHOR: Yan Fu, Ruben V. Coline
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
    :param sigma: float, standard deviation of log normal distribution (for ln(x))
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
    """
    NAME: gumbel_parameter_converter
    VERSION: 0.0.1
    AUTHOR: Yan Fu, Ruben V. Coline
    DATE: 27 Sept 2018
    DESCRIPTION:
    This function is used in conjunction of the scipy.stats.gumbel distribution, converts mean and standard deviation
    (sigma) of samples x to location and scale which is used in scipy.stats.gumbel_r function.

    :param miu: mean value of samples x
    :param sigma: standard deviation of samples x
    :return: location and scale

    EXAMPLE:

    """

    # parameters Gumbel W&S
    alpha = 1.282 / sigma
    u = miu - 0.5772 / alpha

    # parameters Gumbel scipy
    scale = 1 / alpha
    loc = u

    return loc, scale


def gumbel_parameter_converter2(loc, scale):

    # parameters Gumbel W&S
    # alpha = 1.282 / sigma
    # u = miu - 0.5772 / alpha

    sigma = 1.282 / (1 / scale)
    miu = loc + 0.5772 / (1 / scale)

    # parameters Gumbel scipy
    # scale = 1 / alpha
    # loc = u

    return miu, sigma


def gumbel_r_trunc_ppf(a, b, n_rv, loc, scale, cdf_y=None):
    """
    NAME: gumbel_r_trunc_ppf
    VERSION: 0.0.1
    AUTHOR: Yan Fu
    DATE: 27 Sept 2018
    DESCRIPTION:
    Produces evenly sampled random variables based on gumbel distribution (tail to the x+ direction, i.e. median greater
    than mean). Truncation is possible via variables 'a' and 'b'. i.e. inversed cumulative density function f(x), x will
    be sampled in linear space ranging from 'a' to 'b'. Then f(x) is returned. Additionally, if x is defined 'cdf_y'
    then f(cdf_y) is returned.

    PARAMETERS:
    :param a: lower bound of truncation
    :param b: upper bound of truncation
    :param n_rv: number of random variables to be sampled, equal to the length of the returned array
    :param loc: location of the distribution
    :param scale: scale of the distribution
    :param cdf_y: array ranging with range (0, 1)
    :return sampled: sampled random variables

    USAGE:
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> loc, scale = gumbel_parameter_converter(600, 180)
    >>> rv = gumbel_r_trunc_ppf(0, 1800, 10, loc, scale)
    >>> print(np.round(rv, 0))
    [   0.  408.  462.  506.  548.  594.  646.  713.  819. 1800.]

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
    x = np.linspace(a, b, int(n_rv) + 1, endpoint=False)
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


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    loc, scale = gumbel_parameter_converter(600, 180)
    rv = gumbel_r_trunc_ppf(0, 1800, 10, loc, scale)
    print(np.round(rv, 0))
    sns.distplot(rv)
    plt.show()


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


def calc_time_equivalence_worker(arg):
    kwargs, q = arg
    result = calc_time_equivalence(**kwargs)
    q.put("index: {}".format(kwargs["index"]))
    return result


def calc_time_equivalence(
        time_step,
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
        fire_t_alpha,
        fire_gamma_fi_q,
        beam_position,
        beam_rho,
        beam_c,
        beam_cross_section_area,
        beam_temperature_goal,
        beam_loc_z,
        protection_k,
        protection_rho,
        protection_c,
        protection_thickness,
        protection_protected_perimeter,
        iso834_time,
        iso834_temperature,
        index,
        fire_mode=3,
        nft_ubound=1200,
        seek_ubound_iter=20,
        solver_thickness_ubound=0.0500,
        solver_thickness_lbound=0.0001,
        solver_tol=1.,
        return_mode=0,
        **kwargs
):
    """
    NAME: calc_time_equivalence
    AUTHOR: Ian Fu
    DATE: 11 March 2019
    DESCRIPTION:
    Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param time_step: [s], time step used for numerical calculation
    :param time_start: [s], simulation starting time
    :param time_limiting: [-], PARAMETRIC FIRE, see parametric fire function for details
    :param window_height: [m], weighted window opening height
    :param window_width: [m], total window opening width
    :param window_open_fraction: [-], a factor is multiplied with the given total window opening area
    :param room_breadth: [m], room breadth (shorter direction of the floor plan)
    :param room_depth: [m], room depth (longer direction of the floor plan)
    :param room_height: [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param room_wall_thermal_inertia: [J m-2 K-1 s-1/2], thermal inertia of room lining material
    :param fire_load_density: [MJ m-2], fire load per unit area
    :param fire_hrr_density: [MW m-2], fire maximum release rate per unit area
    :param fire_spread_speed: [m s-1], TRAVELLING FIRE, fire spread speed
    :param fire_duration: [s], simulation time
    :param beam_position: [s], beam location
    :param beam_rho: [kg m-3], density of the steel beam element
    :param beam_c: [?], specific heat of the steel element
    :param beam_cross_section_area: [m2], the steel beam element cross section area
    :param beam_temperature_goal: [K], steel beam element expected failure temperature
    :param protection_k: steel beam element protection material thermal conductivity
    :param protection_rho: steel beam element protection material density
    :param protection_c: steel beam element protection material specific heat
    :param protection_thickness: steel beam element protection material thickness
    :param protection_protected_perimeter: [m], steel beam element protection material perimeter
    :param iso834_time: [s], the time (array) component of ISO 834 fire curve
    :param iso834_temperature: [K], the temperature (array) component of ISO 834 fire curve
    :param nft_ubound: [K], TRAVELLING FIRE, maximum temperature of near field temperature
    :param seek_ubound_iter: Maximum allowable iteration counts for seeking solution for time equivalence
    :param solver_thickness_ubound: [m], protection layer thickness upper bound initial condition for solving time equivalence
    :param solver_thickness_lbound: [m], protection layer thickness lower bound initial condition for solving time equivalence
    :param solver_tol: [K], tolerance for solving time equivalence
    :param index: will be returned for indicating the index of the current iteration (was used for multiprocessing)
    :param fire_mode: 0 - parametric, 1 - travelling, 2 - ger parametric, 3 - (0 & 1), 4 (1 & 2)
    :param return_mode: 0 - minimal for teq; 1 - all variables; and 2 - all variables packed in a dict
    :param kwargs: will be discarded
    :return:
    EXAMPLE:
    """

    # DO NOT CHANGE, LEGACY PARAMETERS
    # Used to define fire curve start time, depreciated on 11/03/2019 after introducing the DIN annex ec parametric
    # fire.
    time_start = 0

    # PERMEABLE AND INPUT CHECKS

    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    room_depth, room_breadth = max(room_depth, room_breadth), min(room_depth, room_breadth)

    # Total window opening area
    window_area = window_height * window_width * window_open_fraction

    # Room floor area
    room_floor_area = room_breadth * room_depth

    # Room internal surface area, total, including window openings
    room_total_area = (2 * room_floor_area) + ((room_breadth + room_depth) * 2 * room_height)

    # Fire load density related to the total surface area A_t
    fire_load_density_total = fire_load_density * room_floor_area / room_total_area

    # Opening factor
    opening_factor = window_area * np.sqrt(window_height) / room_total_area

    # Spread speed - Does the fire spread to involve the full compartment?
    fire_spread_entire_room_time = room_depth / fire_spread_speed
    burn_out_time = max([fire_load_density / fire_hrr_density, 900.])

    fire_time = np.arange(time_start, fire_duration + time_step, time_step, dtype=float)

    kwargs_fire_0_paramec = dict(
        t=fire_time,
        A_t=room_total_area,
        A_f=room_floor_area,
        A_v=window_area,
        h_eq=window_height,
        q_fd=fire_load_density * 1e6,
        lambda_=room_wall_thermal_inertia ** 2,
        rho=1,
        c=1,
        t_lim=time_limiting,
        temperature_initial=20 + 273.15,
    )
    kwargs_fire_1_travel = dict(
        t=fire_time,
        fire_load_density_MJm2=fire_load_density,
        heat_release_rate_density_MWm2=fire_hrr_density,
        length_compartment_m=room_depth,
        width_compartment_m=room_breadth,
        fire_spread_rate_ms=fire_spread_speed,
        height_fuel_to_element_m=beam_loc_z,
        length_element_to_fire_origin_m=beam_position,
        nft_max_C=nft_ubound,
        win_width_m=window_width,
        win_height_m=window_height,
        open_fract=window_open_fraction,
    )
    kwargs_fire_2_paramdin = dict(
        t_array_s=fire_time,
        A_w_m2=window_area,
        h_w_m2=window_height,
        A_t_m2=room_total_area,
        A_f_m2=room_floor_area,
        t_alpha_s=fire_t_alpha,
        b_Jm2s05K=room_wall_thermal_inertia,
        q_x_d_MJm2=fire_load_density,
        gamma_fi_Q=fire_gamma_fi_q
    )

    if fire_mode == 0:  # enforced to ec parametric fire

        # opening_factor = min(0.2, max(0.02, opening_factor))  # force opening factor fall in the boundary
    
        fire_temp = _fire_param(**kwargs_fire_0_paramec)
        fire_type = 0  # parametric fire

    elif fire_mode == 1:  # enforced to travelling fire

        fire_temp = _fire_travelling(**kwargs_fire_1_travel) + 273.15
        fire_type = 1  # travelling fire

    elif fire_mode == 2: # enforced to german parametric fire

        fire_temp = _fire_param_ger(**kwargs_fire_2_paramdin)
        fire_type = 2   # german parametric

    elif fire_mode == 3:  # enforced to ec parametric + travelling
        if fire_spread_entire_room_time < burn_out_time and 0.02 < opening_factor <= 0.2 and 50 <= fire_load_density_total <= 1000:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

            fire_temp = _fire_param(**kwargs_fire_0_paramec)
            fire_type = 0  # parametric fire

        else:  # Otherwise, it is a travelling fire

            fire_temp = _fire_travelling(**kwargs_fire_1_travel) + 273.15
            fire_type = 1  # travelling fire

    elif fire_mode == 4:  # enforced to german parametric + travelling

        if fire_spread_entire_room_time < burn_out_time and 0.125 <= (window_area / room_floor_area) <= 0.5:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

            fire_temp = _fire_param_ger(**kwargs_fire_2_paramdin)
            fire_type = 2  # german parametric

        else:  # Otherwise, it is a travelling fire

            fire_temp = _fire_travelling(**kwargs_fire_1_travel) + 273.15
            fire_type = 1  # travelling fire

    # Solve heat transfer using EC3 correlations
    # SI UNITS FOR INPUTS!
    kwarg_ht_ec = dict(
        time=fire_time,
        temperature_ambient=fire_temp,
        rho_steel=beam_rho,
        c_steel_T=beam_c,
        area_steel_section=beam_cross_section_area,
        k_protection=protection_k,
        rho_protection=protection_rho,
        c_protection=protection_c,
        thickness_protection=protection_thickness,
        perimeter_protected=protection_protected_perimeter,
        # terminate_when_cooling=True,
        # terminate_max_temperature=beam_temperature_goal+5*solver_tol,
    )

    # Find maximum steel temperature for the static protection layer thickness
    if protection_thickness > 0:
        temperature_steel_ubound = _steel_temperature_max(**kwarg_ht_ec)
    else:
        temperature_steel_ubound = -1

    # ============================================
    # GOAL SEEK TO MATCH STEEL FAILURE TEMPERATURE
    # ============================================

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS

    solver_iteration_count = 0  # count how many iterations for  the seeking process
    flag_solver_status = False  # flag used to indicate when the seeking is successful

    # Default values
    fire_resistance_equivalence = -1
    solver_steel_temperature_solved = -1
    solver_thickness = -1

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(fire_time, fire_temp)

    if beam_temperature_goal > 0:  # check seeking temperature, opt out if less than 0

        # Minimise f(x) - θ
        # where:
        #   f(x)    is the steel maximum temperature.
        #   x       is the thickness,
        #   θ       is the steel temperature goal

        def f_(x):
            kwarg_ht_ec["thickness_protection"] = x  # NOTE! variable out function scope

            T_ = _steel_temperature_max(**kwarg_ht_ec, terminate_check_wait_time = fire_time[np.argmax(fire_temp)])  # NOTE! variable out function scope
            return T_

        # Check whether there is a solution within predefined protection thickness boundaries
        x1, x2 = solver_thickness_lbound, solver_thickness_ubound
        y1, y2 = f_(x1), f_(x2)
        t1, t2 = beam_temperature_goal - solver_tol, beam_temperature_goal + solver_tol

        if (y2 - solver_tol) <= beam_temperature_goal <= (y1 + solver_tol):

            while True:

                solver_iteration_count += 1

                # Work out linear equation: f(x) = y = a x + b
                # y1 = a x1 + b
                # y2 = a x2 + b
                # y1 - y2 = a (x1 - x2)
                # a = (y1 - y2) / (x1 - x2)
                # b = y1 - a x1
                a = (y1 - y2) / (x1 - x2)
                b = y1 - a * x1

                # work out new y based upon interpolated y
                x3 = solver_thickness = (beam_temperature_goal - b) / a
                y3 = solver_steel_temperature_solved = f_(x3)

                if y3 < t1:  # steel temperature is too low, decrease thickness
                    x2 = x3
                    y2 = y3
                elif y3 > t2:  # steel temperature is too high, increase thickness
                    x1 = x3
                    y1 = y3
                else:
                    flag_solver_status = True

                if flag_solver_status or (solver_iteration_count >= seek_ubound_iter):
                    # CALCULATE BEAM FIRE RESISTANCE PERIOD IN ISO 834
                    # ================================================
                    # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.
                    kwarg_ht_ec["time"] = iso834_time
                    kwarg_ht_ec["temperature_ambient"] = iso834_temperature
                    steel_temperature = _steel_temperature(**kwarg_ht_ec)

                    steel_time = np.concatenate((np.array([0]), iso834_time, np.array([iso834_time[-1]])))
                    steel_temperature = np.concatenate((np.array([-1]), steel_temperature, np.array([1e12])))
                    func_teq = interp1d(steel_temperature, steel_time, kind="linear", bounds_error=False, fill_value=-1)
                    fire_resistance_equivalence = func_teq(beam_temperature_goal)

                    break

                # DEPRECIATED 26/03/2019
                #
                # solver_iteration_count += 1
                #
                # solver_thickness = np.average([solver_thickness_ubound, solver_thickness_lbound])
                #
                # kwarg_ht_ec["thickness_protection"] = solver_thickness
                # t_, T_, d_ = _steel_temperature(**kwarg_ht_ec)
                # solver_steel_temperature_solved = np.max(T_)
                # y_diff_seek = solver_steel_temperature_solved - beam_temperature_goal
                # if abs(y_diff_seek) <= solver_tol:
                #     flag_solver_status = True
                # elif solver_steel_temperature_solved > beam_temperature_goal:  # steel too hot, increase protect thickness
                #     solver_thickness_lbound = solver_thickness
                # else:  # steel is too cold, increase intrumescent paint thickness
                #     solver_thickness_ubound = solver_thickness

        # No solution, thickness upper bound is not thick enough
        elif beam_temperature_goal > y1:
            solver_thickness = x1
            solver_steel_temperature_solved = y1
            fire_resistance_equivalence = 0

        # No solution, thickness lower bound is not thin enough
        elif beam_temperature_goal < y2:
            solver_thickness = x2
            solver_steel_temperature_solved = y2
            fire_resistance_equivalence = 7*24*60*60

    # r_dict = {'room_wall_thermal_inertia': room_wall_thermal_inertia,
    #           'fire_load_density': fire_load_density,
    #           'fire_hrr_density': fire_hrr_density,
    #           'fire_spread_speed': fire_spread_speed,
    #           'fire_duration': fire_duration,
    #           'beam_position': beam_position,
    #           'beam_rho': beam_rho,
    #           'beam_c': beam_c,
    #           'beam_cross_section_area': beam_cross_section_area,
    #           'beam_temperature_goal': beam_temperature_goal,
    #           'protection_k': protection_k,
    #           'protection_rho': protection_rho,
    #           'protection_c': protection_c,
    #           'protection_thickness': protection_thickness,
    #           'protection_protected_perimeter': protection_protected_perimeter,
    #           'iso834_time': iso834_time,
    #           'iso834_temperature': iso834_temperature,
    #           'nft_ubound': nft_ubound,
    #           'seek_ubound_iter': seek_ubound_iter,
    #           'solver_thickness_ubound': solver_thickness_ubound,
    #           'solver_thickness_lbound': solver_thickness_lbound,
    #           'solver_tol': solver_tol,
    #           'index': index,
    #           'fire_resistance_equivalence': fire_resistance_equivalence,
    #           'flag_solver_status': flag_solver_status,
    #           'fire_type': fire_type,
    #           'solver_steel_temperature_solved': solver_steel_temperature_solved,
    #           'solver_thickness': solver_thickness,
    #           'solver_iteration_count': solver_iteration_count,
    #           'temperature_steel_ubound': temperature_steel_ubound,
    #           'fire_time': fire_time,
    #           'fire_temp': fire_temp}

    return time_step, time_start, time_limiting, window_height, window_width, window_open_fraction, room_breadth, room_depth, room_height, room_wall_thermal_inertia, fire_load_density, fire_hrr_density, fire_spread_speed, fire_duration, beam_position, beam_rho, 0, beam_cross_section_area, beam_temperature_goal, protection_k, protection_rho, protection_c, protection_thickness, protection_protected_perimeter, 0, 0, nft_ubound, seek_ubound_iter, solver_thickness_ubound, solver_thickness_lbound, solver_tol, index, fire_resistance_equivalence, flag_solver_status, fire_type, solver_steel_temperature_solved, solver_thickness, solver_iteration_count, temperature_steel_ubound, 0, 0, opening_factor
    # return time_step, time_start, time_limiting, window_height, window_width, window_open_fraction, room_breadth, room_depth, room_height, room_wall_thermal_inertia, fire_load_density, fire_hrr_density, fire_spread_speed, fire_duration, beam_position, beam_rho, beam_c, beam_cross_section_area, beam_temperature_goal, protection_k, protection_rho, protection_c, protection_thickness, protection_protected_perimeter, iso834_time, iso834_temperature, nft_ubound, seek_ubound_iter, solver_thickness_ubound, solver_thickness_lbound, solver_tol, index, fire_resistance_equivalence, flag_solver_status, fire_type, solver_steel_temperature_solved, solver_thickness, solver_iteration_count, temperature_steel_ubound, fire_time, fire_temp, opening_factor


def mc_inputs_generator_worker(arg):
    kwargs, q = arg
    result = calc_time_equivalence(**kwargs)
    q.put("index")
    return result


def mc_inputs_generator(
        simulations,
        room_depth,
        room_opening_fraction_lbound,
        room_opening_fraction_ubound,
        room_opening_fraction_std,
        room_opening_fraction_mean,
        room_opening_permanent_fraction,
        fire_qfd_std,
        fire_qfd_mean,
        fire_qfd_ubound,
        fire_qfd_lbound,
        fire_com_eff_lbound,
        fire_com_eff_ubound,
        fire_spread_lbound,
        fire_spread_ubound,
        fire_nft_mean,
        beam_loc_ratio_lbound,
        beam_loc_ratio_ubound,
        **kwargs
):

    # linear_distribution = lambda min, max, prob: ((max - min) * prob) + min
    def linear_distribution(min_, max_, rv_):
        return ((max_ - min_) * rv_) + min_

    # ==================================================================================================================
    # CHECKS
    # ==================================================================================================================

    # Fire duration has to be as long as travelling fire to travel through the entire floor
    # time_end = np.max([time_end, room_depth / fire_spread_lbound])

    # ==================================================================================================================
    # Distribution variables
    # ==================================================================================================================

    if simulations > 2:

        # lhs_mat = lhs(n=6, samples=simulations, criterion=dict_setting_vars["lhs_criterion"])
        lhs_mat = latin_hypercube_sampling(num_samples=simulations, num_arguments=6)

        # Near field standard deviation
        # fire_nft_mean = fire_nft_mean  # TFM near field temperature - Norm distribution - mean [C]
        std_nft = (1.939 - (np.log(fire_nft_mean) * 0.266)) * fire_nft_mean

        # Convert LHS probabilities to distribution invariants
        comb_lhs = linear_distribution(fire_com_eff_lbound, fire_com_eff_ubound, lhs_mat[:, 0])

        # Fuel load density
        # -----------------
        qfd_loc, qfd_scale = gumbel_parameter_converter(fire_qfd_mean, fire_qfd_std)
        qfd_lhs = gumbel_r_trunc_ppf(fire_qfd_lbound, fire_qfd_ubound, simulations, qfd_loc, qfd_scale) * comb_lhs
        np.random.shuffle(qfd_lhs)

        # CAR PARK FIRE LOAD, NEW 06/03/2019
        # qfd_lhs = stats.norm.ppf(np.linspace(0,1,simulations+2)[1:-1],*(251.90611757771737, 7.015945381175767))

        # DEPRECIATED 14 Aug 2018 - Following codes calculate gumbel parameters for qfd
        # qfd_scale = qfd_std * (6 ** 0.5) / np.pi
        # qfd_loc = qfd_mean - (0.57722 * qfd_scale)
        # qfd_dist = gumbel_r(loc=qfd_loc, scale=qfd_scale)
        # qfd_p_l, qfd_p_u = qfd_dist.cdf(qfd_lbound), qfd_dist.cdf(qfd_ubound)
        # qfd_lhs = gumbel_r(loc=qfd_loc, scale=qfd_scale).ppf(lhs_mat[:, 1]) * comb_lhs

        # Opening fraction factor (glazing fall-out fraction)
        # ---------------------------------------------------
        opening_fraction_mean_conv, opening_fraction_std_conv = lognorm_parameters_true_to_inv(room_opening_fraction_mean,
                                                                                               room_opening_fraction_std)
        glaz_lhs = 1 - lognorm_trunc_ppf(room_opening_fraction_lbound, room_opening_fraction_ubound, simulations,
                                         opening_fraction_std_conv, 0, np.exp(opening_fraction_mean_conv))
        np.random.shuffle(glaz_lhs)

        # count for permanent opening area
        glaz_lhs = [i * (1 - room_opening_permanent_fraction) + room_opening_permanent_fraction for i in glaz_lhs]

        # Beam location
        # -------------
        beam_lhs = linear_distribution(beam_loc_ratio_lbound, beam_loc_ratio_ubound, lhs_mat[:, 3]) * room_depth

        # Fire spread speed (travelling fire)
        # -----------------------------------
        spread_lhs = linear_distribution(fire_spread_lbound, fire_spread_ubound, lhs_mat[:, 4])

        # Near field temperature (travelling fire)
        # ----------------------------------------
        nft_lhs = norm(loc=fire_nft_mean, scale=std_nft).ppf(lhs_mat[:, 5])
        nft_lhs[nft_lhs > 1200] = 1200  # todo: reference?

    # ------------------------------------------------------------------------------------------------------------------
    # Create input kwargs for mc calculation
    # ------------------------------------------------------------------------------------------------------------------

    # if index:
    #     dict_input_kwargs_static['index'] = np.arange(0, simulations)

    list_mc_input_kwargs = []

    if simulations <= 2:
        x_ = {"window_open_fraction": room_opening_fraction_mean,
                   "fire_load_density": fire_qfd_mean,
                   "fire_spread_speed": (fire_spread_ubound + fire_spread_lbound) / 2,
                   "beam_position": room_depth*0.75,
                   "nft_ubound": fire_nft_mean,
                   "index": 0}
        list_mc_input_kwargs.append(x_)
    else:
        for i in range(0, int(simulations)):
            x_ = {"window_open_fraction": glaz_lhs[i],
                       "fire_load_density": qfd_lhs[i],
                       "fire_spread_speed": spread_lhs[i],
                       "beam_position": beam_lhs[i],
                       "nft_ubound": nft_lhs[i],
                       "index": i}
            list_mc_input_kwargs.append(x_)

    df_input = df(list_mc_input_kwargs)
    df_input.set_index("index", inplace=True)

    return df_input
