# -*- coding: utf-8 -*-
from scipy.interpolate import interp1d
import numpy as np

from sfeprapy.func.fire_travelling import fire as _fire_travelling
from sfeprapy.func.temperature_steel_section import protected_steel_eurocode as _steel_temperature
from sfeprapy.func.temperature_steel_section import protected_steel_eurocode_max_temperature as _steel_temperature_max
from sfeprapy.func.fire_parametric_ec import fire as _fire_param
from sfeprapy.func.fire_parametric_ec_din import fire as _fire_param_ger


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
        fire_mode,
        fire_nft_ubound,
        fire_load_density,
        fire_hrr_density,
        fire_spread_speed,
        fire_duration,
        fire_t_alpha,
        fire_gamma_fi_q,
        beam_position,
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
        solver_max_iter=20,
        solver_thickness_ubound=0.0500,
        solver_thickness_lbound=0.0001,
        solver_tol=1.,
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
    :param fire_nft_ubound: [K], TRAVELLING FIRE, maximum temperature of near field temperature
    :param solver_max_iter: Maximum allowable iteration counts for seeking solution for time equivalence
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
    beam_rho = 7850

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
        nft_max_C=fire_nft_ubound,
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

    if fire_temp[0] == np.nan:
        print('damnit')

    # Solve heat transfer using EC3 correlations
    # SI UNITS FOR INPUTS!
    kwarg_ht_ec = dict(
        time=fire_time,
        temperature_ambient=fire_temp,
        rho_steel=beam_rho,
        area_steel_section=beam_cross_section_area,
        k_protection=protection_k,
        rho_protection=protection_rho,
        c_protection=protection_c,
        thickness_protection=protection_thickness,
        perimeter_protected=protection_protected_perimeter,
        # terminate_when_cooling=True,
        terminate_max_temperature=beam_temperature_goal+5*solver_tol,
    )

    # ============================================
    # GOAL SEEK TO MATCH STEEL FAILURE TEMPERATURE
    # ============================================

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS

    solver_iter_count = 0  # count how many iterations for  the seeking process
    flag_solver_status = False  # flag used to indicate when the seeking is successful

    # Default values
    fire_resistance_equivalence = -1
    solver_steel_temperature_solved = -1
    solver_protection_thickness = -1

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

                solver_iter_count += 1

                # Work out linear equation: f(x) = y = a x + b
                a = (y1 - y2) / (x1 - x2)
                b = y1 - a * x1

                # work out new y based upon interpolated y
                x3 = solver_protection_thickness = (beam_temperature_goal - b) / a
                y3 = solver_steel_temperature_solved = f_(x3)

                if y3 < t1:  # steel temperature is too low, decrease thickness
                    x2 = x3
                    y2 = y3
                elif y3 > t2:  # steel temperature is too high, increase thickness
                    x1 = x3
                    y1 = y3
                else:
                    flag_solver_status = True

                if flag_solver_status or (solver_iter_count >= solver_max_iter):
                    # CALCULATE BEAM FIRE RESISTANCE PERIOD IN ISO 834
                    # ================================================
                    # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.
                    kwarg_ht_ec["time"] = iso834_time
                    kwarg_ht_ec["temperature_ambient"] = iso834_temperature
                    if iso834_temperature[0] == np.nan:
                        print('sh')
                    steel_temperature = _steel_temperature(**kwarg_ht_ec)

                    steel_time = np.concatenate((np.array([0]), iso834_time, np.array([iso834_time[-1]])))
                    steel_temperature = np.concatenate((np.array([-1]), steel_temperature, np.array([1e12])))
                    func_teq = interp1d(steel_temperature, steel_time, kind="linear", bounds_error=False, fill_value=-1)
                    fire_resistance_equivalence = func_teq(beam_temperature_goal)

                    break

        # No solution, thickness upper bound is not thick enough
        elif beam_temperature_goal > y1:
            solver_protection_thickness = x1
            solver_steel_temperature_solved = y1
            fire_resistance_equivalence = 0

        # No solution, thickness lower bound is not thin enough
        elif beam_temperature_goal < y2:
            solver_protection_thickness = x2
            solver_steel_temperature_solved = y2
            fire_resistance_equivalence = 7*24*60*60

    results = {
        "window_open_fraction": window_open_fraction,
        "fire_load_density": fire_load_density,
        "fire_hrr_density": fire_hrr_density,
        "fire_spread_speed": fire_spread_speed,
        'beam_position': beam_position,
        'fire_nft_ubound': fire_nft_ubound,
        'index': index,
        'fire_resistance_equivalence': fire_resistance_equivalence,
        'flag_solver_status': flag_solver_status,
        'fire_type': fire_type,
        'solver_steel_temperature_solved': solver_steel_temperature_solved,
        'solver_protection_thickness': solver_protection_thickness,
        'solver_iter_count': solver_iter_count,
        'opening_factor': opening_factor
    }

    return results
