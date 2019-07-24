# -*- coding: utf-8 -*-
import copy
import warnings
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sfeprapy.func.fire_parametric_ec import fire as _fire_param
from sfeprapy.func.fire_parametric_ec_din import fire as _fire_param_ger
from sfeprapy.func.fire_travelling import fire as fire_travelling
from sfeprapy.func.heat_transfer_protected_steel_ec import protected_steel_eurocode as _steel_temperature
from sfeprapy.func.heat_transfer_protected_steel_ec import \
    protected_steel_eurocode_max_temperature as _steel_temperature_max

warnings.filterwarnings('ignore')


def _fire_travelling(**kwargs):
    if isinstance(kwargs['beam_location_length_m'], list) or isinstance(kwargs['beam_location_length_m'], np.ndarray):

        kwarg_ht_ec = dict(
            time=kwargs['t'],
            temperature_ambient=None,
            rho_steel=7850,
            area_steel_section=0.017,
            k_protection=0.2,
            rho_protection=800,
            c_protection=1700,
            thickness_protection=0.005,
            perimeter_protected=2.14,
        )

        temperature_steel_list = list()
        temperature_gas_list = fire_travelling(**kwargs)

        for temperature in temperature_gas_list:
            kwarg_ht_ec['temperature_ambient'] = temperature + 273.15
            temperature_steel_list.append(_steel_temperature_max(**kwarg_ht_ec))

        return temperature_gas_list[np.argmax(temperature_steel_list)]+273.15, \
               kwargs['beam_location_length_m'][[np.argmax(temperature_steel_list)]][0]

    elif isinstance(kwargs['beam_location_length_m'], float):

        return fire_travelling(**kwargs)+273.15, \
               kwargs['beam_location_length_m']


def calc_time_equivalence_worker(arg):
    kwargs, q = arg
    result = calc_time_equivalence(**kwargs)
    q.put("index: {}".format(kwargs["index"]))
    return result


def calc_time_equivalence(
        case_name,
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
        fire_combustion_efficiency,
        fire_hrr_density,
        fire_spread_speed,
        fire_duration,
        fire_t_alpha,
        fire_gamma_fi_q,
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
        probability_weight:int,
        index,
        beam_position: Union[np.ndarray, list, float] = -1,
        solver_max_iter=20,
        solver_thickness_ubound=0.0500,
        solver_thickness_lbound=0.0001,
        solver_tol=1.,
        **_
):
    """
    NAME: grouped_a_b
    AUTHOR: Ian Fu
    DATE: 11 March 2019
    DESCRIPTION:
    Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param time_step: [s], time step used for numerical calculation
    :param time_start: [s], simulation starting time
    :param time_limiting: [s], PARAMETRIC FIRE, see parametric fire function for details
    :param window_height: [m], weighted window opening height
    :param window_width: [m], total window opening width
    :param window_open_fraction: [-], a factor is multiplied with the given total window opening area
    :param room_breadth: [m], room breadth (shorter direction of the floor plan)
    :param room_depth: [m], room depth (longer direction of the floor plan)
    :param room_height: [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param room_wall_thermal_inertia: [J m-2 K-1 s-1/2], thermal inertia of room lining material
    :param fire_load_density_deducted: [MJ m-2], fire load per unit area
    :param fire_hrr_density: [MW m-2], fire maximum release rate per unit area
    :param fire_spread_speed: [m s-1], TRAVELLING FIRE, fire spread speed
    :param fire_duration: [s], simulation time
    :param beam_position: [s], beam location, will be solved for the worst case if less than 0.
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
    :return:
    EXAMPLE:
    """

    # DO NOT CHANGE, LEGACY PARAMETERS
    # Used to define fire curve start time, depreciated on 11/03/2019 after introducing the DIN annex ec parametric
    # fire.
    time_start = 0
    beam_rho = 7850

    # PERMEABLE AND INPUT CHECKS

    fire_load_density_deducted = fire_load_density * fire_combustion_efficiency

    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    room_depth, room_breadth = max(room_depth, room_breadth), min(room_depth, room_breadth)

    # Total window opening area
    window_area = window_height * window_width * window_open_fraction

    # Room floor area
    room_floor_area = room_breadth * room_depth

    # Room internal surface area, total, including window openings
    room_total_area = (2 * room_floor_area) + ((room_breadth + room_depth) * 2 * room_height)

    # Fire load density related to the total surface area A_t
    fire_load_density_total = fire_load_density_deducted * room_floor_area / room_total_area

    # Opening factor
    opening_factor = window_area * np.sqrt(window_height) / room_total_area

    # Spread speed - Does the fire spread to involve the full compartment?
    fire_spread_entire_room_time = room_depth / fire_spread_speed
    burn_out_time = max([fire_load_density_deducted / fire_hrr_density, 900.])

    # if beam_position is not provided, solve for the worst case
    if beam_position < 0:
        beam_position = np.linspace(0.5*room_depth, room_depth, 7)[1:-1]

    fire_time = np.arange(time_start, fire_duration + time_step, time_step, dtype=float)

    kwargs_fire_0_paramec = dict(
        t=fire_time,
        A_t=room_total_area,
        A_f=room_floor_area,
        A_v=window_area,
        h_eq=window_height,
        q_fd=fire_load_density_deducted * 1e6,
        lambda_=room_wall_thermal_inertia ** 2,
        rho=1,
        c=1,
        t_lim=time_limiting,
        temperature_initial=20 + 273.15,
    )

    kwargs_fire_1_travel = dict(
        t=fire_time,
        fire_load_density_MJm2=fire_load_density_deducted,
        fire_hrr_density_MWm2=fire_hrr_density,
        room_length_m=room_depth,
        room_width_m=room_breadth,
        fire_spread_rate_ms=fire_spread_speed,
        beam_location_height_m=beam_loc_z,
        beam_location_length_m=beam_position,
        fire_nft_limit_c=fire_nft_ubound,
        opening_width_m=window_width,
        opening_height_m=window_height,
        opening_fraction=window_open_fraction,
    )

    kwargs_fire_2_paramdin = dict(
        t_array_s=fire_time,
        A_w_m2=window_area,
        h_w_m2=window_height,
        A_t_m2=room_total_area,
        A_f_m2=room_floor_area,
        t_alpha_s=fire_t_alpha,
        b_Jm2s05K=room_wall_thermal_inertia,
        q_x_d_MJm2=fire_load_density_deducted,
        gamma_fi_Q=fire_gamma_fi_q
    )
    if fire_mode == 0:  # enforced to ec parametric fire

        # opening_factor = min(0.2, max(0.02, opening_factor))  # force opening factor fall in the boundary
    
        fire_temperature = _fire_param(**kwargs_fire_0_paramec)
        fire_type = 0  # parametric fire

    elif fire_mode == 1:  # enforced to travelling fire

        fire_temperature, beam_position = _fire_travelling(**kwargs_fire_1_travel)
        fire_type = 1  # travelling fire

    elif fire_mode == 2:  # enforced to german parametric fire

        fire_temperature = _fire_param_ger(**kwargs_fire_2_paramdin)
        fire_type = 2   # german parametric

    elif fire_mode == 3:  # enforced to ec parametric + travelling

        # print(fire_spread_entire_room_time < burn_out_time, 0.02 < opening_factor <= 0.2, 50 <= fire_load_density_total <= 1000)

        if fire_spread_entire_room_time < burn_out_time and 0.02 < opening_factor <= 0.2 and 50 <= fire_load_density_total <= 1000:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

            fire_temperature = _fire_param(**kwargs_fire_0_paramec)
            fire_type = 0  # parametric fire

        else:  # Otherwise, it is a travelling fire

            fire_temperature, beam_position = _fire_travelling(**kwargs_fire_1_travel)
            fire_type = 1  # travelling fire

    elif fire_mode == 4:  # enforced to german parametric + travelling

        if fire_spread_entire_room_time < burn_out_time and 0.125 <= (window_area / room_floor_area) <= 0.5:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

            fire_temperature = _fire_param_ger(**kwargs_fire_2_paramdin)
            fire_type = 2  # german parametric

        else:  # Otherwise, it is a travelling fire

            fire_temperature, beam_position = _fire_travelling(**kwargs_fire_1_travel)
            fire_type = 1  # travelling fire

    else:

        raise ValueError('Unknown fire mode {fire_mode}.'.format(fire_mode=fire_mode))

    if fire_temperature[0] == np.nan:

        warnings.warn('Abnormal fire temperature.')
        print(fire_temperature)

    if not isinstance(beam_position, float):
        beam_position = np.nan

    # ============================================
    # GOAL SEEK TO MATCH STEEL FAILURE TEMPERATURE
    # ============================================

    # HELPER FUNCTION: CALCULATE TEQ FOR A GIVEN PROTECTION THICKNESS

    def find_time_equivalence_for_a_thickness(thickness_protection_, kwarg_ht_ec_, iso834_time_, iso834_temperature_, func_find_steel_temperature_, temperature_goal_):

        # CALCULATE BEAM FIRE RESISTANCE PERIOD IN ISO 834
        # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.

        kwarg_ht_ec_["time"] = iso834_time_
        kwarg_ht_ec_["temperature_ambient"] = iso834_temperature_
        kwarg_ht_ec_["thickness_protection"] = thickness_protection_
        if iso834_temperature_[0] == np.nan:
            print('sh')
        steel_temperature = func_find_steel_temperature_(**kwarg_ht_ec_)

        steel_time = np.concatenate((np.array([0]), iso834_time, np.array([iso834_time[-1]])))
        steel_temperature = np.concatenate((np.array([-1]), steel_temperature, np.array([1e12])))
        func_teq = interp1d(steel_temperature, steel_time, kind="linear", bounds_error=False, fill_value=-1)
        solver_time_equivalence_solved = func_teq(temperature_goal_)

        return solver_time_equivalence_solved

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS

    solver_iter_count = 0  # count how many iterations for  the seeking process
    solver_convergence_status = False  # flag used to indicate when the seeking is successful

    # Default values
    solver_time_equivalence_solved = -1
    solver_steel_temperature_solved = -1
    solver_protection_thickness = -1

    if beam_temperature_goal > 0:  # check seeking temperature, opt out if less than 0

        # Solve heat transfer using EC3 correlations
        # SI UNITS FOR INPUTS!
        kwarg_ht_ec = dict(
            time=fire_time,
            temperature_ambient=fire_temperature,
            rho_steel=beam_rho,
            area_steel_section=beam_cross_section_area,
            k_protection=protection_k,
            rho_protection=protection_rho,
            c_protection=protection_c,
            perimeter_protected=protection_protected_perimeter,
            terminate_max_temperature=beam_temperature_goal + 5 * solver_tol,
        )

        # SOLVER START FROM HERE
        # ======================

        # Minimise f(x) - θ
        # where:
        #   f(x)    is the steel maximum temperature.
        #   x       is the thickness,
        #   θ       is the steel temperature goal

        time_at_max_temperature = fire_time[np.argmax(fire_temperature)]

        def f_(x, terminate_check_wait_time):
            kwarg_ht_ec["thickness_protection"] = x
            T_ = _steel_temperature_max(**kwarg_ht_ec, terminate_check_wait_time=terminate_check_wait_time)
            return T_

        # Check whether there is a solution within predefined protection thickness boundaries
        x1, x2 = solver_thickness_lbound, solver_thickness_ubound
        y1, y2 = f_(x1, time_at_max_temperature), f_(x2, time_at_max_temperature)
        t1, t2 = beam_temperature_goal - solver_tol, beam_temperature_goal + solver_tol

        # if (y2 - solver_tol) <= beam_temperature_goal <= (y1 + solver_tol):
        if y2 <= beam_temperature_goal <= y1:

            while True:

                solver_iter_count += 1

                # Work out linear equation: f(x) = y = a x + b
                a = (y1 - y2) / (x1 - x2)
                b = y1 - a * x1

                # work out new y based upon interpolated y
                x3 = solver_protection_thickness = (beam_temperature_goal - b) / a
                y3 = solver_steel_temperature_solved = f_(x3, time_at_max_temperature)

                if x1 < 0 or x2 < 0 or x3 < 0:
                    print('check')

                if y3 < t1:  # steel temperature is too low, decrease thickness
                    x2 = x3
                    y2 = y3
                elif y3 > t2:  # steel temperature is too high, increase thickness
                    x1 = x3
                    y1 = y3
                else:
                    solver_convergence_status = True

                if solver_convergence_status:

                    # CALCULATE BEAM FIRE RESISTANCE PERIOD IN ISO 834
                    # ================================================
                    # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.

                    kwarg_ht_ec["time"] = iso834_time
                    kwarg_ht_ec["temperature_ambient"] = iso834_temperature
                    kwarg_ht_ec["thickness_protection"] = x3
                    steel_temperature = _steel_temperature(**kwarg_ht_ec)

                    steel_time = np.concatenate((np.array([0]), iso834_time, np.array([iso834_time[-1]])))
                    steel_temperature = np.concatenate((np.array([-1]), steel_temperature, np.array([1e12])))
                    func_teq = interp1d(steel_temperature, steel_time, kind="linear", bounds_error=False, fill_value=-1)
                    solver_time_equivalence_solved = func_teq(beam_temperature_goal)

                    break

                elif solver_iter_count >= solver_max_iter:  # Terminate if maximum solving iteration is reached

                    if beam_temperature_goal > y3:
                        solver_time_equivalence_solved = -np.inf
                    elif beam_temperature_goal < y3:
                        solver_time_equivalence_solved = np.inf
                    else:
                        solver_time_equivalence_solved = np.nan  # theoretically impossible, for syntax error only
                    break

        elif beam_temperature_goal - 2 <= y1 <= beam_temperature_goal + 2:
            solver_protection_thickness = x1
            solver_steel_temperature_solved = y1
            solver_convergence_status = True

            # CALCULATE BEAM FIRE RESISTANCE PERIOD IN ISO 834
            # ================================================
            # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.

            kwarg_ht_ec["time"] = iso834_time
            kwarg_ht_ec["temperature_ambient"] = iso834_temperature
            kwarg_ht_ec["thickness_protection"] = x1
            steel_temperature = _steel_temperature(**kwarg_ht_ec)

            steel_time = np.concatenate((np.array([0]), iso834_time, np.array([iso834_time[-1]])))
            steel_temperature = np.concatenate((np.array([-1]), steel_temperature, np.array([1e12])))
            func_teq = interp1d(steel_temperature, steel_time, kind="linear", bounds_error=False, fill_value=-1)
            solver_time_equivalence_solved = func_teq(beam_temperature_goal)

        elif beam_temperature_goal - 2 <= y2 <= beam_temperature_goal + 2:
            solver_protection_thickness = x2
            solver_steel_temperature_solved = y2
            solver_convergence_status = True

            # CALCULATE BEAM FIRE RESISTANCE PERIOD IN ISO 834
            # ================================================
            # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.

            kwarg_ht_ec["time"] = iso834_time
            kwarg_ht_ec["temperature_ambient"] = iso834_temperature
            kwarg_ht_ec["thickness_protection"] = x2
            steel_temperature = _steel_temperature(**kwarg_ht_ec)

            steel_time = np.concatenate((np.array([0]), iso834_time, np.array([iso834_time[-1]])))
            steel_temperature = np.concatenate((np.array([-1]), steel_temperature, np.array([1e12])))
            func_teq = interp1d(steel_temperature, steel_time, kind="linear", bounds_error=False, fill_value=-1)
            solver_time_equivalence_solved = func_teq(beam_temperature_goal)

        # No solution, thickness upper bound is not thick enough
        elif beam_temperature_goal > y1:
            solver_protection_thickness = x1
            solver_steel_temperature_solved = y1
            solver_time_equivalence_solved = -np.inf

        # No solution, thickness lower bound is not thin enough
        elif beam_temperature_goal < y2:
            solver_protection_thickness = x2
            solver_steel_temperature_solved = y2
            solver_time_equivalence_solved = np.inf

    results = dict(
        index=index,
        case_name=case_name,
        fire_type=fire_type,
        fire_load_density=fire_load_density,
        fire_combustion_efficiency=fire_combustion_efficiency,
        fire_hrr_density=fire_hrr_density,
        fire_spread_speed=fire_spread_speed,
        fire_nft_ubound=fire_nft_ubound,
        fire_time=fire_time,
        fire_temperature=fire_temperature,
        window_open_fraction=window_open_fraction,
        beam_position=beam_position,
        opening_factor=opening_factor,
        solver_convergence_status=solver_convergence_status,
        solver_time_equivalence_solved=solver_time_equivalence_solved,
        solver_steel_temperature_solved=solver_steel_temperature_solved,
        solver_protection_thickness=solver_protection_thickness,
        solver_iter_count=solver_iter_count,
        probability_weight=probability_weight,
    )

    return results


def y_results_summary(df_res: pd.DataFrame):

    df_res = copy.copy(df_res)
    df_res = df_res.replace(to_replace=[np.inf, -np.inf], value=np.nan)
    # df_res.replace(-np.inf, np.nan)
    df_res = df_res.dropna(axis=0, how="any")
    
    str_fmt = '{:<24.24}: {}\n'
    str_fmt2 = '{:<24.24}: {:<.3f}\n'

    str_out = ''

    fire_type = df_res['fire_type'].values
    fire_type_val = dict()
    for k in np.unique(fire_type):
        fire_type_val[k] = np.sum(fire_type == k)

    str_out += str_fmt.format('fire_type', fire_type_val)
    str_out += str_fmt.format('solver_convergence_status', np.sum(df_res['solver_convergence_status'].values))
    str_out += str_fmt2.format('beam_position', df_res['beam_position'].mean())
    str_out += str_fmt2.format('fire_combustion_efficiency', df_res['fire_combustion_efficiency'].mean())
    str_out += str_fmt2.format('fire_hrr_density', df_res['fire_hrr_density'].mean())
    str_out += str_fmt2.format('fire_load_density', df_res['fire_load_density'].mean())
    str_out += str_fmt2.format('fire_nft_ubound', df_res['fire_nft_ubound'].mean())
    str_out += str_fmt2.format('fire_spread_speed', df_res['fire_spread_speed'].mean())
    str_out += str_fmt2.format('window_open_fraction', df_res['window_open_fraction'].mean())

    return str_out
