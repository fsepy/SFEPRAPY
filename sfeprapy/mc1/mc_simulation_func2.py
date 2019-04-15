# -*- coding: utf-8 -*-
from scipy.interpolate import interp1d
import numpy as np

from sfeprapy.func.fire_travelling import fire as _fire_travelling
from sfeprapy.func.temperature_steel_section import protected_steel_eurocode_max_temperature as _steel_temperature_max
from sfeprapy.func.fire_parametric_ec import fire as _fire_param
from sfeprapy.func.fire_parametric_ec_din import fire as _fire_param_ger
from sfeprapy.dat.ec_3_1_2_kyT import func as ec3_ky


def calc_time_equivalence_worker(arg):
    kwargs, q = arg
    result = calc_time_equivalence(**kwargs)
    q.put("index: {}".format(kwargs["index"]))
    return result


def solve_thickness(
        fire_iso834_time,
        fire_iso834_temperature,

        beam_rho,
        beam_cross_section_area,

        protection_k,
        protection_rho,
        protection_c,
        protection_protected_perimeter,

        solver_temperature_target,
        solver_fire_duration,
        solver_thickness_lbound,
        solver_thickness_ubound,
        solver_tol,
        solver_iteration_limit,
):
    # Solve heat transfer using EC3 correlations
    # SI UNITS FOR INPUTS!

    kwarg_ht_ec = dict(
        time=fire_iso834_time[fire_iso834_time <= solver_fire_duration],
        temperature_ambient=fire_iso834_temperature[fire_iso834_time <= solver_fire_duration],
        rho_steel=beam_rho,
        area_steel_section=beam_cross_section_area,
        k_protection=protection_k,
        rho_protection=protection_rho,
        c_protection=protection_c,
        perimeter_protected=protection_protected_perimeter,
        terminate_max_temperature=solver_temperature_target + 5 * solver_tol,
    )

    solver_iteration_count = 0  # count how many iterations for  the seeking process
    flag_solver_status = False  # flag used to indicate when the seeking is successful

    # Default values
    solver_thickness_solved = -1

    if solver_temperature_target > 0:  # check seeking temperature, opt out if less than 0

        # def f_(x):
        #     kwarg_ht_ec["thickness_protection"] = x  # NOTE! variable out function scope
        #     T_ = _steel_temperature_max(**kwarg_ht_ec, terminate_check_wait_time=iso834_time[np.argmax(iso834_temperature)])
        #     return T_

        # Check whether there is a solution within predefined protection thickness boundaries
        x1, x2 = solver_thickness_lbound, solver_thickness_ubound

        # y1, y2 = f_(x1), f_(x2)
        y1 = _steel_temperature_max(
            **kwarg_ht_ec,
            thickness_protection=x1,
            terminate_check_wait_time=solver_fire_duration - 30
        )
        y2 = _steel_temperature_max(
            **kwarg_ht_ec,
            thickness_protection=x2,
            terminate_check_wait_time=solver_fire_duration - 30
        )

        t1, t2 = solver_temperature_target - solver_tol, solver_temperature_target + solver_tol

        if (y2 - solver_tol) <= solver_temperature_target <= (y1 + solver_tol):

            while True:

                solver_iteration_count += 1

                # Work out linear equation: f(x) = y = a x + b
                a = (y1 - y2) / (x1 - x2)
                b = y1 - a * x1

                # work out new y based upon interpolated y
                x3 = solver_thickness_solved = (solver_temperature_target - b) / a
                # y3 = f_(x3)
                y3 = _steel_temperature_max(
                    **kwarg_ht_ec,
                    thickness_protection=x3,
                    terminate_check_wait_time=solver_fire_duration - 30
                )

                if y3 < t1:  # steel temperature is too low, decrease thickness
                    x2 = x3
                    y2 = y3
                elif y3 > t2:  # steel temperature is too high, increase thickness
                    x1 = x3
                    y1 = y3
                else:
                    flag_solver_status = True

                if flag_solver_status or (solver_iteration_count >= solver_iteration_limit):
                    break

        # No solution, thickness upper bound is not thick enough
        elif solver_temperature_target > y1:
            solver_thickness_solved = x1

        # No solution, thickness lower bound is not thin enough
        elif solver_temperature_target < y2:
            solver_thickness_solved = x2

        return solver_thickness_solved, flag_solver_status, solver_iteration_count


def calc_max_temperature(
        fire_time,
        fire_mode,

        room_depth,
        room_breadth,
        room_height,

        room_wall_thermal_inertia,

        window_height,
        window_width,
        window_open_fraction,

        beam_loc_z,
        beam_position,
        beam_rho,
        beam_cross_section_area,

        protection_k,
        protection_rho,
        protection_c,
        protection_protected_perimeter,
        protection_thickness,

        fire_load_density,
        fire_hrr_density,
        fire_limiting_time,
        fire_spread_speed,
        fire_nft_ubound,
        fire_t_alpha,
        fire_gamma_fi_q
):
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
        t_lim=fire_limiting_time,
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

    elif fire_mode == 2:  # enforced to german parametric fire

        fire_temp = _fire_param_ger(**kwargs_fire_2_paramdin)
        fire_type = 2  # german parametric

    elif fire_mode == 3:  # enforced to ec parametric + travelling
        if fire_spread_entire_room_time < burn_out_time and 0.02 < opening_factor <= 0.2 and 50 <= fire_load_density_total <= 1000:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

            fire_temp = _fire_param(**kwargs_fire_0_paramec)
            fire_type = 0  # parametric fire

        else:  # Otherwise, it is a travelling fire

            fire_temp = _fire_travelling(**kwargs_fire_1_travel) + 273.15
            fire_type = 1  # travelling fire

    elif fire_mode == 4:  # enforced to german parametric + travelling

        if fire_spread_entire_room_time < burn_out_time and 0.125 <= (
                window_area / room_floor_area) <= 0.5:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

            fire_temp = _fire_param_ger(**kwargs_fire_2_paramdin)
            fire_type = 2  # german parametric

        else:  # Otherwise, it is a travelling fire

            fire_temp = _fire_travelling(**kwargs_fire_1_travel) + 273.15
            fire_type = 1  # travelling fire

    # Calculate maximum steel temperature
    # -----------------------------------

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
    )

    solver_steel_temperature_solved = _steel_temperature_max(
        **kwarg_ht_ec,
        terminate_check_wait_time=fire_time[np.argmax(fire_temp)]
    )

    return solver_steel_temperature_solved, fire_type


def calc_time_equivalence2(
        fire_iso834_time,
        fire_iso834_temperature,

        beam_rho,
        beam_cross_section_area,

        protection_k,
        protection_rho,
        protection_c,
        protection_protected_perimeter,

        solver_temperature_target,
        solver_time_duration,
        solver_thickness_lbound,
        solver_thickness_ubound,
        solver_tol,
        solver_iteration_limit,

):
    """
    NAME: calc_time_equivalence
    Ian Fu, 12 April 2019

    Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param fire_time_step: [s], time step used for numerical calculation
    :param fire_tlim: [-], PARAMETRIC FIRE, see parametric fire function for details
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
    :param protection_k: steel beam element protection material thermal conductivity
    :param protection_rho: steel beam element protection material density
    :param protection_c: steel beam element protection material specific heat
    :param protection_thickness: steel beam element protection material thickness
    :param protection_protected_perimeter: [m], steel beam element protection material perimeter
    :param index: will be returned for indicating the index of the current iteration (was used for multiprocessing)
    :param fire_mode: 0 - parametric, 1 - travelling, 2 - ger parametric, 3 - (0 & 1), 4 (1 & 2)
    :return:

    EXAMPLE:

    """

    # DO NOT CHANGE, LEGACY PARAMETERS
    # Used to define fire curve start time, depreciated on 11/03/2019 after introducing the DIN annex ec parametric
    # fire.

    # PERMEABLE AND INPUT CHECKS

    # ==================================================================================================================
    # STEP 1: SOLVE THICKNESS FOR GIVEN
    # (A) SECTION FACTOR;
    # (B) CRITICAL TEMPERATURE; AND
    # (C) ISO 834 FIRE CURVE
    # ==================================================================================================================

    solved_protection_thickness = solve_thickness(
        fire_iso834_time=fire_iso834_time,
        fire_iso834_temperature=fire_iso834_temperature,
        beam_rho=beam_rho,
        beam_cross_section_area=beam_cross_section_area,
        protection_k=protection_k,
        protection_rho=protection_rho,
        protection_c=protection_c,
        protection_protected_perimeter=protection_protected_perimeter,
        solver_temperature_target=solver_temperature_target,
        solver_fire_duration=solver_time_duration,
        solver_thickness_lbound=solver_thickness_lbound,
        solver_thickness_ubound=solver_thickness_ubound,
        solver_tol=solver_tol,
        solver_iteration_limit=solver_iteration_limit,
    )

    return solved_protection_thickness


def calc_time_equivalence(
        window_height,
        window_width,
        window_open_fraction,

        room_breadth,
        room_depth,
        room_height,
        room_wall_thermal_inertia,

        fire_iso834_time,
        fire_iso834_temperature,
        fire_mode,
        fire_time_step,
        fire_tlim,
        fire_load_density,
        fire_hrr_density,
        fire_spread_speed,
        fire_nft_ubound,
        fire_duration,
        fire_t_alpha,
        fire_gamma_fi_q,

        beam_position,
        beam_rho,
        beam_cross_section_area,
        beam_loc_z,

        protection_k,
        protection_rho,
        protection_c,
        protection_protected_perimeter,

        solver_temperature_target,
        solver_fire_duration,
        solver_thickness_lbound,
        solver_thickness_ubound,
        solver_tol,
        solver_iteration_limit,

        index,

        **kwargs
):
    """
    NAME: calc_time_equivalence
    AUTHOR: Ian Fu
    DATE: 12 April 2019
    DESCRIPTION:
    Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param fire_iso834_time:
    :param fire_iso834_temperature:
    :param fire_nft_ubound:
    :param fire_t_alpha:
    :param fire_gamma_fi_q:
    :param beam_loc_z:
    :param solver_temperature_target:
    :param solver_fire_duration:
    :param solver_thickness_lbound:
    :param solver_thickness_ubound:
    :param solver_tol:
    :param solver_iteration_limit:
    :param kwargs:
    :return:
    :param fire_time_step: [s], time step used for numerical calculation
    :param fire_tlim: [-], PARAMETRIC FIRE, see parametric fire function for details
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
    :param protection_k: steel beam element protection material thermal conductivity
    :param protection_rho: steel beam element protection material density
    :param protection_c: steel beam element protection material specific heat
    :param protection_thickness: steel beam element protection material thickness
    :param protection_protected_perimeter: [m], steel beam element protection material perimeter
    :param index: will be returned for indicating the index of the current iteration (was used for multiprocessing)
    :param fire_mode: 0 - parametric, 1 - travelling, 2 - ger parametric, 3 - (0 & 1), 4 (1 & 2)
    :return:

    EXAMPLE:

    """

    # DO NOT CHANGE, LEGACY PARAMETERS
    # Used to define fire curve start time, depreciated on 11/03/2019 after introducing the DIN annex ec parametric
    # fire.
    time_start = 0

    # PERMEABLE AND INPUT CHECKS

    # ==================================================================================================================
    # STEP 1: SOLVE THICKNESS FOR GIVEN
    # (A) SECTION FACTOR;
    # (B) CRITICAL TEMPERATURE; AND
    # (C) ISO 834 FIRE CURVE
    # ==================================================================================================================

    solved_protection_thickness, flag_solver_status, solver_iteration_count = solve_thickness(
        fire_iso834_time=fire_iso834_time,
        fire_iso834_temperature=fire_iso834_temperature,
        beam_rho=beam_rho,
        beam_cross_section_area=beam_cross_section_area,
        protection_k=protection_k,
        protection_rho=protection_rho,
        protection_c=protection_c,
        protection_protected_perimeter=protection_protected_perimeter,
        solver_temperature_target=solver_temperature_target,
        solver_fire_duration=solver_fire_duration,
        solver_thickness_lbound=solver_thickness_lbound,
        solver_thickness_ubound=solver_thickness_ubound,
        solver_tol=solver_tol,
        solver_iteration_limit=solver_iteration_limit,
    )

    # ==================================================================================================================
    # STEP 2: CALCULATE MAXIMUM STEEL TEMPERATURE FOR GIVEN
    # (A) THICKNESS CALCULATED FROM STEP 1; AND
    # (B) DESIGN FIRE TEMPERATURE CURVE.
    # ==================================================================================================================

    protection_thickness = solved_protection_thickness

    solver_steel_temperature_solved, fire_type = calc_max_temperature(
        fire_time=np.arange(time_start, fire_duration + fire_time_step, fire_time_step, dtype=float),
        fire_mode=fire_mode,

        room_depth=room_depth,
        room_breadth=room_breadth,
        room_height=room_height,
        room_wall_thermal_inertia=room_wall_thermal_inertia,

        window_height=window_height,
        window_width=window_width,
        window_open_fraction=window_open_fraction,

        beam_loc_z=beam_loc_z,
        beam_position=beam_position,
        beam_rho=beam_rho,
        beam_cross_section_area=beam_cross_section_area,

        protection_k=protection_k,
        protection_rho=protection_rho,
        protection_c=protection_c,
        protection_protected_perimeter=protection_protected_perimeter,
        protection_thickness=protection_thickness,

        fire_load_density=fire_load_density,
        fire_hrr_density=fire_hrr_density,
        fire_limiting_time=fire_tlim,
        fire_spread_speed=fire_spread_speed,
        fire_nft_ubound=fire_nft_ubound,
        fire_t_alpha=fire_t_alpha,
        fire_gamma_fi_q=fire_gamma_fi_q
    )

    # ==================================================================================================================
    # STEP 3: APPLY STEEL RETENTION FACTOR
    # ==================================================================================================================

    steel_strength_reduction_factor = ec3_ky(solver_steel_temperature_solved)

    # ==================================================================================================================
    # FINAL: PACK UP RESULTS
    # ==================================================================================================================

    # window_height,
    # window_width,
    # window_open_fraction,
    #
    # room_breadth,
    # room_depth,
    # room_height,
    # room_wall_thermal_inertia,
    #
    # fire_iso834_time,
    # fire_iso834_temperature,
    # fire_mode,
    # fire_time_step,
    # fire_tlim,
    # fire_load_density,
    # fire_hrr_density,
    # fire_spread_speed,
    # fire_nft_ubound,
    # fire_duration,
    # fire_t_alpha,
    # fire_gamma_fi_q,
    #
    # beam_position,
    # beam_rho,
    # beam_cross_section_area,
    # beam_loc_z,
    #
    # protection_k,
    # protection_rho,
    # protection_c,
    # protection_protected_perimeter,
    #
    # solver_temperature_target,
    # solver_fire_duration,
    # solver_thickness_lbound,
    # solver_thickness_ubound,
    # solver_tol,
    # solver_iteration_limit,
    #
    # index,


    results = dict(
        index=index,

        window_open_fraction=window_open_fraction,
        fire_load_density=fire_load_density,
        fire_hrr_density=fire_hrr_density,
        fire_spread_speed=fire_spread_speed,
        fire_nft_ubound=fire_nft_ubound,

        solver_steel_temperature_solved=solver_steel_temperature_solved,
        solved_protection_thickness=solved_protection_thickness,
        fire_type=fire_type,
        flag_solver_status=flag_solver_status,
        solver_iteration_count=solver_iteration_count,
        steel_strength_reduction_factor=steel_strength_reduction_factor
    )

    return results
