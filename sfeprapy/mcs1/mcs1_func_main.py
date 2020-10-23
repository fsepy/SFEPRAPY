# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from fsetools.lib.fse_bs_en_1991_1_2_parametric_fire import temperature as _fire_param
from fsetools.lib.fse_bs_en_1993_1_2_heat_transfer import temperature_max as _steel_temperature_max
from fsetools.lib.fse_bs_en_1993_1_2_strength_reduction_factor import k_y_theta_prob
from fsetools.lib.fse_din_en_1991_1_2_parametric_fire import temperature as _fire_param_ger
from fsetools.lib.fse_travelling_fire import temperature as _fire_travelling


def a_solve_steel_protection_thickness_in_iso_fire(
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
        solver_tolerance,
        solver_iteration_limit,
        **_
):
    # Solve heat transfer using EC3 correlations
    # SI UNITS FOR INPUTS!

    kwarg_ht_ec = dict(
        fire_time=fire_iso834_time[fire_iso834_time <= solver_fire_duration],
        fire_temperature=fire_iso834_temperature[fire_iso834_time <= solver_fire_duration],
        beam_rho=beam_rho,
        beam_cross_section_area=beam_cross_section_area,
        protection_k=protection_k,
        protection_rho=protection_rho,
        protection_c=protection_c,
        protection_protected_perimeter=protection_protected_perimeter,
    )

    solver_iteration_count = 0  # count how many iterations for  the seeking process
    flag_solver_status = False  # flag used to indicate when the seeking is successful

    # Default values
    solver_thickness_solved = -1

    if (
            solver_temperature_target > 0
    ):  # check seeking temperature, opt out if less than 0

        # def f_(x):
        #     kwarg_ht_ec["thickness_protection"] = x  # NOTE! variable out function scope
        #     T_ = _steel_temperature_max(**kwarg_ht_ec, terminate_check_wait_time=iso834_time[np.argmax(iso834_temperature)])
        #     return T_

        # Check whether there is a solution within predefined protection thickness boundaries
        x1, x2 = solver_thickness_lbound, solver_thickness_ubound

        # y1, y2 = f_(x1), f_(x2)
        y1 = _steel_temperature_max(
            **kwarg_ht_ec,
            protection_thickness=x1,
        )
        y2 = _steel_temperature_max(
            **kwarg_ht_ec,
            protection_thickness=x2,
        )

        t1, t2 = (
            solver_temperature_target - solver_tolerance,
            solver_temperature_target + solver_tolerance,
        )

        if (
                (y2 - solver_tolerance)
                <= solver_temperature_target
                <= (y1 + solver_tolerance)
        ):

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
                    protection_thickness=x3,
                )

                if y3 < t1:  # steel temperature is too low, decrease thickness
                    x2 = x3
                    y2 = y3
                elif y3 > t2:  # steel temperature is too high, increase thickness
                    x1 = x3
                    y1 = y3
                else:
                    flag_solver_status = True

                if flag_solver_status or (
                        solver_iteration_count >= solver_iteration_limit
                ):
                    break

        # No solution, thickness upper bound is not thick enough
        elif solver_temperature_target > y1:
            solver_thickness_solved = x1

        # No solution, thickness lower bound is not thin enough
        elif solver_temperature_target < y2:
            solver_thickness_solved = x2

    return solver_thickness_solved, flag_solver_status, solver_iteration_count


def b_calc_steel_temperature_in_design_fire(
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
        fire_tlim,
        fire_spread_speed,
        fire_nft_ubound,
        fire_t_alpha,
        fire_gamma_fi_q,
        **_
):
    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    room_depth, room_breadth = (
        max(room_depth, room_breadth),
        min(room_depth, room_breadth),
    )

    # Total window opening area
    window_area = window_height * window_width * window_open_fraction

    # Room floor area
    room_floor_area = room_breadth * room_depth

    # Room internal surface area, total, including window openings
    room_total_area = (2 * room_floor_area) + (
            (room_breadth + room_depth) * 2 * room_height
    )

    # Fire load density related to the total surface area A_t
    fire_load_density_total = fire_load_density * room_floor_area / room_total_area

    # Opening factor
    opening_factor = window_area * np.sqrt(window_height) / room_total_area

    # Spread speed - Does the fire spread to involve the full compartment?
    fire_spread_entire_room_time = room_depth / fire_spread_speed
    burn_out_time = max([fire_load_density / fire_hrr_density, 900.0])

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
        t_lim=fire_tlim,
        temperature_initial=20 + 273.15,
    )

    kwargs_fire_1_travel = dict(
        t=fire_time,
        fire_load_density_MJm2=fire_load_density,
        fire_hrr_density_MWm2=fire_hrr_density,
        room_length_m=room_depth,
        room_width_m=room_breadth,
        fire_spread_rate_ms=fire_spread_speed,
        beam_location_height_m=beam_loc_z,
        beam_location_length_m=beam_position,
        fire_nft_c=fire_nft_ubound,
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
        q_x_d_MJm2=fire_load_density,
        gamma_fi_Q=fire_gamma_fi_q,
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
        if (
                fire_spread_entire_room_time < burn_out_time
                and 0.02 < opening_factor <= 0.2
                and 50 <= fire_load_density_total <= 1000
        ):  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

            fire_temp = _fire_param(**kwargs_fire_0_paramec)
            fire_type = 0  # parametric fire

        else:  # Otherwise, it is a travelling fire

            fire_temp = _fire_travelling(**kwargs_fire_1_travel) + 273.15
            fire_type = 1  # travelling fire

    elif fire_mode == 4:  # enforced to german parametric + travelling

        if (
                fire_spread_entire_room_time < burn_out_time
                and 0.125 <= (window_area / room_floor_area) <= 0.5
        ):  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

            fire_temp = _fire_param_ger(**kwargs_fire_2_paramdin)
            fire_type = 2  # german parametric

        else:  # Otherwise, it is a travelling fire

            fire_temp = _fire_travelling(**kwargs_fire_1_travel) + 273.15
            fire_type = 1  # travelling fire

    else:
        raise ValueError("unknown fire_mode.")

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
        **kwarg_ht_ec, terminate_check_wait_time=fire_time[np.argmax(fire_temp)]
    )

    return solver_steel_temperature_solved, fire_type


def c_strength_reduction_factor(
        solver_steel_temperature_solved: np.ndarray, is_random_q: bool = True
):
    if is_random_q:
        epsilon_q = np.random.random_sample(len(solver_steel_temperature_solved))
    else:
        epsilon_q = np.full(len(solver_steel_temperature_solved), 0.5)

    steel_strength_reduction_factor = k_y_theta_prob(theta_a=solver_steel_temperature_solved, epsilon_q=epsilon_q)

    return steel_strength_reduction_factor


def main(df_mc_params: pd.DataFrame):
    dict_mc_params = df_mc_params.to_dict(orient="index")

    # ==================================================================================================================
    # solve for protection thickness
    # ==================================================================================================================

    solver_thickness_solved, flag_solver_status, solver_iteration_count = a_solve_steel_protection_thickness_in_iso_fire(
        **df_mc_params.loc[0].to_dict()
    )

    _ = [
        df_mc_params.loc[0].to_dict()["solver_fire_duration"],
        solver_thickness_solved,
        flag_solver_status,
        solver_iteration_count,
    ]
    print(
        "fire duration {:.0f}\nsolved thickness {:.10f}\nsolver status: {}\nsolver iter: {}\n".format(
            *_
        )
    )

    for uid, kwargs in dict_mc_params.items():
        solver_steel_temperature_solved, fire_type = b_calc_steel_temperature_in_design_fire(protection_thickness=solver_thickness_solved, **kwargs)

        dict_mc_params[uid]["solver_thickness_solved"] = solver_thickness_solved
        dict_mc_params[uid]["flag_solver_status"] = flag_solver_status
        dict_mc_params[uid]["solver_iteration_count"] = solver_iteration_count
        dict_mc_params[uid]["fire_type"] = fire_type
        dict_mc_params[uid]["solver_steel_temperature_solved"] = solver_steel_temperature_solved

    df_out = pd.DataFrame.from_dict(dict_mc_params, orient="index")
    df_out.set_index("index", inplace=True)  # assign 'index' column as DataFrame index

    list_c_strength_reduction_factor = c_strength_reduction_factor(df_out["solver_steel_temperature_solved"].values, True)
    df_out["strength_reduction_factor"] = list_c_strength_reduction_factor

    # TODO: HASH ITEMS BELOW TO REMOVE FROM OUTPUT CSV,
    for k in ("fire_time", "fire_iso834_time", "fire_iso834_temperature", "window_height", "window_width", "room_wall_thermal_inertia", "fire_mode", "fire_time_step", "fire_tlim",
              "fire_hrr_density", "fire_duration", "fire_t_alpha", "fire_gamma_fi_q", "beam_rho", "beam_cross_section_area", "beam_loc_z", "protection_k", "protection_rho",
              "protection_c", "protection_protected_perimeter", "solver_temperature_target", "solver_fire_duration", "solver_thickness_lbound", "solver_thickness_ubound",
              "solver_tolerance", "solver_iteration_limit"):
        df_out.pop(k)

    return df_out
