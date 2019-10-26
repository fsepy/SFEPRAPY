# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sfeprapy.dat.ec_3_1_2_kyT import ky2T_probabilistic_vectorised
from sfeprapy.func.fire_parametric_ec import fire as _fire_param
from sfeprapy.func.fire_parametric_ec_din import fire as _fire_param_ger
from sfeprapy.func.fire_travelling import fire as _fire_travelling
from sfeprapy.func.heat_transfer_protected_steel_ec import (
    protected_steel_eurocode_max_temperature as _steel_temperature_max,
)


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
        time=fire_iso834_time[fire_iso834_time <= solver_fire_duration],
        temperature_ambient=fire_iso834_temperature[
            fire_iso834_time <= solver_fire_duration
        ],
        rho_steel=beam_rho,
        area_steel_section=beam_cross_section_area,
        k_protection=protection_k,
        rho_protection=protection_rho,
        c_protection=protection_c,
        perimeter_protected=protection_protected_perimeter,
        terminate_max_temperature=solver_temperature_target + 5 * solver_tolerance,
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
            thickness_protection=x1,
            terminate_check_wait_time=solver_fire_duration - 30
        )
        y2 = _steel_temperature_max(
            **kwarg_ht_ec,
            thickness_protection=x2,
            terminate_check_wait_time=solver_fire_duration - 30
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

    steel_strength_reduction_factor = ky2T_probabilistic_vectorised(
        T=solver_steel_temperature_solved, epsilon_q=epsilon_q
    )

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

        solver_steel_temperature_solved, fire_type = b_calc_steel_temperature_in_design_fire(
            protection_thickness=solver_thickness_solved, **kwargs
        )

        dict_mc_params[uid]["solver_thickness_solved"] = solver_thickness_solved
        dict_mc_params[uid]["flag_solver_status"] = flag_solver_status
        dict_mc_params[uid]["solver_iteration_count"] = solver_iteration_count
        dict_mc_params[uid]["fire_type"] = fire_type
        dict_mc_params[uid][
            "solver_steel_temperature_solved"
        ] = solver_steel_temperature_solved

    df_out = pd.DataFrame.from_dict(dict_mc_params, orient="index")
    df_out.set_index("index", inplace=True)  # assign 'index' column as DataFrame index

    list_c_strength_reduction_factor = c_strength_reduction_factor(
        df_out["solver_steel_temperature_solved"].values, True
    )
    df_out["strength_reduction_factor"] = list_c_strength_reduction_factor

    # TODO: HASH ITEMS BELOW TO REMOVE FROM OUTPUT CSV
    df_out.pop("fire_time")
    df_out.pop("fire_iso834_time")
    df_out.pop("fire_iso834_temperature")
    df_out.pop("window_height")
    df_out.pop("window_width")
    # df_mcs_in.pop("room_breadth")
    # df_mcs_in.pop("room_depth")
    # df_mcs_in.pop("room_height")
    df_out.pop("room_wall_thermal_inertia")
    df_out.pop("fire_mode")
    df_out.pop("fire_time_step")
    df_out.pop("fire_tlim")
    df_out.pop("fire_hrr_density")
    df_out.pop("fire_duration")
    df_out.pop("fire_t_alpha")
    df_out.pop("fire_gamma_fi_q")
    df_out.pop("beam_rho")
    df_out.pop("beam_cross_section_area")
    df_out.pop("beam_loc_z")
    df_out.pop("protection_k")
    df_out.pop("protection_rho")
    df_out.pop("protection_c")
    df_out.pop("protection_protected_perimeter")
    df_out.pop("solver_temperature_target")
    df_out.pop("solver_fire_duration")
    df_out.pop("solver_thickness_lbound")
    df_out.pop("solver_thickness_ubound")
    df_out.pop("solver_tolerance")
    df_out.pop("solver_iteration_limit")
    # df_mcs_in.pop("solver_thickness_solved")
    # df_mcs_in.pop("flag_solver_status")
    # df_mcs_in.pop("solver_iteration_count")
    # df_mcs_in.pop("fire_type")

    return df_out


# if self.n_threads <= 1:
#     results = []
#     for dict_mc_params in mc_param_list:
#         results.append(calc_main(**dict_mc_params))
#
# else:
#     time_simulation_start = time.perf_counter()
#     m = mp.Manager()
#     q = m.Queue()
#     p = mp.Pool(n_threads, maxtasksperchild=1000)
#     jobs = p.map_async(calc_time_equivalence_worker, [(dict_mc_param, q) for dict_mc_param in mc_param_list])
#     n_steps = 24  # length of the progress bar
#     while True:
#         if jobs.ready():
#             time_simulation_consumed = time.perf_counter() - time_simulation_start
#             print("{}{} {:.1f}s ".format('█' * round(n_steps), '-' * round(0), time_simulation_consumed))
#             break
#         else:
#             path_ = q.qsize() / n_simulations * n_steps
#             print("{}{} {:03.1f}%".format('█' * int(round(path_)), '-' * int(n_steps - round(path_)),
#                                           path_ / n_steps * 100),
#                   end='\r')
#             time.sleep(1)
#     p.close()
#     p.join()
#     results = jobs.get()
