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
from sfeprapy.func.heat_transfer_protected_steel_ec import (
    protected_steel_eurocode as _steel_temperature,
)
from sfeprapy.func.heat_transfer_protected_steel_ec import (
    protected_steel_eurocode_max_temperature as _steel_temperature_max,
)


def _fire_travelling(**kwargs):
    if isinstance(kwargs["beam_location_length_m"], list) or isinstance(
        kwargs["beam_location_length_m"], np.ndarray
    ):

        kwarg_ht_ec = dict(
            time=kwargs["t"],
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
            kwarg_ht_ec["temperature_ambient"] = temperature + 273.15
            temperature_steel_list.append(_steel_temperature_max(**kwarg_ht_ec))

        return (
            temperature_gas_list[np.argmax(temperature_steel_list)] + 273.15,
            kwargs["beam_location_length_m"][[np.argmax(temperature_steel_list)]][0],
        )

    elif isinstance(kwargs["beam_location_length_m"], float) or isinstance(
        kwargs["beam_location_length_m"], int
    ):

        return fire_travelling(**kwargs) + 273.15, kwargs["beam_location_length_m"]


def decide_fire(
    window_height: float,
    window_width: float,
    window_open_fraction: float,
    room_breadth: float,
    room_depth: float,
    room_height: float,
    fire_mode: int,
    fire_load_density: float,
    fire_combustion_efficiency: float,
    fire_hrr_density: float,
    fire_spread_speed: float,
    **_,
) -> dict:
    """Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param window_height: [m], weighted window opening height
    :param window_width: [m], total window opening width
    :param window_open_fraction: [-], a factor is multiplied with the given total window opening area
    :param room_breadth: [m], room breadth (shorter direction of the floor plan)
    :param room_depth: [m], room depth (longer direction of the floor plan)
    :param room_height: [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param fire_hrr_density: [MW m-2], fire maximum release rate per unit area
    :param fire_load_density:
    :param fire_combustion_efficiency:
    :param fire_spread_speed: [m s-1], TRAVELLING FIRE, fire spread speed
    :param fire_mode: 0 - parametric, 1 - travelling, 2 - ger parametric, 3 - (0 & 1), 4 (1 & 2)
    :return:
    EXAMPLE:
    """

    # PERMEABLE AND INPUT CHECKS

    fire_load_density_deducted = fire_load_density * fire_combustion_efficiency

    # Total window opening area
    window_area = window_height * window_width * window_open_fraction

    # Room floor area
    room_floor_area = room_breadth * room_depth

    # Room internal surface area, total, including window openings
    room_total_area = (2 * room_floor_area) + (
        (room_breadth + room_depth) * 2 * room_height
    )

    # Fire load density related to the total surface area A_t
    fire_load_density_total = (
        fire_load_density_deducted * room_floor_area / room_total_area
    )

    # Opening factor
    opening_factor = window_area * np.sqrt(window_height) / room_total_area

    # Spread speed - Does the fire spread to involve the full compartment?
    fire_spread_entire_room_time = room_depth / fire_spread_speed
    burn_out_time = max([fire_load_density_deducted / fire_hrr_density, 900.0])

    if (
        fire_mode == 0 or fire_mode == 1 or fire_mode == 2
    ):  # enforced to ec parametric fire

        fire_type = fire_mode
    elif fire_mode == 3:  # enforced to ec parametric + travelling
        if (
            fire_spread_entire_room_time < burn_out_time
            and 0.01 < opening_factor <= 0.2
            and 50 <= fire_load_density_total <= 1000
        ):
            fire_type = 0  # parametric fire
        else:  # Otherwise, it is a travelling fire
            fire_type = 1  # travelling fire
    elif fire_mode == 4:  # enforced to german parametric + travelling
        # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire
        if (
            fire_spread_entire_room_time < burn_out_time
            and 0.125 <= (window_area / room_floor_area) <= 0.5
        ):
            fire_type = 2  # german parametric
        else:  # Otherwise, it is a travelling fire
            fire_type = 1  # travelling fire
    else:
        raise ValueError("Unknown fire mode {fire_mode}.".format(fire_mode=fire_mode))

    results = dict(fire_type=fire_type)

    return results


def evaluate_fire_temperature(
    window_height: float,
    window_width: float,
    window_open_fraction: float,
    room_breadth: float,
    room_depth: float,
    room_height: float,
    room_wall_thermal_inertia: float,
    fire_tlim: float,
    fire_type: float,
    fire_time: Union[list, np.ndarray],
    fire_nft_limit: float,
    fire_load_density: float,
    fire_combustion_efficiency: float,
    fire_hrr_density: float,
    fire_spread_speed: float,
    fire_t_alpha: float,
    fire_gamma_fi_q: float,
    beam_position_vertical: float,
    beam_position_horizontal: Union[np.ndarray, list, float] = -1.0,
    **_,
) -> dict:
    """Calculate temperature array of pre-defined fire type `fire_type`.

    PARAMETERS:
    :param window_height: [m], weighted window opening height
    :param window_width: [m], total window opening width
    :param window_open_fraction: [-], a factor is multiplied with the given total window opening area
    :param room_breadth: [m], room breadth (shorter direction of the floor plan)
    :param room_depth: [m], room depth (longer direction of the floor plan)
    :param room_height: [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param room_wall_thermal_inertia: [J m-2 K-1 s-1/2], thermal inertia of room lining material
    :param fire_tlim: [s], PARAMETRIC FIRE, see parametric fire function for details
    :param fire_type: [-],
    :param fire_time: [K],
    :param fire_load_density:
    :param fire_combustion_efficiency:
    :param fire_t_alpha:
    :param fire_gamma_fi_q:
    :param beam_position_vertical:
    :param fire_hrr_density: [MW m-2], fire maximum release rate per unit area
    :param fire_spread_speed: [m s-1], TRAVELLING FIRE, fire spread speed
    :param beam_position_horizontal: [s], beam location, will be solved for the worst case if less than 0.
    :param fire_nft_limit: [K], TRAVELLING FIRE, maximum temperature of near field temperature
    :return:
    EXAMPLE:
    """

    fire_load_density_deducted = fire_load_density * fire_combustion_efficiency

    # Total window opening area
    window_area = window_height * window_width * window_open_fraction

    # Room floor area
    room_floor_area = room_breadth * room_depth

    # Room internal surface area, total, including window openings
    room_total_area = (2 * room_floor_area) + (
        (room_breadth + room_depth) * 2 * room_height
    )

    if fire_type == 0:
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
            t_lim=fire_tlim,
            temperature_initial=20 + 273.15,
        )
        fire_temperature = _fire_param(**kwargs_fire_0_paramec)

    elif fire_type == 1:
        if beam_position_horizontal < 0:
            beam_position_horizontal = np.linspace(0.5 * room_depth, room_depth, 7)[
                1:-1
            ]

        kwargs_fire_1_travel = dict(
            t=fire_time,
            fire_load_density_MJm2=fire_load_density_deducted,
            fire_hrr_density_MWm2=fire_hrr_density,
            room_length_m=room_depth,
            room_width_m=room_breadth,
            fire_spread_rate_ms=fire_spread_speed,
            beam_location_height_m=beam_position_vertical,
            beam_location_length_m=beam_position_horizontal,
            fire_nft_limit_c=fire_nft_limit - 273.15,
            opening_width_m=window_width,
            opening_height_m=window_height,
            opening_fraction=window_open_fraction,
        )
        fire_temperature, beam_position_horizontal = _fire_travelling(
            **kwargs_fire_1_travel
        )

        if beam_position_horizontal <= 0:
            raise ValueError("Beam position less or equal to 0.")

    elif fire_type == 2:
        kwargs_fire_2_param_din = dict(
            t_array_s=fire_time,
            A_w_m2=window_area,
            h_w_m2=window_height,
            A_t_m2=room_total_area,
            A_f_m2=room_floor_area,
            t_alpha_s=fire_t_alpha,
            b_Jm2s05K=room_wall_thermal_inertia,
            q_x_d_MJm2=fire_load_density_deducted,
            gamma_fi_Q=fire_gamma_fi_q,
        )
        fire_temperature = _fire_param_ger(**kwargs_fire_2_param_din)

    else:
        fire_temperature = None

    results = dict(
        fire_temperature=fire_temperature,
        beam_position_horizontal=beam_position_horizontal,
    )

    return results


def solve_time_equivalence(
    fire_time: Union[list, np.ndarray],
    fire_temperature: Union[list, np.ndarray],
    beam_cross_section_area: float,
    beam_rho: float,
    protection_k: float,
    protection_rho: float,
    protection_c: float,
    protection_protected_perimeter: float,
    fire_time_iso834: Union[list, np.ndarray],
    fire_temperature_iso834: Union[list, np.ndarray],
    solver_temperature_goal: float,
    solver_max_iter: int,
    solver_thickness_ubound: float,
    solver_thickness_lbound: float,
    solver_tol: float,
    phi_teq: float,
    **_,
) -> dict:
    """Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param fire_time: [s], time array
    :param fire_temperature: [K], temperature array
    :param beam_cross_section_area: [m2], the steel beam element cross section area
    :param beam_rho: [kg/m3], steel beam element density
    :param solver_temperature_goal: [K], steel beam element expected failure temperature
    :param protection_k: steel beam element protection material thermal conductivity
    :param protection_rho: steel beam element protection material density
    :param protection_c: steel beam element protection material specific heat
    :param protection_protected_perimeter: [m], steel beam element protection material perimeter
    :param fire_time_iso834: [s], the time (array) component of ISO 834 fire curve
    :param fire_temperature_iso834: [K], the temperature (array) component of ISO 834 fire curve
    :param solver_max_iter: Maximum allowable iteration counts for seeking solution for time equivalence
    :param solver_thickness_ubound: [m], protection layer thickness upper bound initial condition for solving time equivalence
    :param solver_thickness_lbound: [m], protection layer thickness lower bound initial condition for solving time equivalence
    :param solver_tol: [K], tolerance for solving time equivalence
    :param phi_teq: [-], model uncertainty factor
    :return results: dict
    EXAMPLE:
    """

    # ============================================
    # GOAL SEEK TO MATCH STEEL FAILURE TEMPERATURE
    # ============================================

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS

    solver_iter_count = 0  # count how many iterations for  the seeking process
    solver_convergence_status = (
        False
    )  # flag used to indicate when the seeking is successful

    # Default values
    solver_time_equivalence_solved = -1
    solver_steel_temperature_solved = -1
    solver_protection_thickness = -1

    if solver_temperature_goal > 0:  # check seeking temperature, opt out if less than 0

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
            terminate_max_temperature=solver_temperature_goal + 2 * solver_tol,
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
            T_ = _steel_temperature_max(
                **kwarg_ht_ec, terminate_check_wait_time=terminate_check_wait_time
            )
            return T_

        # Check whether there is a solution within predefined protection thickness boundaries
        x1, x2 = solver_thickness_lbound, solver_thickness_ubound
        y1, y2 = f_(x1, time_at_max_temperature), f_(x2, time_at_max_temperature)
        t1, t2 = (
            solver_temperature_goal - solver_tol,
            solver_temperature_goal + solver_tol,
        )

        if y2 <= solver_temperature_goal <= y1:

            while True:

                solver_iter_count += 1

                # Work out linear equation: f(x) = y = a x + b
                a = (y1 - y2) / (x1 - x2)
                b = y1 - a * x1

                # work out new y based upon interpolated y
                x3 = solver_protection_thickness = (solver_temperature_goal - b) / a
                y3 = solver_steel_temperature_solved = f_(x3, time_at_max_temperature)

                if x1 < 0 or x2 < 0 or x3 < 0:
                    print("check")

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

                    kwarg_ht_ec["time"] = fire_time_iso834
                    kwarg_ht_ec["temperature_ambient"] = fire_temperature_iso834
                    kwarg_ht_ec["thickness_protection"] = x3
                    steel_temperature = _steel_temperature(**kwarg_ht_ec)

                    steel_time = np.concatenate(
                        (
                            np.array([0]),
                            fire_time_iso834,
                            np.array([fire_time_iso834[-1]]),
                        )
                    )
                    steel_temperature = np.concatenate(
                        (np.array([-1]), steel_temperature, np.array([1e12]))
                    )
                    func_teq = interp1d(
                        steel_temperature,
                        steel_time,
                        kind="linear",
                        bounds_error=False,
                        fill_value=-1,
                    )
                    solver_time_equivalence_solved = func_teq(solver_temperature_goal)

                    break

                elif (
                    solver_iter_count >= solver_max_iter
                ):  # Terminate if maximum solving iteration is reached

                    if solver_temperature_goal > y3:
                        solver_time_equivalence_solved = -np.inf
                    elif solver_temperature_goal < y3:
                        solver_time_equivalence_solved = np.inf
                    else:
                        solver_time_equivalence_solved = (
                            np.nan
                        )  # theoretically impossible, for syntax error only
                    break

        elif solver_temperature_goal - 2 <= y1 <= solver_temperature_goal + 2:
            solver_protection_thickness = x1
            solver_steel_temperature_solved = y1
            solver_convergence_status = True

            # CALCULATE BEAM FIRE RESISTANCE PERIOD IN ISO 834
            # ================================================
            # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.

            kwarg_ht_ec["time"] = fire_time_iso834
            kwarg_ht_ec["temperature_ambient"] = fire_temperature_iso834
            kwarg_ht_ec["thickness_protection"] = x1
            steel_temperature = _steel_temperature(**kwarg_ht_ec)

            steel_time = np.concatenate(
                (np.array([0]), fire_time_iso834, np.array([fire_time_iso834[-1]]))
            )
            steel_temperature = np.concatenate(
                (np.array([-1]), steel_temperature, np.array([1e12]))
            )
            func_teq = interp1d(
                steel_temperature,
                steel_time,
                kind="linear",
                bounds_error=False,
                fill_value=-1,
            )
            solver_time_equivalence_solved = func_teq(solver_temperature_goal)

        elif solver_temperature_goal - 2 <= y2 <= solver_temperature_goal + 2:
            solver_protection_thickness = x2
            solver_steel_temperature_solved = y2
            solver_convergence_status = True

            # CALCULATE BEAM FIRE RESISTANCE PERIOD IN ISO 834
            # ================================================
            # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.

            kwarg_ht_ec["time"] = fire_time_iso834
            kwarg_ht_ec["temperature_ambient"] = fire_temperature_iso834
            kwarg_ht_ec["thickness_protection"] = x2
            steel_temperature = _steel_temperature(**kwarg_ht_ec)

            steel_time = np.concatenate(
                (np.array([0]), fire_time_iso834, np.array([fire_time_iso834[-1]]))
            )
            steel_temperature = np.concatenate(
                (np.array([-1]), steel_temperature, np.array([1e12]))
            )
            func_teq = interp1d(
                steel_temperature,
                steel_time,
                kind="linear",
                bounds_error=False,
                fill_value=-1,
            )
            solver_time_equivalence_solved = func_teq(solver_temperature_goal)

        # No solution, thickness upper bound is not thick enough
        elif solver_temperature_goal > y1:
            solver_protection_thickness = x1
            solver_steel_temperature_solved = y1
            solver_time_equivalence_solved = -np.inf

        # No solution, thickness lower bound is not thin enough
        elif solver_temperature_goal < y2:
            solver_protection_thickness = x2
            solver_steel_temperature_solved = y2
            solver_time_equivalence_solved = np.inf

    solver_time_equivalence_solved *= phi_teq

    results = dict(
        solver_convergence_status=solver_convergence_status,
        solver_time_equivalence_solved=solver_time_equivalence_solved,
        solver_steel_temperature_solved=solver_steel_temperature_solved,
        solver_protection_thickness=solver_protection_thickness,
        solver_iter_count=solver_iter_count,
    )

    return results


def mcs_out_post(df: pd.DataFrame) -> pd.DataFrame:
    df_res = copy.copy(df)
    df_res = df_res.replace(to_replace=[np.inf, -np.inf], value=np.nan)
    df_res = df_res.dropna(axis=0, how="any")

    dict_ = dict()
    dict_["fire_type"] = str(
        {
            k: np.sum(df_res["fire_type"].values == k)
            for k in np.unique(df_res["fire_type"].values)
        }
    )

    for k in [
        "beam_position_horizontal",
        "fire_combustion_efficiency",
        "fire_hrr_density",
        "fire_load_density",
        "fire_nft_limit",
        "fire_spread_speed",
        "window_open_fraction",
        "phi_teq",
        "timber_fire_load",
    ]:
        try:
            x = df_res[k].values
            x1, x2, x3 = np.min(x), np.mean(x), np.max(x)
            dict_[k] = f"{x1:<9.3f} {x2:<9.3f} {x3:<9.3f}"
        except (KeyError, ValueError):
            pass

    list_ = [f"{k:<24.24}: {v}" for k, v in dict_.items()]

    print("\n".join(list_), "\n")

    return df


def teq_main_wrapper(args):
    try:
        kwargs, q = args
        q.put("index: {}".format(kwargs["index"]))
        return teq_main(**kwargs)
    except (ValueError, AttributeError):
        return teq_main(**args)


def teq_main(
    case_name: str,
    n_simulations: int,
    probability_weight: float,
    index: int,
    beam_cross_section_area: float,
    beam_position_vertical: float,
    beam_position_horizontal: float,
    beam_rho: float,
    fire_time_duration: float,
    fire_time_step: float,
    fire_combustion_efficiency: float,
    fire_gamma_fi_q: float,
    fire_hrr_density: float,
    fire_load_density: float,
    fire_mode: int,
    fire_nft_limit: float,
    fire_spread_speed: float,
    fire_t_alpha: float,
    fire_tlim: float,
    protection_c: float,
    protection_k: float,
    protection_protected_perimeter: float,
    protection_rho: float,
    room_breadth: float,
    room_depth: float,
    room_height: float,
    room_wall_thermal_inertia: float,
    solver_temperature_goal: float,
    solver_max_iter: int,
    solver_thickness_lbound: float,
    solver_thickness_ubound: float,
    solver_tol: float,
    window_height: float,
    window_open_fraction: float,
    window_width: float,
    window_open_fraction_permanent: float,
    phi_teq: float = 1.0,
    timber_charring_rate=None,
    timber_hc: float = None,
    timber_density: float = None,
    timber_exposed_area: float = None,
    timber_solver_tol: float = None,
    timber_solver_ilim: float = None,
    **_,
) -> dict:
    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    if room_depth < room_breadth:
        room_depth += room_breadth
        room_breadth = room_depth - room_breadth
        room_depth -= room_breadth

    window_open_fraction = (
        window_open_fraction * (1 - window_open_fraction_permanent)
        + window_open_fraction_permanent
    )

    # Calculate fire time, this is used for all fire curves in the calculation

    fire_time = np.arange(0, fire_time_duration + fire_time_step, fire_time_step)

    # Calculate ISO 834 fire temperature

    fire_time_iso834 = fire_time
    fire_temperature_iso834 = (
        345.0 * np.log10((fire_time / 60.0) * 8.0 + 1.0) + 20.0
    ) + 273.15  # in [K]

    # Inject results, i.e. these values will be in the output
    res = dict(
        case_name=case_name,
        n_simulations=n_simulations,
        probability_weight=probability_weight,
        index=index,
        # beam_cross_section_area=beam_cross_section_area,
        # beam_position_vertical=beam_position_vertical,
        beam_position_horizontal=beam_position_horizontal,
        # beam_rho=beam_rho,
        # fire_time=fire_time,
        fire_combustion_efficiency=fire_combustion_efficiency,
        # fire_gamma_fi_q=fire_gamma_fi_q,
        fire_hrr_density=fire_hrr_density,
        fire_load_density=fire_load_density,
        # fire_mode=fire_mode,
        fire_nft_limit=fire_nft_limit,
        fire_spread_speed=fire_spread_speed,
        # fire_t_alpha=fire_t_alpha,
        # fire_tlim=fire_tlim,
        # fire_temperature_iso834=fire_temperature_iso834,
        # fire_time_iso834=fire_time_iso834,
        # protection_c=protection_c,
        # protection_k=protection_k,
        # protection_protected_perimeter=protection_protected_perimeter,
        # protection_rho=protection_rho,
        # room_breadth=room_breadth,
        # room_depth=room_depth,
        # room_height=room_height,
        # room_wall_thermal_inertia=room_wall_thermal_inertia,
        # solver_temperature_goal=solver_temperature_goal,
        # solver_max_iter=solver_max_iter,
        # solver_thickness_lbound=solver_thickness_lbound,
        # solver_thickness_ubound=solver_thickness_ubound,
        # solver_tol=solver_tol,
        # window_height=window_height,
        window_open_fraction=window_open_fraction,
        # window_width=window_width,
        phi_teq=phi_teq,
        # timber_charring_rate=timber_charring_rate,
        # timber_hc=timber_hc,
        # timber_density=timber_density,
        # timber_exposed_area=timber_exposed_area,
        # timber_solver_tol=timber_solver_tol,
        # timber_solver_ilim=timber_solver_ilim,
    )

    # initial timber exposure time
    if timber_exposed_area > 0:
        timber_exposed_duration = 1200
    else:
        timber_exposed_duration = 0

    timber_solver_iter = (
        -1
    )  # initialise solver iteration count for timber fuel contribution

    if isinstance(timber_charring_rate, (int, float)):
        timber_charring_rate_ = copy.copy(timber_charring_rate)
        timber_charring_rate = lambda x: timber_charring_rate_

    while True:
        timber_solver_iter += 1
        timber_charring_rate_ = timber_charring_rate(timber_exposed_duration)
        timber_charring_rate_ *= 1 / 1000  # [mm/min] -> [m/min]
        timber_charring_rate_ *= 1 / 60  # [m/min] -> [m/s]
        timber_charred_depth = timber_charring_rate_ * timber_exposed_duration
        timber_charred_volume = timber_charred_depth * timber_exposed_area
        timber_charred_mass = timber_density * timber_charred_volume
        timber_fire_load = timber_charred_mass * timber_hc
        timber_fire_load_density = timber_fire_load / (room_breadth * room_depth)

        # To check what design fire to use

        res_decide_fire = decide_fire(
            window_height,
            window_width,
            window_open_fraction,
            room_breadth,
            room_depth,
            room_height,
            fire_mode,
            fire_load_density + timber_fire_load_density,
            fire_combustion_efficiency,
            fire_hrr_density,
            fire_spread_speed,
        )

        # To calculate design fire temperature

        res_evaluate_fire_temperature = evaluate_fire_temperature(
            window_height,
            window_width,
            window_open_fraction,
            room_breadth,
            room_depth,
            room_height,
            room_wall_thermal_inertia,
            fire_tlim,
            res_decide_fire["fire_type"],
            fire_time,
            fire_nft_limit,
            fire_load_density + timber_fire_load_density,
            fire_combustion_efficiency,
            fire_hrr_density,
            fire_spread_speed,
            fire_t_alpha,
            fire_gamma_fi_q,
            beam_position_vertical,
            beam_position_horizontal,
        )

        # To calculate time equivalence

        res_solve_time_equivalence = solve_time_equivalence(
            fire_time,
            res_evaluate_fire_temperature["fire_temperature"],
            beam_cross_section_area,
            beam_rho,
            protection_k,
            protection_rho,
            protection_c,
            protection_protected_perimeter,
            fire_time_iso834,
            fire_temperature_iso834,
            solver_temperature_goal,
            solver_max_iter,
            solver_thickness_ubound,
            solver_thickness_lbound,
            solver_tol,
            phi_teq,
        )

        if timber_exposed_area <= 0 or timber_exposed_area is None:  # no timber exposed
            break
        elif not res_solve_time_equivalence[
            "solver_convergence_status"
        ]:  # no time equivalence solution
            break
        elif (
            timber_solver_iter >= timber_solver_ilim
        ):  # over the solver iteration limit
            break
        elif (
            abs(
                timber_exposed_duration
                - res_solve_time_equivalence["solver_time_equivalence_solved"]
            )
            <= timber_solver_tol
        ):  # convergence sought successfully
            break
        else:
            timber_exposed_duration = res_solve_time_equivalence[
                "solver_time_equivalence_solved"
            ]

    res_timber_solver = dict(
        timber_charring_rate=timber_charring_rate_,
        timber_exposed_duration=timber_exposed_duration,
        timber_solver_iter=timber_solver_iter,
        timber_fire_load=timber_fire_load,
        timber_charred_depth=timber_charred_depth,
        timber_charred_mass=timber_charred_mass,
        timber_charred_volume=timber_charred_volume,
    )

    res.update(res_timber_solver)
    res.update(res_decide_fire)
    res.update(res_evaluate_fire_temperature)
    res.update(res_solve_time_equivalence)

    return res


def _test_teq_phi():
    warnings.filterwarnings("ignore")

    from sfeprapy.func.fire_iso834 import fire as fire_iso834

    fire_time_ = np.arange(0, 2 * 60 * 60, 1)
    fire_temperature_iso834_ = fire_iso834(fire_time_, 293.15)

    input_param = dict(
        index=0,
        case_name="Standard 1",
        probability_weight=1,
        fire_time_step=30,
        fire_time_duration=5 * 60 * 60,
        n_simulations=1,
        beam_cross_section_area=0.017,
        beam_position_vertical=2.5,
        beam_position_horizontal=18,
        beam_rho=7850,
        fire_combustion_efficiency=0.8,
        fire_gamma_fi_q=1,
        fire_hrr_density=0.25,
        fire_load_density=420,
        fire_mode=0,
        fire_nft_limit=1050,
        fire_spread_speed=0.01,
        fire_t_alpha=300,
        fire_tlim=0.333,
        fire_temperature_iso834=fire_temperature_iso834_,
        fire_time_iso834=fire_time_,
        protection_c=1700,
        protection_k=0.2,
        protection_protected_perimeter=2.14,
        protection_rho=7850,
        room_breadth=16,
        room_depth=31.25,
        room_height=3,
        room_wall_thermal_inertia=720,
        solver_temperature_goal=620 + 273.15,
        solver_max_iter=20,
        solver_thickness_lbound=0.0001,
        solver_thickness_ubound=0.0500,
        solver_tol=1.0,
        window_height=2,
        window_open_fraction=0.8,
        window_width=72,
        window_open_fraction_permanent=0,
        phi_teq=0.1,
        timber_charring_rate=0.7,
        timber_exposed_area=0,
        timber_hc=400,
        timber_density=500,
        timber_solver_ilim=20,
        timber_solver_tol=1,
    )

    input_param["phi_teq"] = 1.0
    teq_10 = teq_main(**input_param)["solver_time_equivalence_solved"]

    input_param["phi_teq"] = 0.1
    teq_01 = teq_main(**input_param)["solver_time_equivalence_solved"]

    assert abs(teq_10 / teq_01 - 10) < 0.001


def _test_standard_case():
    import copy
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
    # from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen
    from scipy.interpolate import interp1d
    import numpy as np

    # increase the number of simulations so it gives sensible results
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    mcs_config = copy.deepcopy(EXAMPLE_CONFIG_DICT)
    for k in list(mcs_input.keys()):
        mcs_input[k]["phi_teq"] = 1
        mcs_input[k]["n_simulations"] = 333
        mcs_input[k]["probability_weight"] = 1 / 3.0
        mcs_input[k]["fire_time_duration"] = 10000
        mcs_input[k]["timber_exposed_area"] = 0
        mcs_input[k].pop("beam_position_horizontal")
        mcs_input[k]["beam_position_horizontal:dist"] = "uniform_"
        mcs_input[k]["beam_position_horizontal:ubound"] = (
            mcs_input[k]["room_depth"] * 0.9
        )
        mcs_input[k]["beam_position_horizontal:lbound"] = (
            mcs_input[k]["room_depth"] * 0.6
        )

    # increase the number of threads so it runs faster
    mcs_config["n_threads"] = 1  # coverage does not support
    mcs = MCS()
    mcs.define_problem(data=mcs_input, config=mcs_config)
    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)
    mcs.run_mcs()
    mcs_out = mcs.mcs_out
    teq = mcs_out["solver_time_equivalence_solved"] / 60.0
    hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.5))
    x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    teq_at_80_percentile = interp1d(y, x)(0.8)
    print(teq_at_80_percentile)
    target, target_tol = 60, 2
    assert target - target_tol < teq_at_80_percentile < target + target_tol
