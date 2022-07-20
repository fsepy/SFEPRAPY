# -*- coding: utf-8 -*-
import copy
import logging
import os
import threading
from typing import Union, Callable

import numpy as np
import pandas as pd
from fsetools.lib.fse_bs_en_1991_1_2_parametric_fire import temperature as _fire_param
from fsetools.lib.fse_bs_en_1993_1_2_heat_transfer_c import protection_thickness as _protection_thickness
from fsetools.lib.fse_bs_en_1993_1_2_heat_transfer_c import temperature as _steel_temperature
from fsetools.lib.fse_bs_en_1993_1_2_heat_transfer_c import temperature_max as _steel_temperature_max
from fsetools.lib.fse_din_en_1991_1_2_parametric_fire import temperature as _fire_param_ger
from fsetools.lib.fse_travelling_fire import temperature as fire_travelling
from pandas import DataFrame

from sfeprapy.mcs.mcs import MCS

logger = logging.getLogger('gui')


def _fire_travelling(**kwargs):
    if isinstance(kwargs["beam_location_length_m"], list) or isinstance(
            kwargs["beam_location_length_m"], np.ndarray
    ):
        kwarg_ht_ec = dict(
            fire_time=kwargs["t"],
            beam_rho=7850,
            beam_cross_section_area=0.017,
            protection_k=0.2,
            protection_rho=800,
            protection_c=1700,
            protection_thickness=0.005,
            protection_protected_perimeter=2.14,
        )

        temperature_steel_list = list()
        temperature_gas_list = fire_travelling(**kwargs)

        for temperature in temperature_gas_list:
            kwarg_ht_ec["fire_temperature"] = temperature + 273.15
            T_a_max, t = _steel_temperature_max(**kwarg_ht_ec)
            temperature_steel_list.append(T_a_max)

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
        *_,
        **__,
) -> dict:
    """Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param window_height:               [m], weighted window opening height
    :param window_width:                [m], total window opening width
    :param window_open_fraction:        [-], a factor is multiplied with the given total window opening area
    :param room_breadth:                [m], room breadth (shorter direction of the floor plan)
    :param room_depth:                  [m], room depth (longer direction of the floor plan)
    :param room_height:                 [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param fire_hrr_density:            [MW/m], fire maximum release rate per unit area
    :param fire_load_density:
    :param fire_combustion_efficiency:  [-]
    :param fire_spread_speed:           [m/s], TRAVELLING FIRE, fire spread speed
    :param fire_mode:                   0 - parametric, 1 - travelling, 2 - ger parametric, 3 - (0 & 1), 4 (1 & 2)
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
    room_total_area = (2 * room_floor_area) + ((room_breadth + room_depth) * 2 * room_height)

    # Fire load density related to the total surface area A_t
    fire_load_density_total = (
            fire_load_density_deducted * room_floor_area / room_total_area
    )

    # Opening factor
    opening_factor = window_area * np.sqrt(window_height) / room_total_area

    # Spread speed - Does the fire spread to involve the full compartment?
    fire_spread_entire_room_time = room_depth / fire_spread_speed
    burn_out_time = max([fire_load_density_deducted / fire_hrr_density, 900.0])

    if fire_mode == 0 or fire_mode == 1 or fire_mode == 2:
        # enforced to selected fire, i.e. 0 is ec parametric; 1 is travelling; and 2 is din ec parametric
        fire_type = fire_mode
    elif fire_mode == 3:
        # enforced to ec parametric + travelling
        if (
                fire_spread_entire_room_time < burn_out_time
                and 0.01 < opening_factor <= 0.2
                and 50 <= fire_load_density_total <= 1000
        ):
            fire_type = 0  # parametric fire
        else:  # Otherwise, it is a travelling fire
            fire_type = 1  # travelling fire
    elif fire_mode == 4:
        # enforced to german parametric + travelling
        # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire
        if (
                fire_spread_entire_room_time < burn_out_time
                and 0.125 <= (window_area / room_floor_area) <= 0.5
        ):
            fire_type = 2  # german parametric
        else:
            # Otherwise, it is a travelling fire
            fire_type = 1  # travelling fire
    else:
        raise ValueError("Unknown fire mode {fire_mode}.".format(fire_mode=fire_mode))

    return dict(fire_type=fire_type)


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
        fire_time: np.ndarray,
        fire_nft_limit: float,
        fire_load_density: float,
        fire_combustion_efficiency: float,
        fire_hrr_density: float,
        fire_spread_speed: float,
        fire_t_alpha: float,
        fire_gamma_fi_q: float,
        beam_position_vertical: float,
        beam_position_horizontal: Union[np.ndarray, list, float] = -1.0,
        *_,
        **__,
) -> dict:
    """Calculate temperature array of pre-defined fire type `fire_type`.

    PARAMETERS:
    :param window_height:               [m], weighted window opening height
    :param window_width:                [m], total window opening width
    :param window_open_fraction:        [-], a factor is multiplied with the given total window opening area
    :param room_breadth:                [m], room breadth (shorter direction of the floor plan)
    :param room_depth:                  [m], room depth (longer direction of the floor plan)
    :param room_height:                 [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param room_wall_thermal_inertia:   [J/m2/K/s0.5], thermal inertia of room lining material
    :param fire_tlim:                   [s], PARAMETRIC FIRE, see parametric fire function for details
    :param fire_type:                   [-],
    :param fire_time:                   [K],
    :param fire_load_density:
    :param fire_combustion_efficiency:
    :param fire_t_alpha:
    :param fire_gamma_fi_q:
    :param beam_position_vertical:
    :param fire_hrr_density:            [MW/m2], fire maximum release rate per unit area
    :param fire_spread_speed:           [m/s], TRAVELLING FIRE, fire spread speed
    :param beam_position_horizontal:    [s], beam location, will be solved for the worst case if less than 0.
    :param fire_nft_limit:              [K], TRAVELLING FIRE, maximum temperature of near field temperature
    :return:
    EXAMPLE:
    """

    fire_load_density_deducted = fire_load_density * fire_combustion_efficiency

    # Total window opening area
    window_area = window_height * window_width * window_open_fraction

    # Room floor area
    room_floor_area = room_breadth * room_depth

    # Room internal surface area, total, including window openings
    room_total_area = 2 * room_floor_area + (room_breadth + room_depth) * 2 * room_height

    if fire_type == 0:
        kwargs_fire_0_paramec = dict(
            t=fire_time,
            A_t=room_total_area,
            A_f=room_floor_area,
            A_v=window_area,
            h_eq=window_height,
            q_fd=fire_load_density_deducted * 1e6,
            lbd=room_wall_thermal_inertia ** 2,
            rho=1,
            c=1,
            t_lim=fire_tlim,
            T_0=20 + 273.15,
        )
        fire_temperature = _fire_param(**kwargs_fire_0_paramec)

    elif fire_type == 1:
        if beam_position_horizontal < 0:
            beam_position_horizontal = np.linspace(0.5 * room_depth, room_depth, 7)[1:-1]

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
        )
        fire_temperature, beam_position_horizontal = _fire_travelling(**kwargs_fire_1_travel)
        if beam_position_horizontal <= 0:
            raise ValueError("Beam position less or equal to 0.")

    elif fire_type == 2:
        kwargs_fire_2_param_din = dict(
            t=fire_time,
            A_w=window_area,
            h_w=window_height,
            A_t=room_total_area,
            A_f=room_floor_area,
            t_alpha=fire_t_alpha,
            b=room_wall_thermal_inertia,
            q_x_d=fire_load_density_deducted * 1e6,
            gamma_fi_Q=fire_gamma_fi_q,
        )
        fire_temperature = _fire_param_ger(**kwargs_fire_2_param_din)

    else:
        fire_temperature = None

    return dict(
        fire_temperature=fire_temperature,
        beam_position_horizontal=beam_position_horizontal,
    )


def solve_time_equivalence_iso834(
        fire_time: np.ndarray,
        beam_cross_section_area: float,
        beam_rho: float,
        protection_k: float,
        protection_rho: float,
        protection_c: float,
        protection_protected_perimeter: float,
        solver_temperature_goal: float,
        solver_protection_thickness: float,
        phi_teq: float,
        *_,
        **__,
) -> dict:
    """
    Calculates equivalent time exposure for a protected steel element member in more realistic fire environment (i.e. travelling fire, parameteric fires)
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param beam_cross_section_area:             [m2], the steel beam element cross section area
    :param beam_rho:                            [kg/m3], steel beam element density
    :param protection_k:                        [], steel beam element protection material thermal conductivity
    :param protection_rho:                      [kg/m3], steel beam element protection material density
    :param protection_c:                        [], steel beam element protection material specific heat
    :param protection_protected_perimeter:      [m], steel beam element protection material perimeter
    :param solver_temperature_goal:             [K], steel beam element expected failure temperature
    :param solver_max_iter:                     [-], Maximum allowable iteration counts for seeking solution for time equivalence
    :param solver_thickness_ubound:             [m], protection layer thickness upper bound initial condition for solving time equivalence
    :param solver_thickness_lbound:             [m], protection layer thickness lower bound initial condition for solving time equivalence
    :param solver_tol:                          [K], tolerance for solving time equivalence
    :param solver_protection_thickness:         [m], steel section protection layer thickness
    :param phi_teq:                             [-], model uncertainty factor
    :return results:                            A dict containing `solver_time_equivalence_solved` which is ,[s], solved equivalent time exposure
    EXAMPLE:
    """

    # ============================================
    # GOAL SEEK TO MATCH STEEL FAILURE TEMPERATURE
    # ============================================

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS

    # Solve equivalent time exposure in ISO 834
    solver_d_p = solver_protection_thickness

    if -np.inf < solver_d_p < np.inf:
        fire_temperature_iso834 = (345.0 * np.log10((fire_time / 60.0) * 8.0 + 1.0) + 20.0) + 273.15  # in [K]
        steel_temperature = _steel_temperature(
            fire_time=fire_time,
            fire_temperature=fire_temperature_iso834,
            beam_rho=beam_rho,
            beam_cross_section_area=beam_cross_section_area,
            protection_k=protection_k,
            protection_rho=protection_rho,
            protection_c=protection_c,
            protection_thickness=solver_d_p,
            protection_protected_perimeter=protection_protected_perimeter,
        )

        # Check whether steel temperature (when exposed to ISO 834 fire temperature) contains `solver_temperature_goal`
        if solver_temperature_goal < np.amin(steel_temperature):
            # critical temperature is lower than exposed steel temperature
            # this shouldn't be theoretically possible unless the given critical temperature is less than ambient temperature
            logger.error('Unexpected outputs when solving for time equivalence')
            solver_time_equivalence_solved = np.nan
        elif solver_temperature_goal > np.amax(steel_temperature):
            solver_time_equivalence_solved = np.inf
        else:
            # func_teq = interp1d(steel_temperature, fire_time, kind="linear", bounds_error=False, fill_value=-1)
            # solver_time_equivalence_solved = func_teq(solver_temperature_goal)
            solver_time_equivalence_solved = np.interp(solver_temperature_goal, steel_temperature, fire_time)
            solver_time_equivalence_solved = solver_time_equivalence_solved * phi_teq

    elif solver_d_p == np.inf:
        solver_time_equivalence_solved = np.inf
    elif solver_d_p == -np.inf:
        solver_time_equivalence_solved = -np.inf
    elif solver_d_p is np.nan:
        solver_time_equivalence_solved = np.nan
    else:
        raise ValueError(f'This error should not occur, solver_d_p = {solver_d_p}')

    return dict(solver_time_equivalence_solved=solver_time_equivalence_solved)


def solve_protection_thickness(
        fire_time: Union[list, np.ndarray],
        fire_temperature: Union[list, np.ndarray],
        beam_cross_section_area: float,
        beam_rho: float,
        protection_k: float,
        protection_rho: float,
        protection_c: float,
        protection_protected_perimeter: float,
        solver_temperature_goal: float,
        solver_max_iter: int,
        solver_thickness_ubound: float,
        solver_thickness_lbound: float,
        solver_tol: float,
        *_,
        **__,
) -> dict:
    """
    Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param fire_time:                       [s], time array
    :param fire_temperature:                [K], temperature array
    :param beam_cross_section_area:         [m2], the steel beam element cross section area
    :param beam_rho:                        [kg/m3], steel beam element density
    :param protection_k:                    [], steel beam element protection material thermal conductivity
    :param protection_rho:                  [kg/m3], steel beam element protection material density
    :param protection_c:                    [], steel beam element protection material specific heat
    :param protection_protected_perimeter:  [m], steel beam element protection material perimeter
    :param solver_temperature_goal:         [K], steel beam element expected failure temperature
    :param solver_max_iter:                 Maximum allowable iteration counts for seeking solution for time equivalence
    :param solver_thickness_ubound:         [m], protection layer thickness upper bound initial condition for solving time equivalence
    :param solver_thickness_lbound:         [m], protection layer thickness lower bound initial condition for solving time equivalence
    :param solver_tol:                      [K], tolerance for solving time equivalence
    :param phi_teq:                         [-], model uncertainty factor
    :return results:
        A dict containing the following items.
        solver_convergence_status:          [-], True if time equivalence has been successfully solved.
        solver_steel_temperature_solved
        solver_time_critical_temp_solved
        solver_protection_thickness
        solver_iter_count
    EXAMPLE:
    """

    # ============================================
    # GOAL SEEK TO MATCH STEEL FAILURE TEMPERATURE
    # ============================================

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS

    # Solve protection properties for `solver_temperature_goal`
    solver_d_p, solver_T_max_a, solver_t, solver_iter_count = _protection_thickness(
        fire_time=fire_time,
        fire_temperature=fire_temperature,
        beam_rho=beam_rho,
        beam_cross_section_area=beam_cross_section_area,
        protection_k=protection_k,
        protection_rho=protection_rho,
        protection_c=protection_c,
        protection_protected_perimeter=protection_protected_perimeter,
        solver_temperature_goal=solver_temperature_goal,
        solver_temperature_goal_tol=solver_tol,
        solver_max_iter=solver_max_iter,
        d_p_1=solver_thickness_lbound,
        d_p_2=solver_thickness_ubound,
    )

    return dict(
        solver_convergence_status=-np.inf < solver_d_p < np.inf,
        solver_steel_temperature_solved=solver_T_max_a,
        solver_time_critical_temp_solved=solver_t,
        solver_protection_thickness=solver_d_p,
        solver_iter_count=solver_iter_count,
    )


def teq_main_mp_worker(args):
    try:
        kwargs, q = args
        q.put("index: {}".format(kwargs["index"]))
        return teq_main(**kwargs)
    except (ValueError, AttributeError):
        return teq_main(**args)


def teq_main(
        case_name: str,
        n_simulations: int,
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
        timber_charred_depth=None,
        timber_hc: float = None,
        timber_density: float = None,
        timber_exposed_area: float = None,
        timber_depth: float = None,
        timber_solver_tol: float = None,
        timber_solver_ilim: float = None,
        car_cluster_size: int = None,
        *_,
        **__,
) -> dict:
    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    if room_depth < room_breadth:
        room_depth += room_breadth
        room_breadth = room_depth - room_breadth
        room_depth -= room_breadth

    # todo: wip for car park!!!
    if 'occupancy_type' in __ and __['occupancy_type'] == '__CAR_PARK__':
        fire_mode = 1  # force to travelling fire only
        # work out new room_depth_car based on how many cars are involved in fire
        if car_cluster_size is not None and car_cluster_size >= 0:
            car_cluster_size = int(car_cluster_size) + 1
            room_depth_original = float(room_depth)
            parking_bay_width = 2.3
            n_parking_bay_row = 2
            average_area_per_parking_bay = 4283 / 202

            room_depth = car_cluster_size * parking_bay_width / n_parking_bay_row
            room_floor_area = car_cluster_size * average_area_per_parking_bay
            room_breadth = room_floor_area / room_depth

            beam_position_horizontal = (beam_position_horizontal / room_depth_original) * room_depth

    window_open_fraction = (
            window_open_fraction * (1 - window_open_fraction_permanent) + window_open_fraction_permanent)

    # Fix ventilation opening size so it doesn't exceed wall area
    if window_height > room_height:
        window_height = room_height

    # Calculate fire time, this is used for all fire curves in the calculation
    fire_time = np.arange(0, fire_time_duration + fire_time_step, fire_time_step)

    # Calculate ISO 834 fire temperature
    # fire_time_iso834 = fire_time
    # fire_temperature_iso834 = (345.0 * np.log10((fire_time / 60.0) * 8.0 + 1.0) + 20.0) + 273.15  # in [K]

    inputs = copy.deepcopy(locals())
    inputs.pop('_'), inputs.pop('__')

    # initialise solver iteration count for timber fuel contribution
    timber_solver_iter_count = -1
    timber_exposed_duration = 0  # initial condition, timber exposed duration
    _fire_load_density_ = inputs.pop('fire_load_density')  # preserve original fire load density

    while True:
        timber_solver_iter_count += 1
        # the following `if` decide whether to calculate `timber_charred_depth_i` from `timber_charring_rate` or
        # `timber_charred_depth`
        if timber_charred_depth is None:
            # calculate from timber charring rate
            if isinstance(timber_charring_rate, (float, int)):
                timber_charring_rate_i = timber_charring_rate
            elif isinstance(timber_charring_rate, Callable):
                timber_charring_rate_i = timber_charring_rate(timber_exposed_duration)
            else:
                raise TypeError('`timber_charring_rate_i` is not numerical nor Callable type')
            timber_charring_rate_i *= 1. / 1000.  # [mm/min] -> [m/min]
            timber_charring_rate_i *= 1. / 60.  # [m/min] -> [m/s]
            timber_charred_depth_i = timber_charring_rate_i * timber_exposed_duration
        else:
            # calculate from timber charred depth
            if isinstance(timber_charred_depth, (float, int)):
                timber_charred_depth_i = timber_charred_depth
            elif isinstance(timber_charred_depth, Callable):
                timber_charred_depth_i = timber_charred_depth(timber_exposed_duration)
            else:
                raise TypeError('`timber_charring_rate_i` is not numerical nor Callable type')
            timber_charred_depth_i /= 1000.

        # make sure the calculated charred depth does not exceed the the available timber depth
        if timber_depth is not None:
            timber_charred_depth_i = min(timber_charred_depth_i, timber_depth)

        timber_charred_volume = timber_charred_depth_i * timber_exposed_area
        timber_charred_mass = timber_density * timber_charred_volume
        timber_fire_load = timber_charred_mass * timber_hc
        timber_fire_load_density = timber_fire_load / (room_breadth * room_depth)

        inputs['fire_load_density'] = _fire_load_density_ + timber_fire_load_density

        # To check what design fire to use
        inputs.update(decide_fire(**inputs))

        # To calculate design fire temperature
        inputs.update(evaluate_fire_temperature(**inputs))

        # To solve protection thickness at critical temperature
        inputs.update(solve_protection_thickness(**inputs))

        # To solve time equivalence in ISO 834
        inputs.update(solve_time_equivalence_iso834(**inputs))

        # additional fuel contribution from timber
        if timber_exposed_area <= 0 or timber_exposed_area is None:  # no timber exposed
            # Exit timber fuel contribution solver if:
            #     1. no timber exposed
            #     2. timber exposed area undefined
            break
        elif timber_solver_iter_count >= timber_solver_ilim:
            inputs['solver_convergence_status'] = np.nan
            inputs['solver_time_critical_temp_solved'] = np.nan
            inputs['solver_time_equivalence_solved'] = np.nan
            inputs['solver_steel_temperature_solved'] = np.nan
            inputs['solver_protection_thickness'] = np.nan
            inputs['solver_iter_count'] = np.nan
            timber_exposed_duration = np.nan
            break
        elif not -np.inf < inputs["solver_protection_thickness"] < np.inf:
            # no protection thickness solution
            timber_exposed_duration = inputs['solver_protection_thickness']
            break
        elif abs(timber_exposed_duration - inputs["solver_time_equivalence_solved"]) <= timber_solver_tol:
            # convergence sought successfully
            break
        else:
            timber_exposed_duration = inputs["solver_time_equivalence_solved"]

    inputs.update(
        dict(
            timber_charring_rate=timber_charred_depth_i / timber_exposed_duration if timber_exposed_duration else 0,
            timber_exposed_duration=timber_exposed_duration,
            timber_solver_iter_count=timber_solver_iter_count,
            timber_fire_load=timber_fire_load,
            timber_charred_depth=timber_charred_depth_i,
            timber_charred_mass=timber_charred_mass,
            timber_charred_volume=timber_charred_volume,
        )
    )

    return inputs


def mcs_out_post_per_case(
        df: DataFrame, fp: str = None, print_stats: bool = True, print_teq_plot: bool = False
) -> DataFrame:
    df = df.copy()
    df = df.select_dtypes(include=['number'])
    # save outputs if work direction is provided per iteration
    if fp is not None:
        def _save_(fp_: str):
            try:
                if not os.path.exists(os.path.dirname(fp_)):
                    os.makedirs(os.path.dirname(fp_))
            except Exception as e:
                logger.error(e)

            # only write columns contains non-unique values
            df_ = df.copy()
            nunique = df_.apply(pd.Series.nunique)
            drop_items = list(nunique[nunique == 1].index)
            if 'fire_type' in drop_items: drop_items.remove('fire_type')
            if 'fire_type' in drop_items: drop_items.remove('fire_type')
            df_.drop(drop_items, axis=1).to_csv(os.path.join(fp_), index=False)

        threading.Thread(target=_save_, kwargs=dict(fp_=fp)).start()

    if print_stats is True or print_teq_plot is True:
        df_res = df.copy()
        df_res = df_res.replace(to_replace=[np.inf, -np.inf], value=np.nan)
        df_res = df_res.dropna(axis=0, how="any")

    if print_stats is True:
        dict_ = dict()
        dict_["fire_type"] = str({k: np.sum(df_res["fire_type"].values == k) for k in df_res["fire_type"].unique()})

        for k in [
            "beam_position_horizontal", "window_open_fraction",
            "fire_combustion_efficiency", "fire_hrr_density", "fire_load_density",
            "fire_nft_limit", "fire_spread_speed",
            "phi_teq", "timber_fire_load",
        ]:
            try:
                x = df_res[k].values
                x1, x2, x3 = np.min(x), np.mean(x), np.max(x)
                dict_[k] = f"{x1:<9.3f} {x2:<9.3f} {x3:<9.3f}"
            except Exception:
                pass

        list_ = [f"{k:<24.24}: {v}" for k, v in dict_.items()]

    return df


def mcs_out_post_all_cases(df: DataFrame, fp: str = None):
    # if dir_work is not None:
    #     df[['case_name', 'index', 'solver_time_equivalence_solved']].to_csv(dir_work, index=False)
    pass


class MCS0(MCS):
    def __init__(self, print_stats: bool = True):
        self.__print_stats = print_stats
        super().__init__()

    def mcs_deterministic_calc(self, *args, **kwargs) -> dict:
        return teq_main(*args, **kwargs)

    def mcs_deterministic_calc_mp(self, *args, **kwargs) -> dict:
        return teq_main_mp_worker(*args, **kwargs)

    def post_all(self, *args, **kwargs):
        pass

    def post_case(self, df: DataFrame, write_outputs: bool = True, *_, **__):
        case_name = df['case_name'].to_numpy()
        assert (case_name == case_name[0]).all()
        case_name = case_name[0]

        try:
            fp = os.path.join(self.cwd, self.DEFAULT_TEMP_FOLDER_NAME, f'{case_name}.csv')
        except TypeError:
            fp = None

        if write_outputs:
            return mcs_out_post_per_case(df=df, fp=fp, print_stats=self.__print_stats)
        else:
            return mcs_out_post_per_case(df=df, print_stats=self.__print_stats)


def cli_main(fp_mcs_in: str, n_threads: int = 1):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS0()
    mcs.inputs = fp_mcs_in
    mcs.n_threads = n_threads
    mcs.run()
