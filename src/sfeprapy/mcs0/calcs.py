__all__ = (
    'decide_fire', 'evaluate_fire_temperature', 'solve_time_equivalence_iso834', 'solve_protection_thickness',
    'teq_main', 'TeqResult',
)

from random import random
from typing import NamedTuple, Union

import numpy as np

from ._fsetools import parametric_fire_temperature as _fire_param
from ._fsetools import travelling_fire_temperature as fire_travelling
from ._fsetools import protection_thickness_2 as _protection_thickness_2
from ._fsetools import temperature as _steel_temperature


# ---------------------------------------------------------------------
# Eurocode parametric-fire validity limits (BS EN 1991-1-2 Annex A).
# These are code-defined applicability ranges, not tunable parameters.
# ---------------------------------------------------------------------
# BS EN 1991-1-2 Annex A parametric fire (EC): opening factor O [m^0.5*s^0.5]
# is valid in [0.02, 0.20]; the UK National Annex widens the lower bound to 0.01.
OPENING_FACTOR_LBOUND_EC = 0.01   # [m^0.5*s^0.5], UK NA lower bound
OPENING_FACTOR_UBOUND_EC = 0.20   # [m^0.5*s^0.5], EC upper bound
FIRE_LOAD_DENSITY_TOTAL_LBOUND_EC = 50.0    # [MJ/m^2] related to A_t
FIRE_LOAD_DENSITY_TOTAL_UBOUND_EC = 1000.0  # [MJ/m^2] related to A_t
# Minimum burnout time, below which the fuel is deemed to burn out before the
# fire can spread across the whole compartment.
MIN_BURNOUT_TIME = 900.0  # [s]


class TeqResult(NamedTuple):
    """Result of :func:`teq_main`.

    A ``NamedTuple`` so callers read fields by name (``r.solver_time_equivalence_solved``)
    rather than position. It is still a plain ``tuple``.

    Note: fields that merely echo an input argument (``index``,
    ``fire_combustion_efficiency``, ``fire_hrr_density``, ``fire_nft_limit``,
    ``fire_spread_speed``, ``beam_position_horizontal``, ``fire_load_density``) are
    intentionally *not* returned -- the caller already has them. The total fire load
    density actually used is ``fire_load_density`` (input) plus
    ``timber_fire_load / (room_breadth * room_depth)`` (treat ``timber_fire_load`` as 0
    when it is NaN, i.e. no timber).
    """
    fire_type: int
    t1: float
    t2: float
    t3: float
    solver_steel_temperature_solved: float
    solver_time_critical_temp_solved: float
    solver_protection_thickness: float
    solver_iter_count: int
    solver_time_equivalence_solved: float
    timber_exposed_duration: float
    timber_solver_iter_count: int
    timber_fire_load: float


def decide_fire(
        window_height: float,
        window_width: float,
        room_breadth: float,
        room_depth: float,
        room_height: float,
        fire_mode: int,
        fire_load_density: float,
        fire_combustion_efficiency: float,
        fire_hrr_density: float,
        fire_spread_speed: float,
) -> int:
    """Calculates equivalent time exposure for a protected steel element member in more realistic fire environment
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param window_height:               [m], weighted window opening height
    :param window_width:                [m], total window opening width (the ventilation opening geometry directly)
    :param room_breadth:                [m], room breadth (shorter direction of the floor plan)
    :param room_depth:                  [m], room depth (longer direction of the floor plan)
    :param room_height:                 [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param fire_hrr_density:            [MW/m], fire maximum release rate per unit area
    :param fire_load_density:
    :param fire_combustion_efficiency:  [-]
    :param fire_spread_speed:           [m/s], TRAVELLING FIRE, fire spread speed
    :param fire_mode:                   0 - parametric, 1 - travelling, 3 - (0 & 1) auto-selected
    :return:
    EXAMPLE:
    """

    # PERMEABLE AND INPUT CHECKS

    fire_load_density_deducted = fire_load_density * fire_combustion_efficiency

    # Total window opening area
    window_area = window_height * window_width

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
    burn_out_time = max([fire_load_density_deducted / fire_hrr_density, MIN_BURNOUT_TIME])

    if fire_mode == 0 or fire_mode == 1:
        # enforced to selected fire, i.e. 0 is ec parametric; 1 is travelling
        fire_type = fire_mode
    elif fire_mode == 3:
        # enforced to ec parametric + travelling
        if (
                fire_spread_entire_room_time < burn_out_time
                and OPENING_FACTOR_LBOUND_EC < opening_factor <= OPENING_FACTOR_UBOUND_EC
                and FIRE_LOAD_DENSITY_TOTAL_LBOUND_EC <= fire_load_density_total <= FIRE_LOAD_DENSITY_TOTAL_UBOUND_EC
        ):
            fire_type = 0  # parametric fire
        else:  # Otherwise, it is a travelling fire
            fire_type = 1  # travelling fire
    else:
        raise ValueError("Unknown fire mode {fire_mode}.".format(fire_mode=fire_mode))

    return fire_type


def evaluate_fire_temperature(
        window_height: float,
        window_width: float,
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
        beam_position_vertical: float,
        beam_position_horizontal: float,
) -> tuple:
    """Calculate temperature array of pre-defined fire type `fire_type`.

    PARAMETERS:
    :param window_height:               [m], weighted window opening height
    :param window_width:                [m], total window opening width (the ventilation opening geometry directly)
    :param room_breadth:                [m], room breadth (shorter direction of the floor plan)
    :param room_depth:                  [m], room depth (longer direction of the floor plan)
    :param room_height:                 [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param room_wall_thermal_inertia:   [J/m2/K/s0.5], thermal inertia of room lining material
    :param fire_tlim:                   [s], PARAMETRIC FIRE, see parametric fire function for details
    :param fire_type:                   [-],
    :param fire_time:                   [K],
    :param fire_load_density:
    :param fire_combustion_efficiency:
    :param beam_position_vertical:
    :param fire_hrr_density:            [MW/m2], fire maximum release rate per unit area
    :param fire_spread_speed:           [m/s], TRAVELLING FIRE, fire spread speed
    :param beam_position_horizontal:    [m], beam lateral distance from the fire origin (travelling fire only)
    :param fire_nft_limit:              [K], TRAVELLING FIRE, maximum temperature of near field temperature
    :return:
    EXAMPLE:
    """

    fire_load_density_deducted = fire_load_density * fire_combustion_efficiency

    # Total window opening area
    window_area = window_height * window_width

    # Room floor area
    room_floor_area = room_breadth * room_depth

    # Room internal surface area, total, including window openings
    room_total_area = 2 * room_floor_area + (room_breadth + room_depth) * 2 * room_height

    if fire_type == 0:
        fire_temperature = _fire_param(
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
        t1, t2, t3 = np.nan, np.nan, np.nan

    elif fire_type == 1:
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
        fire_temperature = fire_travelling(**kwargs_fire_1_travel) + 273.15

        t1 = min(room_depth / fire_spread_speed, fire_load_density_deducted / fire_hrr_density)
        t2 = max(room_depth / fire_spread_speed, fire_load_density_deducted / fire_hrr_density)
        t3 = t1 + t2

    else:
        fire_temperature = np.nan
        t1, t2, t3 = np.nan, np.nan, np.nan

    return fire_temperature, t1, t2, t3


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
) -> float:
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
    :param solver_protection_thickness:         [m], steel section protection layer thickness
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
            # this shouldn't be theoretically possible unless the given critical temperature is less than ambient
            # temperature
            solver_time_equivalence_solved = np.nan
        elif solver_temperature_goal > np.amax(steel_temperature):
            solver_time_equivalence_solved = np.inf
        else:
            # func_teq = interp1d(steel_temperature, fire_time, kind="linear", bounds_error=False, fill_value=-1)
            # solver_time_equivalence_solved = func_teq(solver_temperature_goal)
            solver_time_equivalence_solved = np.interp(solver_temperature_goal, steel_temperature, fire_time)

    elif solver_d_p == np.inf:
        solver_time_equivalence_solved = np.inf
    elif solver_d_p == -np.inf:
        solver_time_equivalence_solved = -np.inf
    elif solver_d_p is np.nan:
        solver_time_equivalence_solved = np.nan
    else:
        raise ValueError(f'This error should not occur, solver_d_p = {solver_d_p}')

    return solver_time_equivalence_solved


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
        solver_tol: float,
        *_,
        **__,
) -> tuple:
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
    :param solver_tol:                      [K], tolerance for solving time equivalence
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
    # solver_d_p, solver_T_max_a, solver_t, solver_iter_count = _protection_thickness(
    #     fire_time=fire_time,
    #     fire_temperature=fire_temperature,
    #     beam_rho=beam_rho,
    #     beam_cross_section_area=beam_cross_section_area,
    #     protection_k=protection_k,
    #     protection_rho=protection_rho,
    #     protection_c=protection_c,
    #     protection_protected_perimeter=protection_protected_perimeter,
    #     solver_temperature_goal=solver_temperature_goal,
    #     solver_temperature_goal_tol=solver_tol,
    #     solver_max_iter=solver_max_iter,
    #     d_p_1=solver_thickness_lbound,
    #     d_p_2=solver_thickness_ubound,
    # )
    # return solver_T_max_a, solver_t, solver_d_p, solver_iter_count

    solver_d_p, solver_T_max_a, solver_t, solver_iter_count, solver_status = _protection_thickness_2(
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
        solver_max_iter=100,
        d_p_1=0.0001,
        d_p_2=0.0801,
        d_p_i=0.0025 + random() * 0.0025,
    )

    if solver_status == 0:
        return solver_T_max_a, solver_t, solver_d_p, solver_iter_count
    elif solver_status == 1:
        return -np.inf, solver_t, solver_d_p, solver_iter_count
    elif solver_status == 2:
        return np.inf, solver_t, solver_d_p, solver_iter_count
    elif solver_status == 3:
        return np.nan, np.nan, np.nan, solver_iter_count
    elif solver_status == 4:
        # Monotonicity failed: the solver returns the last valid point. Treat it as the best
        # available solution so downstream time-equivalence solving can still proceed.
        return solver_T_max_a, solver_t, solver_d_p, solver_iter_count


def _solve_teq_once(
        fire_time: np.ndarray,
        fire_load_density: float,
        # geometry / ventilation
        window_height: float,
        window_width: float,
        room_breadth: float,
        room_depth: float,
        room_height: float,
        room_wall_thermal_inertia: float,
        # fire
        fire_tlim: float,
        fire_mode: int,
        fire_nft_limit: float,
        fire_combustion_efficiency: float,
        fire_hrr_density: float,
        fire_spread_speed: float,
        beam_position_vertical: float,
        beam_position_horizontal: float,
        # steel / protection
        beam_cross_section_area: float,
        beam_rho: float,
        protection_k: float,
        protection_rho: float,
        protection_c: float,
        protection_protected_perimeter: float,
        # solver
        solver_temperature_goal: float,
        solver_tol: float,
) -> tuple:
    """Single-pass time-equivalence solve (no timber coupling).

    Runs ``decide_fire`` -> ``evaluate_fire_temperature`` -> ``solve_protection_thickness``
    -> ``solve_time_equivalence_iso834`` for a given ``fire_load_density`` and returns all
    per-iteration results. This is the core calculation that ``teq_main`` calls once for the
    plain case, or repeatedly inside the timber-convergence loop.
    """
    fire_type = decide_fire(
        window_height=window_height, window_width=window_width,
        room_breadth=room_breadth, room_depth=room_depth, room_height=room_height, fire_mode=fire_mode,
        fire_load_density=fire_load_density, fire_combustion_efficiency=fire_combustion_efficiency,
        fire_hrr_density=fire_hrr_density, fire_spread_speed=fire_spread_speed
    )

    fire_temperature, t1, t2, t3 = evaluate_fire_temperature(
        window_height=window_height, window_width=window_width,
        room_breadth=room_breadth, room_depth=room_depth, room_height=room_height,
        room_wall_thermal_inertia=room_wall_thermal_inertia, fire_tlim=fire_tlim, fire_type=fire_type,
        fire_time=fire_time, fire_nft_limit=fire_nft_limit, fire_load_density=fire_load_density,
        fire_combustion_efficiency=fire_combustion_efficiency, fire_hrr_density=fire_hrr_density,
        fire_spread_speed=fire_spread_speed,
        beam_position_vertical=beam_position_vertical, beam_position_horizontal=beam_position_horizontal
    )

    (
        solver_steel_temperature_solved, solver_time_critical_temp_solved, solver_protection_thickness,
        solver_iter_count
    ) = solve_protection_thickness(
        fire_time=fire_time, fire_temperature=fire_temperature, beam_cross_section_area=beam_cross_section_area,
        beam_rho=beam_rho, protection_k=protection_k, protection_rho=protection_rho, protection_c=protection_c,
        protection_protected_perimeter=protection_protected_perimeter,
        solver_temperature_goal=solver_temperature_goal,
        solver_tol=solver_tol
    )

    solver_time_equivalence_solved = solve_time_equivalence_iso834(
        fire_time=fire_time, beam_cross_section_area=beam_cross_section_area, beam_rho=beam_rho,
        protection_k=protection_k, protection_rho=protection_rho, protection_c=protection_c,
        protection_protected_perimeter=protection_protected_perimeter,
        solver_temperature_goal=solver_temperature_goal, solver_protection_thickness=solver_protection_thickness,
    )

    return (fire_type, t1, t2, t3,
            solver_steel_temperature_solved, solver_time_critical_temp_solved, solver_protection_thickness,
            solver_iter_count, solver_time_equivalence_solved)


def teq_main(
        beam_cross_section_area: float,
        beam_position_vertical: float,
        beam_position_horizontal: float,
        beam_rho: float,
        fire_time_duration: float,
        fire_time_step: float,
        fire_combustion_efficiency: float,
        fire_hrr_density: float,
        fire_load_density: float,
        fire_mode: int,
        fire_nft_limit: float,
        fire_spread_speed: float,
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
        solver_tol: float,
        window_height: float,
        window_width: float,
        timber_burning_rate: float = 0.,
        timber_fire_load_max: float = None,
        timber_solver_tol: float = None,
        timber_solver_ilim: float = None,
) -> tuple:
    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    if room_depth < room_breadth:
        room_depth += room_breadth
        room_breadth = room_depth - room_breadth
        room_depth -= room_breadth

    # Fix ventilation opening size, so it doesn't exceed wall area
    if window_height > room_height:
        window_height = room_height

    # Calculate fire time, this is used for all fire curves in the calculation
    fire_time = np.arange(0, fire_time_duration + fire_time_step, fire_time_step)

    _fire_load_density_ = float(fire_load_density)  # preserve original fire load density

    # Common keyword bundle for the single-pass solver (see ``_solve_teq_once``).
    once_kwargs = dict(
        fire_time=fire_time,
        window_height=window_height, window_width=window_width,
        room_breadth=room_breadth, room_depth=room_depth, room_height=room_height,
        room_wall_thermal_inertia=room_wall_thermal_inertia, fire_tlim=fire_tlim, fire_mode=fire_mode,
        fire_nft_limit=fire_nft_limit, fire_combustion_efficiency=fire_combustion_efficiency,
        fire_hrr_density=fire_hrr_density, fire_spread_speed=fire_spread_speed,
        beam_position_vertical=beam_position_vertical, beam_position_horizontal=beam_position_horizontal,
        beam_cross_section_area=beam_cross_section_area, beam_rho=beam_rho,
        protection_k=protection_k, protection_rho=protection_rho, protection_c=protection_c,
        protection_protected_perimeter=protection_protected_perimeter,
        solver_temperature_goal=solver_temperature_goal,
        solver_tol=solver_tol,
    )

    has_timber = timber_burning_rate is not None and timber_burning_rate > 0

    if not has_timber:
        # ----- Plain path: a single time-equivalence solve, no timber coupling. -----
        (
            fire_type, t1, t2, t3,
            solver_steel_temperature_solved, solver_time_critical_temp_solved, solver_protection_thickness,
            solver_iter_count, solver_time_equivalence_solved,
        ) = _solve_teq_once(fire_load_density=_fire_load_density_, **once_kwargs)

        # Timber outputs take their "not applicable" values.
        timber_exposed_duration = 0
        timber_solver_iter_count = 0
        timber_fire_load = np.nan

    else:
        # ----- Timber add-on: iterate the time-equivalence solve to convergence. -----
        # Timber releases energy at ``timber_burning_rate`` [MJ/s] for as long as it is exposed
        # (``timber_exposed_duration`` == the solved time equivalence), capped at a total
        # ``timber_fire_load_max`` [MJ]. That extra energy raises the fire load density, which
        # changes the fire, which changes the solved duration -- hence the iteration.
        timber_solver_iter_count = -1
        timber_exposed_duration = 0  # initial condition, timber exposed duration
        room_floor_area = room_breadth * room_depth

        while True:
            timber_solver_iter_count += 1

            # timber energy released over the assumed exposed duration, capped at the max
            timber_fire_load = timber_burning_rate * timber_exposed_duration
            if timber_fire_load_max is not None:
                timber_fire_load = min(timber_fire_load, timber_fire_load_max)

            fire_load_density = _fire_load_density_ + timber_fire_load / room_floor_area

            (
                fire_type, t1, t2, t3,
                solver_steel_temperature_solved, solver_time_critical_temp_solved, solver_protection_thickness,
                solver_iter_count, solver_time_equivalence_solved,
            ) = _solve_teq_once(fire_load_density=fire_load_density, **once_kwargs)

            if timber_solver_iter_count >= timber_solver_ilim:
                solver_time_critical_temp_solved = np.nan
                solver_time_equivalence_solved = np.nan
                solver_steel_temperature_solved = np.nan
                solver_protection_thickness = np.nan
                solver_iter_count = np.nan
                timber_exposed_duration = np.nan
                break
            elif not -np.inf < solver_protection_thickness < np.inf:
                # no protection thickness solution
                timber_exposed_duration = solver_protection_thickness
                break
            elif abs(timber_exposed_duration - solver_time_equivalence_solved) <= timber_solver_tol:
                # convergence sought successfully
                break
            else:
                timber_exposed_duration = solver_time_equivalence_solved

    return TeqResult(
        fire_type, t1, t2, t3,
        solver_steel_temperature_solved, solver_time_critical_temp_solved, solver_protection_thickness,
        solver_iter_count, solver_time_equivalence_solved,
        timber_exposed_duration, timber_solver_iter_count, timber_fire_load,
    )
