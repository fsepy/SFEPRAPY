__all__ = (
    'decide_fire', 'evaluate_fire_temperature', 'solve_time_equivalence_iso834', 'solve_protection_thickness',
    'teq_main', 'TeqResult',
)

from typing import NamedTuple

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
# fire can spread across the whole compartment. (Travelling-fire modeling floor,
# not an EC clause.)
MIN_BURNOUT_TIME = 900.0  # [s]


class _Compartment(NamedTuple):
    """Derived geometric/ventilation quantities shared by fire-type selection and the
    fire-curve evaluation. Computed once from the raw inputs to avoid recomputing
    and diverging between :func:`decide_fire` and :func:`evaluate_fire_temperature`."""
    window_area: float                  # [m2] ventilation opening area
    room_floor_area: float              # [m2]
    room_total_area: float              # [m2] internal surface area incl. openings
    fire_load_density_deducted: float   # [MJ/m2] after combustion efficiency
    fire_load_density_total: float      # [MJ/m2] related to A_t
    opening_factor: float               # [m^0.5*s^0.5]


def _compartment_params(
        window_height: float, window_width: float,
        room_breadth: float, room_depth: float, room_height: float,
        fire_load_density: float, fire_combustion_efficiency: float,
) -> _Compartment:
    """Compute the shared compartment/ventilation derived quantities."""
    room_floor_area = room_breadth * room_depth
    room_total_area = 2 * room_floor_area + (room_breadth + room_depth) * 2 * room_height
    window_area = window_height * window_width
    fire_load_density_deducted = fire_load_density * fire_combustion_efficiency
    fire_load_density_total = fire_load_density_deducted * room_floor_area / room_total_area
    opening_factor = window_area * np.sqrt(window_height) / room_total_area
    return _Compartment(
        window_area=window_area, room_floor_area=room_floor_area, room_total_area=room_total_area,
        fire_load_density_deducted=fire_load_density_deducted,
        fire_load_density_total=fire_load_density_total, opening_factor=opening_factor,
    )


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
    """Decide which design fire type to use for the given compartment and fire parameters.

    Returns an int fire type: 0 = EC parametric fire, 1 = travelling fire.

    PARAMETERS:
    :param window_height:               [m], weighted window opening height
    :param window_width:                [m], total window opening width (the ventilation opening geometry directly)
    :param room_breadth:                [m], room breadth (shorter direction of the floor plan)
    :param room_depth:                  [m], room depth (longer direction of the floor plan)
    :param room_height:                 [m], room height from floor to soffit (structural), disregard any non fire resisting floors
    :param fire_hrr_density:            [MW/m2], fire maximum release rate per unit area
    :param fire_load_density:           [MJ/m2], fire load density related to floor area
    :param fire_combustion_efficiency:  [-]
    :param fire_spread_speed:           [m/s], TRAVELLING FIRE, fire spread speed
    :param fire_mode:                   0 - parametric, 1 - travelling, 3 - (0 & 1) auto-selected
    :return:                            int fire type (0 or 1)
    """

    # PERMEABLE AND INPUT CHECKS

    p = _compartment_params(
        window_height=window_height, window_width=window_width,
        room_breadth=room_breadth, room_depth=room_depth, room_height=room_height,
        fire_load_density=fire_load_density, fire_combustion_efficiency=fire_combustion_efficiency,
    )

    # Spread speed - Does the fire spread to involve the full compartment?
    fire_spread_entire_room_time = room_depth / fire_spread_speed
    burn_out_time = max([p.fire_load_density_deducted / fire_hrr_density, MIN_BURNOUT_TIME])

    if fire_mode == 0 or fire_mode == 1:
        # enforced to selected fire, i.e. 0 is ec parametric; 1 is travelling
        fire_type = fire_mode
    elif fire_mode == 3:
        # enforced to ec parametric + travelling
        if (
                fire_spread_entire_room_time < burn_out_time
                and OPENING_FACTOR_LBOUND_EC < p.opening_factor <= OPENING_FACTOR_UBOUND_EC
                and FIRE_LOAD_DENSITY_TOTAL_LBOUND_EC <= p.fire_load_density_total <= FIRE_LOAD_DENSITY_TOTAL_UBOUND_EC
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
    :param fire_type:                   [-], 0 = parametric, 1 = travelling
    :param fire_time:                   [s], time array
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

    p = _compartment_params(
        window_height=window_height, window_width=window_width,
        room_breadth=room_breadth, room_depth=room_depth, room_height=room_height,
        fire_load_density=fire_load_density, fire_combustion_efficiency=fire_combustion_efficiency,
    )

    if fire_type == 0:
        fire_temperature = _fire_param(
            t=fire_time,
            A_t=p.room_total_area,
            A_f=p.room_floor_area,
            A_v=p.window_area,
            h_eq=window_height,
            q_fd=p.fire_load_density_deducted * 1e6,
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
            fire_load_density_MJm2=p.fire_load_density_deducted,
            fire_hrr_density_MWm2=fire_hrr_density,
            room_length_m=room_depth,
            room_width_m=room_breadth,
            fire_spread_rate_ms=fire_spread_speed,
            beam_location_height_m=beam_position_vertical,
            beam_location_length_m=beam_position_horizontal,
            fire_nft_limit_c=fire_nft_limit - 273.15,
        )
        fire_temperature = fire_travelling(**kwargs_fire_1_travel) + 273.15

        t1 = min(room_depth / fire_spread_speed, p.fire_load_density_deducted / fire_hrr_density)
        t2 = max(room_depth / fire_spread_speed, p.fire_load_density_deducted / fire_hrr_density)
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
    Solve the equivalent time exposure to the ISO 834 standard fire for a protected steel
    element, given a solved protection thickness and target critical (failure) temperature.

    PARAMETERS:
    :param fire_time:                           [s], time array
    :param beam_cross_section_area:             [m2], the steel beam element cross section area
    :param beam_rho:                            [kg/m3], steel beam element density
    :param protection_k:                        [W/m/K], protection material thermal conductivity
    :param protection_rho:                      [kg/m3], protection material density
    :param protection_c:                        [J/kg/K], protection material specific heat
    :param protection_protected_perimeter:      [m], protection material protected perimeter
    :param solver_temperature_goal:             [K], steel beam element expected failure temperature
    :param solver_protection_thickness:         [m], steel section protection layer thickness
    :return:                                    [s], solved equivalent time exposure (np.nan/inf if unsolvable)
    """

    # ============================================
    # GOAL SEEK TO MATCH STEEL FAILURE TEMPERATURE
    # ============================================

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS

    # Solve equivalent time exposure in ISO 834
    solver_d_p = solver_protection_thickness

    # Handle non-finite protection thickness first: NaN must be checked before inf
    # (any equality test against NaN is False, so `== inf`/`== -inf` would miss it).
    if np.isnan(solver_d_p):
        return np.nan
    if not np.isfinite(solver_d_p):
        # +inf -> infinite time to reach critical temp; -inf -> not physically meaningful
        return float(solver_d_p)

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
        solver_time_equivalence_solved = np.interp(solver_temperature_goal, steel_temperature, fire_time)

    return solver_time_equivalence_solved


def solve_protection_thickness(
        fire_time: np.ndarray,
        fire_temperature: np.ndarray,
        beam_cross_section_area: float,
        beam_rho: float,
        protection_k: float,
        protection_rho: float,
        protection_c: float,
        protection_protected_perimeter: float,
        solver_temperature_goal: float,
        solver_tol: float,
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
        d_p_i=0.00375,  # fixed mid-range step (was random jitter; deterministic for reproducibility)
    )

    if solver_status == 1:
        # out of lower bound: temp at d_p_1 already too low -> infinite protection
        return -np.inf, solver_t, solver_d_p, solver_iter_count
    elif solver_status == 2:
        # out of upper bound: temp at d_p_2 still too high -> no finite protection
        return np.inf, solver_t, solver_d_p, solver_iter_count
    elif solver_status == 3:
        # max iterations reached without convergence
        return np.nan, np.nan, np.nan, solver_iter_count
    # status 0 (success) or 4 (monotonicity failed -- last valid point): return best available
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
) -> TeqResult:
    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    if room_depth < room_breadth:
        room_depth += room_breadth
        room_breadth = room_depth - room_breadth
        room_depth -= room_breadth

    # Fix ventilation opening size, so it doesn't exceed wall area
    if window_height > room_height:
        window_height = room_height

    # Calculate fire time, this is used for all fire curves in the calculation.
    # Use linspace (not arange) to avoid float-step drift in the array length.
    n_steps = int(round(fire_time_duration / fire_time_step))
    fire_time = np.linspace(0.0, fire_time_duration, n_steps + 1)

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

            # timber energy released over the assumed exposed duration, capped at the max.
            # timber_fire_load is GROSS fuel energy [MJ] -- it is added to the base fire
            # load *before* fire_combustion_efficiency is applied (that discounting happens
            # inside _compartment_params, downstream). So a single efficiency factor scales
            # the combined (contents + timber) fuel, consistent with both fuel sources
            # burning incompletely.
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
