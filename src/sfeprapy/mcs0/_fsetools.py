"""Vendored fire/heat-transfer physics.

This module replaces the ``fsetools`` dependency with self-contained, pure-Python
implementations of the routines used by :mod:`sfeprapy.mcs0.calcs`.

The parametric and travelling fire temperature functions are adapted from
``fsetools`` (pure Python, ``numpy`` only). The steel heat-transfer routines
(``temperature``, ``temperature_max``, ``protection_thickness_2``) are a
pure-Python port of the former Cython module ``fse_bs_en_1993_1_2_heat_transfer_c``
-- the algorithm is identical; only the C type annotations have been removed.
"""

import copy
from typing import Union

import numpy as np

__all__ = (
    "temperature",           # steel temperature history (BS EN 1993-1-2)
    "temperature_max",       # peak steel temperature + time
    "protection_thickness_2",  # solve protection thickness for a goal steel temperature
    "parametric_fire_temperature",       # BS EN 1991-1-2 parametric fire
    "travelling_fire_temperature",        # travelling fire
)


# =====================================================================================
# BS EN 1993-1-2 protected steel heat transfer (pure-Python port of the Cython module)
# =====================================================================================

def c_steel_T(T: float) -> float:
    """Specific heat of carbon steel [J/kg/K] as a function of temperature [K].

    BS EN 1993-1-2:2005, 3.4.1.2. Piecewise correlation, implemented exactly as the
    original Cython ``cdef double c_steel_T(double T)``.
    """
    T = T - 273.15  # K -> degC
    if T < 20:
        return 425 + 0.773 * 20 - 1.69e-3 * 400 + 2.22e-6 * 8000
    elif T < 600:
        return 425 + 0.773 * T - 1.69e-3 * (T ** 2) + 2.22e-6 * (T ** 3)
    elif T < 735:
        return 666 + 13002 / (738 - T)
    elif T < 900:
        return 545 + 17820 / (T - 731)
    else:
        return 650


def temperature(
        fire_time,
        fire_temperature,
        beam_rho: float,
        beam_cross_section_area: float,
        protection_k: float,
        protection_rho: float,
        protection_c: float,
        protection_thickness: float,
        protection_protected_perimeter: float,
        **__,
) -> np.ndarray:
    """Calculate the steel temperature history for a protected steel member [K].

    SI units throughout. BS EN 1993-1-2:2005, Clauses 4.2.5.2 (Eq. 4.27).

    :param fire_time:                       Time array [s]
    :param fire_temperature:                Gas temperature array [K]
    :param beam_rho:                        Steel beam density [kg/m3]
    :param beam_cross_section_area:         Steel beam cross sectional area [m2]
    :param protection_k:                    Protection thermal conductivity [W/m/K]
    :param protection_rho:                  Protection density [kg/m3]
    :param protection_c:                    Protection specific heat capacity [J/kg/K]
    :param protection_thickness:            Protection layer thickness [m]
    :param protection_protected_perimeter:  Protection protected perimeter [m]
    :return:                                Steel beam temperature array [K]
    """
    V = beam_cross_section_area
    rho_a = beam_rho
    lambda_p = protection_k
    rho_p = protection_rho
    d_p = protection_thickness
    A_p = protection_protected_perimeter
    c_p = protection_c

    fire_time = np.asarray(fire_time, dtype=np.float64)
    fire_temperature = np.asarray(fire_temperature, dtype=np.float64)

    T_a = np.zeros(len(fire_time), dtype=np.float64)

    # Check time step <= 30 seconds. [BS EN 1993-1-2:2005, Clauses 4.2.5.2 (3)]
    T_a[0] = fire_temperature[0]  # assign steel initial temperature

    for i in range(1, len(fire_time)):
        T_g = fire_temperature[i]

        c_s = c_steel_T(T_a[i - 1])

        # Steel temperature equations are from [BS EN 1993-1-2:2005, Clauses 4.2.5.2, Eq. 4.27]
        phi = (c_p * rho_p / c_s / rho_a) * d_p * A_p / V

        a = (lambda_p * A_p / V) / (d_p * c_s * rho_a)
        b = (T_g - T_a[i - 1]) / (1.0 + phi / 3.0)
        c = (2.718 ** (phi / 10.0) - 1.0) * (T_g - fire_temperature[i - 1])
        d = fire_time[i] - fire_time[i - 1]

        dT = (a * b * d - c) / d  # deviated from e4.27, converted to rate [s-1]
        if dT < 0 < (T_g - fire_temperature[i - 1]):
            dT = 0

        T_a[i] = T_a[i - 1] + dT * d

        # NOTE: Steel temperature can be in cooling phase at the beginning of calculation,
        #       even the ambient (fire) temperature is hot. This is due to the factor 'phi'
        #       which intends to address the energy locked within the protection layer. The
        #       steel temperature is forced to be increased or remain as previous when ambient
        #       temperature and its previous temperature are all higher than the current
        #       calculated temperature. A better implementation is perhaps to use a 1-D heat
        #       transfer model.

    return T_a


def temperature_max(
        fire_time,
        fire_temperature,
        beam_rho: float,
        beam_cross_section_area: float,
        protection_k: float,
        protection_rho: float,
        protection_c: float,
        protection_thickness: float,
        protection_protected_perimeter: float,
):
    """Calculate the maximum steel temperature and the time it occurs for a protected member.

    SI units throughout. BS EN 1993-1-2:2005.

    LIMITATIONS:
        1. Constant time interval in ``fire_time`` throughout;
        2. ``fire_temperature`` has *one* maxima.

    :return: ``(T_a_max [K], t_at_max [s])``
    """
    V = beam_cross_section_area
    rho_a = beam_rho
    lambda_p = protection_k
    rho_p = protection_rho
    d_p = protection_thickness
    A_p = protection_protected_perimeter
    c_p = protection_c

    fire_time = np.asarray(fire_time, dtype=np.float64)
    fire_temperature = np.asarray(fire_temperature, dtype=np.float64)

    T = fire_temperature[0]  # current steel temperature
    d = fire_time[1] - fire_time[0]

    i = 1
    for i in range(1, len(fire_temperature)):
        T_g = fire_temperature[i]

        c_s = c_steel_T(T)

        # Steel temperature equations are from [BS EN 1993-1-2:2005, Clauses 4.2.5.2, Eq. 4.27]
        # If below get divide by zero error, it's very likely due to T = nan and causing c_s = 0
        phi = (c_p * rho_p / c_s / rho_a) * d_p * A_p / V

        a = (lambda_p * A_p / V) / (d_p * c_s * rho_a)
        b = (T_g - T) / (1.0 + phi / 3.0)
        c = (2.718 ** (phi / 10.0) - 1.0) * (T_g - fire_temperature[i - 1])

        dT = (a * b * d - c) / d  # deviated from e4.27, converted to rate [s-1]
        if dT < 0 < (T_g - fire_temperature[i - 1]):
            dT = 0

        T = T + dT * d

        # Terminate early if maximum temperature is reached
        if dT < 0:
            T -= dT * d
            break

    return T, fire_time[i - 1]


def protection_thickness_2(
        fire_time,
        fire_temperature,
        beam_rho: float,
        beam_cross_section_area: float,
        protection_k: float,
        protection_rho: float,
        protection_c: float,
        protection_protected_perimeter: float,
        solver_temperature_goal: float,            # Target max steel temperature [K]
        solver_temperature_goal_tol: float,        # Tolerance [K]
        solver_max_iter: int = 100,
        d_p_1: float = 0.0001,                     # Lower bound of protection thickness [m]
        d_p_2: float = 0.0900,                     # Upper bound of protection thickness [m]
        d_p_i: float = 0.0010,                     # Step size for initial linear search [m]
):
    """Find protection thickness ``d_p`` so the max steel temperature is near the goal.

    Assumes ``T_a_max`` monotonically decreases as ``d_p`` increases. Uses a linear step
    search from ``d_p_1`` then a binary search to refine.

    LIMITATIONS:
        1. Constant time interval in ``fire_time`` throughout;
        2. ``fire_temperature`` has *one* maxima;
        3. Requires ``T_a_max`` to be MONOTONIC DECREASING with ``protection_thickness``.

    :return: tuple ``(d_p, T_a_max, t_at_max, iter_count, status)`` where ``status`` is:
                        0: Success
                        1: Out of Lower Bound (temp at d_p_1 already too low)
                        2: Out of Upper Bound (temp at d_p_2 still too high)
                        3: Max Iterations Reached (returned value is best found)
                        4: Monotonicity Failed (T_max increased unexpectedly with increased d_p)
    """
    fire_time = np.asarray(fire_time, dtype=np.float64)
    fire_temperature = np.asarray(fire_temperature, dtype=np.float64)

    # Status constants
    STATUS_SUCCESS = 0
    STATUS_OUT_OF_LOWER_BOUND = 1
    STATUS_OUT_OF_UPPER_BOUND = 2
    STATUS_MAX_ITERATIONS_REACHED = 3
    STATUS_MONOTONICITY_FAILED = 4

    # Input validation (basic)
    if d_p_1 < 0 or d_p_2 <= d_p_1 or d_p_i <= 0:
        raise ValueError("Invalid bounds or step size (d_p_1 >= 0, d_p_2 > d_p_1, d_p_i > 0 required)")
    if solver_temperature_goal_tol <= 0:
        raise ValueError("Solver tolerance must be positive")
    if solver_max_iter < 2:
        raise ValueError("Solver max iterations must be at least 2")
    if fire_time.shape[0] == 0 or fire_time.shape[0] != fire_temperature.shape[0]:
        raise ValueError("fire_time and fire_temperature must be non-empty and have the same length")

    # Result tracking variables (initialize with values from d_p_1)
    best_d_p = d_p_1
    best_T = 0.0  # Will be overwritten by first call
    best_t = 0.0  # Will be overwritten by first call
    min_abs_diff_found = 1e18  # Initialize with a large value
    total_iter_count = 0

    # Individual iteration variables
    d_p_low = -1.0  # Sentinel value indicating bracket not yet found
    d_p_high = -1.0  # Sentinel value

    # --- Initial Check at Lower Bound (d_p_1) ---
    T_current, t_current = temperature_max(
        fire_time, fire_temperature, beam_rho, beam_cross_section_area,
        protection_k, protection_rho, protection_c,
        d_p_1, protection_protected_perimeter,
    )
    total_iter_count += 1

    # Initialise best solution tracking using the first result
    min_abs_diff_found = abs(T_current - solver_temperature_goal)
    best_d_p = d_p_1
    best_T = T_current
    best_t = t_current

    # Check if T(d_p_1) is already too low (below target - tolerance)
    if T_current < solver_temperature_goal - solver_temperature_goal_tol:
        return best_d_p, best_T, best_t, total_iter_count, STATUS_OUT_OF_LOWER_BOUND

    # Check if T(d_p_1) is within tolerance
    if T_current <= solver_temperature_goal + solver_temperature_goal_tol:
        return best_d_p, best_T, best_t, total_iter_count, STATUS_SUCCESS

    # --- Linear Step Search (from d_p_1 + d_p_i up to d_p_2) ---
    d_p_previous = d_p_1
    T_previous = T_current  # Store result from d_p_1
    t_previous = t_current
    d_p_current = d_p_1

    while True:
        # Check iteration count before potentially expensive calculation
        if total_iter_count >= solver_max_iter:
            return best_d_p, best_T, best_t, total_iter_count, STATUS_MAX_ITERATIONS_REACHED

        # Calculate next d_p, clamped to d_p_2
        d_p_current = d_p_previous + d_p_i
        if d_p_current >= d_p_2:
            d_p_current = d_p_2

        # Avoid infinite loop if stuck at d_p_2 (e.g., if d_p_i is tiny)
        if d_p_current == d_p_previous:
            # This means we are at d_p_2. Exit loop to check final T.
            break

        # Solve T for current d_p
        T_current, t_current = temperature_max(
            fire_time, fire_temperature, beam_rho, beam_cross_section_area,
            protection_k, protection_rho, protection_c,
            d_p_current, protection_protected_perimeter,
        )
        total_iter_count += 1

        # --- Monotonicity Check ---
        if T_current > T_previous:
            # Temperature increased unexpectedly! Violates assumption.
            return d_p_previous, T_previous, t_previous, total_iter_count, STATUS_MONOTONICITY_FAILED
        # --- End Monotonicity Check ---

        # Update best solution found so far (closest to goal)
        current_diff = abs(T_current - solver_temperature_goal)
        if current_diff < min_abs_diff_found:
            min_abs_diff_found = current_diff
            best_d_p = d_p_current
            best_T = T_current
            best_t = t_current

        # Check if T_current is now low enough to bracket the solution or hit target
        if T_current <= solver_temperature_goal + solver_temperature_goal_tol:
            # We have found a bracket: [d_p_previous, d_p_current]
            d_p_low = d_p_previous
            d_p_high = d_p_current
            break

        # Prepare for next iteration of linear search
        d_p_previous = d_p_current
        T_previous = T_current
        t_previous = t_current

    # --- Post Linear Search ---

    # Case 1: Did we exit because d_p reached d_p_2?
    if d_p_current == d_p_2 and d_p_low < 0:  # Bracket not found via T <= goal+tol check
        if T_current > solver_temperature_goal + solver_temperature_goal_tol:
            # Even at max thickness d_p_2, the temperature is still too high
            return best_d_p, best_T, best_t, total_iter_count, STATUS_OUT_OF_UPPER_BOUND
        else:
            d_p_low = d_p_previous
            d_p_high = d_p_current  # d_p_current == d_p_2 here

    # Case 2: We exited because a bracket [d_p_low, d_p_high] was found.
    # Proceed only if a valid bracket was established (d_p_low >= 0)
    if d_p_low >= 0 and d_p_low < d_p_high:
        # --- Binary Search Refinement ---
        for i in range(total_iter_count, solver_max_iter):  # Count total iterations correctly
            d_p_mid = d_p_low + 0.5 * (d_p_high - d_p_low)

            # Check if interval is already tiny (machine precision or negligible difference)
            if (d_p_high - d_p_low) < 1e-12:
                (T_mid, t_mid) = temperature_max(
                    fire_time, fire_temperature, beam_rho, beam_cross_section_area,
                    protection_k, protection_rho, protection_c,
                    d_p_mid, protection_protected_perimeter,
                )
                total_iter_count += 1
                mid_diff = abs(T_mid - solver_temperature_goal)
                if mid_diff < min_abs_diff_found:
                    return d_p_mid, T_mid, t_mid, total_iter_count, STATUS_SUCCESS
                else:
                    return best_d_p, best_T, best_t, total_iter_count, STATUS_SUCCESS

            # Evaluate temperature at midpoint
            (T_current, t_current) = temperature_max(
                fire_time, fire_temperature, beam_rho, beam_cross_section_area,
                protection_k, protection_rho, protection_c,
                d_p_mid, protection_protected_perimeter,
            )
            total_iter_count += 1

            # Update best solution tracking during binary search
            current_diff = abs(T_current - solver_temperature_goal)
            if current_diff < min_abs_diff_found:
                min_abs_diff_found = current_diff
                best_d_p = d_p_mid
                best_T = T_current
                best_t = t_current

            # Check if solution is within tolerance [goal - tol, goal + tol]
            if T_current <= solver_temperature_goal + solver_temperature_goal_tol and \
               T_current >= solver_temperature_goal - solver_temperature_goal_tol:
                return d_p_mid, T_current, t_current, total_iter_count, STATUS_SUCCESS

            # Update binary search bounds based on midpoint temperature
            if T_current > solver_temperature_goal:
                # Temp too high, need thicker protection -> increase lower bound
                d_p_low = d_p_mid
            else:
                # Temp too low, need thinner protection -> decrease upper bound
                d_p_high = d_p_mid

            if total_iter_count >= solver_max_iter:
                return best_d_p, best_T, best_t, total_iter_count, STATUS_MAX_ITERATIONS_REACHED

        # If binary search loop finishes without converging
        return best_d_p, best_T, best_t, total_iter_count, STATUS_MAX_ITERATIONS_REACHED

    # --- Fallback / Unexpected Exit ---
    final_status = STATUS_MAX_ITERATIONS_REACHED
    if d_p_current == d_p_2 and best_T > solver_temperature_goal + solver_temperature_goal_tol:
        final_status = STATUS_OUT_OF_UPPER_BOUND
    elif abs(best_T - solver_temperature_goal) <= solver_temperature_goal_tol:
        final_status = STATUS_SUCCESS

    return best_d_p, best_T, best_t, total_iter_count, final_status


# =====================================================================================
# BS EN 1991-1-2 Appendix A parametric fire (verbatim from fsetools, numpy only)
# =====================================================================================

def _eq_3_12_T_g(t_star, T_0: float = 20):
    # eq. 3.12
    T_g = 1325 * (1 - 0.324 * np.exp(-0.2 * t_star) - 0.204 * np.exp(-1.7 * t_star) - 0.472 * np.exp(-19 * t_star))
    T_g += T_0
    return T_g


def _eq_3_16_T_g(t_star_max, T_max, t_star):  # ventilation controlled
    # eq. 3.16
    if t_star_max <= 0.5:
        T_g = T_max - 625 * (t_star - t_star_max)
    elif 0.5 < t_star_max < 2.0:
        T_g = T_max - 250 * (3 - t_star_max) * (t_star - t_star_max)
    elif 2.0 <= t_star_max:
        T_g = T_max - 250 * (t_star - t_star_max)
    else:
        T_g = np.nan
    return T_g


def _eq_3_22_T_g(t_star_max, T_max, t_star, Gamma, t_lim):  # fuel controlled
    # eq. 3.22
    if t_star_max <= 0.5:
        T_g = T_max - 625 * (t_star - Gamma * t_lim)
    elif 0.5 < t_star_max < 2.0:
        T_g = T_max - 250 * (3 - t_star_max) * (t_star - Gamma * t_lim)
    elif 2.0 <= t_star_max:
        T_g = T_max - 250 * (t_star - Gamma * t_lim)
    else:
        T_g = np.nan
    return T_g


def _variables_1(t, Gamma, t_max):
    t_star = Gamma * t
    t_star_max = Gamma * t_max
    return t_star, t_star_max


def _variables_2(t, t_lim, q_td, b, O):
    O_lim = 0.0001 * q_td / t_lim
    Gamma_lim = ((O_lim / 0.04) / (b / 1160)) ** 2

    if O > 0.04 and q_td < 75 and b < 1160:
        k = 1 + ((O - 0.04) / (0.04)) * ((q_td - 75) / (75)) * ((1160 - b) / (1160))
        Gamma_lim *= k

    t_star_ = Gamma_lim * t
    t_star_max_ = Gamma_lim * t_lim
    return t_star_, t_star_max_


def parametric_fire_temperature(
        t: np.ndarray, A_t: float, A_f: float, A_v: float, h_eq: float, q_fd: float, lbd: float, rho: float,
        c: float, t_lim: float, T_0: float = 293.15):
    """Time-temperature curve according to Eurocode 1 part 1-2, Appendix A.

    :param t: numpy.ndarray, [s], time evolution.
    :param A_t:     [m2], total surface area (including openings).
    :param A_f:     [m2], floor area.
    :param A_v:     [m2], opening area.
    :param h_eq:    [m2], opening height.
    :param q_fd:    [J/m2], fuel density.
    :param lbd:     [K/kg/m], lining thermal conductivity.
    :param rho:     [kg/m3], lining density.
    :param c:       [J/K/kg], lining thermal capacity.
    :param t_lim:   [s], limiting time for the fire.
    :return T_g:    [K], temperature evolution.
    """
    # Reference: Eurocode 1991-1-2; Jean-Marc Franssen, Paulo Vila Real (2010) - Fire Design of Steel Structures

    # Convert units SI -> Local
    q_fd = q_fd / 1e6  # [J/m2] -> [MJ/m2]
    t_lim = t_lim / 3600  # [s] -> [hr]
    t = t / 3600  # [s] -> [hr]
    T_0 = T_0 - 273.15  # [K] -> [C]

    # ACQUIRING REQUIRED VARIABLES
    b = (lbd * rho * c) ** 0.5  # thermal inertia
    O = A_v * h_eq ** 0.5 / A_t  # opening factor
    q_td = q_fd * A_f / A_t  # total fuel load
    Gamma = ((O / 0.04) / (b / 1160)) ** 2

    t_max = 0.0002 * q_td / O

    t_star, t_star_max = _variables_1(t, Gamma, t_max)

    if t_max >= t_lim:  # ventilation controlled fire
        T_max = _eq_3_12_T_g(t_star_max, T_0)
        T_g_heating = _eq_3_12_T_g(Gamma * t, T_0)
        T_g_cooling = _eq_3_16_T_g(t_star_max, T_max, t_star)
    else:  # fuel controlled fire
        t_star_f, t_star_max_f = _variables_2(t, t_lim, q_td, b, O)
        T_max = _eq_3_12_T_g(t_star_max_f, T_0)
        T_g_heating = _eq_3_12_T_g(t_star_f, T_0)
        T_g_cooling = _eq_3_22_T_g(t_star_max, T_max, t_star, Gamma, t_lim)

    T_g = np.minimum(T_g_heating, T_g_cooling)
    T_g[T_g < T_0] = T_0

    # UNITS: Eq. -> SI
    T_g += 273.15

    return T_g


# =====================================================================================
# Travelling fire (verbatim from fsetools, numpy only)
# =====================================================================================

def travelling_fire_temperature(
        t: np.array,
        fire_load_density_MJm2: float,
        fire_hrr_density_MWm2: float,
        room_length_m: float,
        room_width_m: float,
        fire_spread_rate_ms: float,
        beam_location_height_m: float,
        beam_location_length_m: Union[float, list, np.ndarray],
        fire_nft_limit_c: float,
        *_,
        **__,
):
    """Calculate a travelling fire temperature array [degC].

    NOTE: NOT in SI units (see param units below).

    :param t: in s, the time array
    :param fire_load_density_MJm2: in MJ/m2, fuel density on the floor
    :param fire_hrr_density_MWm2: in MW/m2, heat release rate density
    :param room_length_m: in m, room length
    :param room_width_m: in m, room width
    :param fire_spread_rate_ms: in m/s, fire spread speed
    :param beam_location_height_m: in m, beam height above the floor
    :param beam_location_length_m: in m, beam lateral distance to fire origin
    :param fire_nft_limit_c: in degC, maximum near field temperature
    :return: in degC, calculated gas temperature
    """
    # re-assign variable names for equation readability
    q_fd = fire_load_density_MJm2
    HRRPUA = fire_hrr_density_MWm2
    s = fire_spread_rate_ms
    h_s = beam_location_height_m
    l_s = beam_location_length_m
    l = room_length_m
    w = room_width_m
    if l < w:
        l += w
        w = l - w
        l -= w

    # workout burning time etc.
    t_burn = max([q_fd / HRRPUA, 900.0])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    time_interval_s = t[1] - t[0]
    t_decay_ = round(t_decay / time_interval_s, 0) * time_interval_s
    t_lim_ = round(t_lim / time_interval_s, 0) * time_interval_s
    if t_decay_ == t_lim_:
        t_lim_ -= time_interval_s

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (HRRPUA * w * s * t) * (t < t_lim_)
    Q_peak = (
            min([HRRPUA * w * s * t_burn, HRRPUA * w * l]) * (t >= t_lim_) * (t <= t_decay_)
    )
    Q_decay = (max(Q_peak) - (t - t_decay_) * w * s * HRRPUA) * (t > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.0

    # workout the distance between fire median to the structural element r
    l_fire_front = s * t
    l_fire_front[l_fire_front < 0] = 0
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (t - t_lim)
    l_fire_end[l_fire_end < 0] = 0.0
    l_fire_end[l_fire_end > l] = l
    l_fire_median = (l_fire_front + l_fire_end) / 2.0

    # workout the far field temperature of gas T_g
    if isinstance(l_s, float) or isinstance(l_s, int):
        r = np.absolute(l_s - l_fire_median)
        T_g = np.where((r / h_s) > 0.18, (5.38 * np.power(Q / r, 2 / 3) / h_s) + 20.0, 0)
        T_g = np.where((r / h_s) <= 0.18, (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5 / 3)) + 20.0, T_g)
        T_g[T_g >= fire_nft_limit_c] = fire_nft_limit_c
        return T_g
    elif isinstance(l_s, np.ndarray) or isinstance(l_s, list):
        l_s_list = copy.copy(l_s)
        T_g_list = list()
        for l_s in l_s_list:
            r = np.absolute(l_s - l_fire_median)
            T_g = np.where((r / h_s) > 0.18, (5.38 * np.power(Q / r, 2 / 3) / h_s) + 20.0, 0)
            T_g = np.where((r / h_s) <= 0.18, (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5 / 3)) + 20.0, T_g)
            T_g[T_g >= fire_nft_limit_c] = fire_nft_limit_c
            T_g_list.append(T_g)
        return T_g_list
    else:
        raise TypeError('Unknown type of parameter "l_s": {}'.format(type(l_s)))
