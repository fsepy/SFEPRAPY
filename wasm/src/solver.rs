//! Protection-thickness solver. Port of `protection_thickness_2` from `sfeprapy._fsetools`.
//!
//! Finds the protection thickness `d_p` such that the peak steel temperature is within
//! `tol` of the goal. Linear step search then binary-search refinement. Assumes peak
//! steel temperature monotonically *decreases* as `d_p` increases.

use crate::steel::temperature_max;

/// Status codes (match the Python exactly).
pub const STATUS_SUCCESS: i32 = 0;
pub const STATUS_OUT_OF_LOWER_BOUND: i32 = 1;
pub const STATUS_OUT_OF_UPPER_BOUND: i32 = 2;
pub const STATUS_MAX_ITERATIONS_REACHED: i32 = 3;
pub const STATUS_MONOTONICITY_FAILED: i32 = 4;

/// Result of [`protection_thickness_2`].
pub struct SolverResult {
    pub d_p: f64,
    pub t_a_max: f64,
    pub t_at_max: f64,
    pub iter_count: i32,
    pub status: i32,
}

/// Solve protection thickness so peak steel temp ≈ `solver_temperature_goal`.
///
/// Returns `(d_p, T_a_max, t_at_max, iter_count, status)`. Algorithmically identical to
/// the Python `protection_thickness_2`; the caller (`solve_protection_thickness`) hardcodes
/// `d_p_1=0.0001, d_p_2=0.0801, d_p_i=0.00375, max_iter=100`.
#[allow(clippy::too_many_arguments)]
pub fn protection_thickness_2(
    fire_time: &[f64],
    fire_temperature: &[f64],
    beam_rho: f64,
    beam_cross_section_area: f64,
    protection_k: f64,
    protection_rho: f64,
    protection_c: f64,
    protection_protected_perimeter: f64,
    solver_temperature_goal: f64,
    solver_temperature_goal_tol: f64,
    solver_max_iter: i32,
    d_p_1: f64,
    d_p_2: f64,
    d_p_i: f64,
) -> SolverResult {
    // Input validation (matches Python; panics <=> Python's ValueError).
    assert!(d_p_1 >= 0.0, "Invalid bounds: d_p_1 >= 0 required");
    assert!(d_p_2 > d_p_1, "Invalid bounds: d_p_2 > d_p_1 required");
    assert!(d_p_i > 0.0, "Invalid step: d_p_i > 0 required");
    assert!(solver_temperature_goal_tol > 0.0, "tol must be positive");
    assert!(solver_max_iter >= 2, "max_iter must be >= 2");
    assert!(!fire_time.is_empty() && fire_time.len() == fire_temperature.len(),
            "arrays non-empty and equal length");

    let mut best_d_p = d_p_1;
    let mut best_t = 0.0_f64;
    let mut best_t_time = 0.0_f64;
    let mut min_abs_diff_found = 1e18_f64;
    let mut total_iter_count = 0_i32;

    let mut d_p_low = -1.0_f64;
    let mut d_p_high = -1.0_f64;

    // --- Initial check at lower bound d_p_1 ---
    let (mut t_current, mut t_current_time) = temperature_max(
        fire_time, fire_temperature, beam_rho, beam_cross_section_area,
        protection_k, protection_rho, protection_c, d_p_1, protection_protected_perimeter,
    );
    total_iter_count += 1;

    min_abs_diff_found = (t_current - solver_temperature_goal).abs();
    best_d_p = d_p_1;
    best_t = t_current;
    best_t_time = t_current_time;

    if t_current < solver_temperature_goal - solver_temperature_goal_tol {
        return SolverResult { d_p: best_d_p, t_a_max: best_t, t_at_max: best_t_time,
                              iter_count: total_iter_count, status: STATUS_OUT_OF_LOWER_BOUND };
    }
    if t_current <= solver_temperature_goal + solver_temperature_goal_tol {
        return SolverResult { d_p: best_d_p, t_a_max: best_t, t_at_max: best_t_time,
                              iter_count: total_iter_count, status: STATUS_SUCCESS };
    }

    // --- Linear step search ---
    let mut d_p_previous = d_p_1;
    let mut t_previous = t_current;
    let mut t_previous_time = t_current_time;
    let mut d_p_current = d_p_1;

    loop {
        if total_iter_count >= solver_max_iter {
            return SolverResult { d_p: best_d_p, t_a_max: best_t, t_at_max: best_t_time,
                                  iter_count: total_iter_count, status: STATUS_MAX_ITERATIONS_REACHED };
        }
        d_p_current = d_p_previous + d_p_i;
        if d_p_current >= d_p_2 {
            d_p_current = d_p_2;
        }
        if d_p_current == d_p_previous {
            break; // stuck at d_p_2
        }

        let (t_new, t_new_time) = temperature_max(
            fire_time, fire_temperature, beam_rho, beam_cross_section_area,
            protection_k, protection_rho, protection_c, d_p_current, protection_protected_perimeter,
        );
        t_current = t_new;
        t_current_time = t_new_time;
        total_iter_count += 1;

        // Monotonicity check
        if t_current > t_previous {
            return SolverResult { d_p: d_p_previous, t_a_max: t_previous, t_at_max: t_previous_time,
                                  iter_count: total_iter_count, status: STATUS_MONOTONICITY_FAILED };
        }

        let current_diff = (t_current - solver_temperature_goal).abs();
        if current_diff < min_abs_diff_found {
            min_abs_diff_found = current_diff;
            best_d_p = d_p_current;
            best_t = t_current;
            best_t_time = t_current_time;
        }

        if t_current <= solver_temperature_goal + solver_temperature_goal_tol {
            d_p_low = d_p_previous;
            d_p_high = d_p_current;
            break;
        }

        d_p_previous = d_p_current;
        t_previous = t_current;
        t_previous_time = t_current_time;
    }

    // --- Post linear search: did we hit d_p_2 without bracketing? ---
    if d_p_current == d_p_2 && d_p_low < 0.0 {
        if t_current > solver_temperature_goal + solver_temperature_goal_tol {
            return SolverResult { d_p: best_d_p, t_a_max: best_t, t_at_max: best_t_time,
                                  iter_count: total_iter_count, status: STATUS_OUT_OF_UPPER_BOUND };
        } else {
            d_p_low = d_p_previous;
            d_p_high = d_p_current;
        }
    }

    // --- Binary search refinement ---
    if d_p_low >= 0.0 && d_p_low < d_p_high {
        for _ in total_iter_count..solver_max_iter {
            let d_p_mid = d_p_low + 0.5 * (d_p_high - d_p_low);

            if (d_p_high - d_p_low) < 1e-12 {
                let (t_mid, t_mid_time) = temperature_max(
                    fire_time, fire_temperature, beam_rho, beam_cross_section_area,
                    protection_k, protection_rho, protection_c, d_p_mid, protection_protected_perimeter,
                );
                total_iter_count += 1;
                let mid_diff = (t_mid - solver_temperature_goal).abs();
                if mid_diff < min_abs_diff_found {
                    return SolverResult { d_p: d_p_mid, t_a_max: t_mid, t_at_max: t_mid_time,
                                          iter_count: total_iter_count, status: STATUS_SUCCESS };
                } else {
                    return SolverResult { d_p: best_d_p, t_a_max: best_t, t_at_max: best_t_time,
                                          iter_count: total_iter_count, status: STATUS_SUCCESS };
                }
            }

            let (t_new, t_new_time) = temperature_max(
                fire_time, fire_temperature, beam_rho, beam_cross_section_area,
                protection_k, protection_rho, protection_c, d_p_mid, protection_protected_perimeter,
            );
            t_current = t_new;
            t_current_time = t_new_time;
            total_iter_count += 1;

            let current_diff = (t_current - solver_temperature_goal).abs();
            if current_diff < min_abs_diff_found {
                min_abs_diff_found = current_diff;
                best_d_p = d_p_mid;
                best_t = t_current;
                best_t_time = t_current_time;
            }

            if t_current <= solver_temperature_goal + solver_temperature_goal_tol
                && t_current >= solver_temperature_goal - solver_temperature_goal_tol
            {
                return SolverResult { d_p: d_p_mid, t_a_max: t_current, t_at_max: t_current_time,
                                      iter_count: total_iter_count, status: STATUS_SUCCESS };
            }

            if t_current > solver_temperature_goal {
                d_p_low = d_p_mid; // too hot, need thicker
            } else {
                d_p_high = d_p_mid; // too cool, need thinner
            }

            if total_iter_count >= solver_max_iter {
                return SolverResult { d_p: best_d_p, t_a_max: best_t, t_at_max: best_t_time,
                                      iter_count: total_iter_count, status: STATUS_MAX_ITERATIONS_REACHED };
            }
        }
        return SolverResult { d_p: best_d_p, t_a_max: best_t, t_at_max: best_t_time,
                              iter_count: total_iter_count, status: STATUS_MAX_ITERATIONS_REACHED };
    }

    // --- Fallback ---
    let mut final_status = STATUS_MAX_ITERATIONS_REACHED;
    if d_p_current == d_p_2 && best_t > solver_temperature_goal + solver_temperature_goal_tol {
        final_status = STATUS_OUT_OF_UPPER_BOUND;
    } else if (best_t - solver_temperature_goal).abs() <= solver_temperature_goal_tol {
        final_status = STATUS_SUCCESS;
    }
    SolverResult { d_p: best_d_p, t_a_max: best_t, t_at_max: best_t_time,
                   iter_count: total_iter_count, status: final_status }
}
