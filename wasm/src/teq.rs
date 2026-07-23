//! The time-equivalence pipeline. Port of `sfeprapy.calcs`.
//!
//! Combines fire-type selection, fire-curve evaluation, the protection-thickness solver,
//! and the ISO 834 equivalence into the single `teq_main` entry point.

use crate::fire::{parametric_fire_temperature, travelling_fire_temperature};
use crate::fire::py_round;
use crate::solver::{protection_thickness_2, SolverResult};
use crate::steel::temperature as steel_temperature;

// Eurocode parametric-fire validity limits (BS EN 1991-1-2 Annex A).
const OPENING_FACTOR_LBOUND_EC: f64 = 0.01;
const OPENING_FACTOR_UBOUND_EC: f64 = 0.20;
const FIRE_LOAD_DENSITY_TOTAL_LBOUND_EC: f64 = 50.0;
const FIRE_LOAD_DENSITY_TOTAL_UBOUND_EC: f64 = 1000.0;
const MIN_BURNOUT_TIME: f64 = 900.0;

/// Derived compartment quantities (matches `_compartment_params`).
struct Compartment {
    window_area: f64,
    room_floor_area: f64,
    room_total_area: f64,
    fire_load_density_deducted: f64,
    fire_load_density_total: f64,
    opening_factor: f64,
}

fn compartment_params(
    window_height: f64, window_width: f64,
    room_breadth: f64, room_depth: f64, room_height: f64,
    fire_load_density: f64, fire_combustion_efficiency: f64,
) -> Compartment {
    let room_floor_area = room_breadth * room_depth;
    let room_total_area = 2.0 * room_floor_area + (room_breadth + room_depth) * 2.0 * room_height;
    let window_area = window_height * window_width;
    let fire_load_density_deducted = fire_load_density * fire_combustion_efficiency;
    let fire_load_density_total = fire_load_density_deducted * room_floor_area / room_total_area;
    let opening_factor = window_area * window_height.sqrt() / room_total_area;
    Compartment {
        window_area, room_floor_area, room_total_area,
        fire_load_density_deducted, fire_load_density_total, opening_factor,
    }
}

/// Decide fire type: 0 = parametric, 1 = travelling. Port of `decide_fire`.
fn decide_fire(
    window_height: f64, window_width: f64,
    room_breadth: f64, room_depth: f64, room_height: f64,
    fire_mode: i32, fire_load_density: f64,
    fire_combustion_efficiency: f64, fire_hrr_density: f64, fire_spread_speed: f64,
) -> i32 {
    let p = compartment_params(window_height, window_width, room_breadth, room_depth, room_height,
                                fire_load_density, fire_combustion_efficiency);
    let fire_spread_entire_room_time = room_depth / fire_spread_speed;
    let burn_out_time = (p.fire_load_density_deducted / fire_hrr_density).max(MIN_BURNOUT_TIME);

    if fire_mode == 0 || fire_mode == 1 {
        fire_mode
    } else if fire_mode == 3 {
        if fire_spread_entire_room_time < burn_out_time
            && OPENING_FACTOR_LBOUND_EC < p.opening_factor
            && p.opening_factor <= OPENING_FACTOR_UBOUND_EC
            && FIRE_LOAD_DENSITY_TOTAL_LBOUND_EC <= p.fire_load_density_total
            && p.fire_load_density_total <= FIRE_LOAD_DENSITY_TOTAL_UBOUND_EC
        {
            0 // parametric
        } else {
            1 // travelling
        }
    } else {
        panic!("Unknown fire mode {}", fire_mode);
    }
}

/// Evaluate the fire temperature curve. Returns `(fire_temperature, t1, t2, t3)`.
#[allow(clippy::too_many_arguments)]
fn evaluate_fire_temperature(
    window_height: f64, window_width: f64,
    room_breadth: f64, room_depth: f64, room_height: f64,
    room_wall_thermal_inertia: f64, fire_tlim: f64, fire_type: i32,
    fire_time: &[f64], fire_nft_limit: f64, fire_load_density: f64,
    fire_combustion_efficiency: f64, fire_hrr_density: f64, fire_spread_speed: f64,
    beam_position_vertical: f64, beam_position_horizontal: f64,
) -> (Vec<f64>, f64, f64, f64) {
    let p = compartment_params(window_height, window_width, room_breadth, room_depth, room_height,
                                fire_load_density, fire_combustion_efficiency);
    let nan = f64::NAN;

    if fire_type == 0 {
        let tg = parametric_fire_temperature(
            fire_time, p.room_total_area, p.room_floor_area, p.window_area, window_height,
            p.fire_load_density_deducted * 1e6, room_wall_thermal_inertia.powi(2),
            1.0, 1.0, fire_tlim, 20.0 + 273.15,
        );
        (tg, nan, nan, nan)
    } else if fire_type == 1 {
        let tg_c = travelling_fire_temperature(
            fire_time, p.fire_load_density_deducted, fire_hrr_density,
            room_depth, room_breadth, fire_spread_speed,
            beam_position_vertical, beam_position_horizontal,
            fire_nft_limit - 273.15,
        );
        let tg: Vec<f64> = tg_c.iter().map(|&v| v + 273.15).collect();
        let t1 = (room_depth / fire_spread_speed).min(p.fire_load_density_deducted / fire_hrr_density);
        let t2 = (room_depth / fire_spread_speed).max(p.fire_load_density_deducted / fire_hrr_density);
        (tg, t1, t2, t1 + t2)
    } else {
        (vec![nan; fire_time.len()], nan, nan, nan)
    }
}

/// Solve equivalent time exposure to ISO 834. Port of `solve_time_equivalence_iso834`.
#[allow(clippy::too_many_arguments)]
fn solve_time_equivalence_iso834(
    fire_time: &[f64], beam_cross_section_area: f64, beam_rho: f64,
    protection_k: f64, protection_rho: f64, protection_c: f64,
    protection_protected_perimeter: f64,
    solver_temperature_goal: f64, solver_protection_thickness: f64,
) -> f64 {
    let solver_d_p = solver_protection_thickness;
    // NaN first, then non-finite (matches the Python fix).
    if solver_d_p.is_nan() {
        return f64::NAN;
    }
    if !solver_d_p.is_finite() {
        return solver_d_p; // +/- inf
    }

    // ISO 834 curve [K]
    let iso834: Vec<f64> = fire_time.iter()
        .map(|&t| 345.0 * ((t / 60.0) * 8.0 + 1.0).log10() + 20.0 + 273.15)
        .collect();
    let steel_temp = steel_temperature(
        fire_time, &iso834, beam_rho, beam_cross_section_area,
        protection_k, protection_rho, protection_c, solver_d_p, protection_protected_perimeter,
    );

    let min_st = steel_temp.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_st = steel_temp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if solver_temperature_goal < min_st {
        f64::NAN
    } else if solver_temperature_goal > max_st {
        f64::INFINITY
    } else {
        // np.interp linear scan: xp = steel_temp, fp = fire_time.
        np_interp(solver_temperature_goal, &steel_temp, fire_time)
    }
}

/// Replicate np.interp(x, xp, fp): linear scan assuming xp is increasing, clip to endpoints.
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] { return fp[0]; }
    if x >= xp[n - 1] { return fp[n - 1]; }
    for i in 0..n - 1 {
        if xp[i] <= x && x <= xp[i + 1] {
            let dx = xp[i + 1] - xp[i];
            if dx == 0.0 { return fp[i]; }
            return fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / dx;
        }
    }
    fp[n - 1] // unreachable given the bounds checks
}

/// Solve protection thickness. Port of `solve_protection_thickness`.
/// Returns `(steel_temp_solved, time_critical_temp, protection_thickness, iter_count)`.
#[allow(clippy::too_many_arguments)]
fn solve_protection_thickness(
    fire_time: &[f64], fire_temperature: &[f64],
    beam_cross_section_area: f64, beam_rho: f64,
    protection_k: f64, protection_rho: f64, protection_c: f64,
    protection_protected_perimeter: f64,
    solver_temperature_goal: f64, solver_tol: f64,
) -> (f64, f64, f64, f64) {
    let SolverResult { d_p: solver_d_p, t_a_max: solver_t_max_a, t_at_max: solver_t,
                       iter_count: solver_iter_count, status: solver_status } = protection_thickness_2(
        fire_time, fire_temperature, beam_rho, beam_cross_section_area,
        protection_k, protection_rho, protection_c, protection_protected_perimeter,
        solver_temperature_goal, solver_tol, 100, 0.0001, 0.0801, 0.00375,
    );
    let iter = solver_iter_count as f64;
    match solver_status {
        crate::solver::STATUS_OUT_OF_LOWER_BOUND => (-f64::INFINITY, solver_t, solver_d_p, iter),
        crate::solver::STATUS_OUT_OF_UPPER_BOUND => (f64::INFINITY, solver_t, solver_d_p, iter),
        crate::solver::STATUS_MAX_ITERATIONS_REACHED => (f64::NAN, f64::NAN, f64::NAN, iter),
        // status 0 (success) or 4 (monotonicity): return best available
        _ => (solver_t_max_a, solver_t, solver_d_p, iter),
    }
}

/// Inputs for a single-pass time-equivalence solve (the parts that don't change across
/// timber iterations).
struct OnceInputs<'a> {
    fire_time: &'a [f64],
    window_height: f64, window_width: f64,
    room_breadth: f64, room_depth: f64, room_height: f64, room_wall_thermal_inertia: f64,
    fire_tlim: f64, fire_mode: i32, fire_nft_limit: f64,
    fire_combustion_efficiency: f64, fire_hrr_density: f64, fire_spread_speed: f64,
    beam_position_vertical: f64, beam_position_horizontal: f64,
    beam_cross_section_area: f64, beam_rho: f64,
    protection_k: f64, protection_rho: f64, protection_c: f64, protection_protected_perimeter: f64,
    solver_temperature_goal: f64, solver_tol: f64,
}

/// Result fields produced by a single pass.
struct OnceResult {
    fire_type: i32,
    t1: f64, t2: f64, t3: f64,
    solver_steel_temperature_solved: f64,
    solver_time_critical_temp_solved: f64,
    solver_protection_thickness: f64,
    solver_iter_count: f64,
    solver_time_equivalence_solved: f64,
}

fn solve_teq_once(fire_load_density: f64, inp: &OnceInputs) -> OnceResult {
    let fire_type = decide_fire(
        inp.window_height, inp.window_width, inp.room_breadth, inp.room_depth, inp.room_height,
        inp.fire_mode, fire_load_density, inp.fire_combustion_efficiency,
        inp.fire_hrr_density, inp.fire_spread_speed,
    );
    let (fire_temp, t1, t2, t3) = evaluate_fire_temperature(
        inp.window_height, inp.window_width, inp.room_breadth, inp.room_depth, inp.room_height,
        inp.room_wall_thermal_inertia, inp.fire_tlim, fire_type, inp.fire_time, inp.fire_nft_limit,
        fire_load_density, inp.fire_combustion_efficiency, inp.fire_hrr_density, inp.fire_spread_speed,
        inp.beam_position_vertical, inp.beam_position_horizontal,
    );
    let (solver_steel_temperature_solved, solver_time_critical_temp_solved,
         solver_protection_thickness, solver_iter_count) = solve_protection_thickness(
        inp.fire_time, &fire_temp, inp.beam_cross_section_area, inp.beam_rho,
        inp.protection_k, inp.protection_rho, inp.protection_c, inp.protection_protected_perimeter,
        inp.solver_temperature_goal, inp.solver_tol,
    );
    let solver_time_equivalence_solved = solve_time_equivalence_iso834(
        inp.fire_time, inp.beam_cross_section_area, inp.beam_rho,
        inp.protection_k, inp.protection_rho, inp.protection_c, inp.protection_protected_perimeter,
        inp.solver_temperature_goal, solver_protection_thickness,
    );
    OnceResult {
        fire_type, t1, t2, t3,
        solver_steel_temperature_solved, solver_time_critical_temp_solved,
        solver_protection_thickness, solver_iter_count, solver_time_equivalence_solved,
    }
}

/// Full TeqResult, matching the Python `TeqResult` NamedTuple.
#[derive(Debug, Clone, Default)]
pub struct TeqResult {
    pub fire_type: i64,
    pub t1: f64,
    pub t2: f64,
    pub t3: f64,
    pub solver_steel_temperature_solved: f64,
    pub solver_time_critical_temp_solved: f64,
    pub solver_protection_thickness: f64,
    pub solver_iter_count: f64,
    pub solver_time_equivalence_solved: f64,
    pub timber_exposed_duration: f64,
    pub timber_solver_iter_count: f64,
    pub timber_fire_load: f64,
}

/// The main entry point. Port of `sfeqprapy.teq_main`.
#[allow(clippy::too_many_arguments)]
pub fn teq_main(args: &TeqMainArgs) -> TeqResult {
    // Geometry normalization.
    let (mut room_depth, mut room_breadth) = (args.room_depth, args.room_breadth);
    if room_depth < room_breadth {
        room_depth += room_breadth;
        room_breadth = room_depth - room_breadth;
        room_depth -= room_breadth;
    }
    let window_height = args.window_height.min(args.room_height);

    // Time grid: linspace(0, duration, round(duration/step)+1).
    let n_steps = py_round(args.fire_time_duration / args.fire_time_step) as i64;
    let fire_time: Vec<f64> = (0..=n_steps)
        .map(|i| i as f64 * args.fire_time_duration / n_steps as f64)
        .collect();

    let fire_load_density_base = args.fire_load_density;
    let has_timber = args.timber_burning_rate > 0.0;

    let inp = OnceInputs {
        fire_time: &fire_time,
        window_height, window_width: args.window_width,
        room_breadth, room_depth, room_height: args.room_height,
        room_wall_thermal_inertia: args.room_wall_thermal_inertia,
        fire_tlim: args.fire_tlim, fire_mode: args.fire_mode, fire_nft_limit: args.fire_nft_limit,
        fire_combustion_efficiency: args.fire_combustion_efficiency,
        fire_hrr_density: args.fire_hrr_density, fire_spread_speed: args.fire_spread_speed,
        beam_position_vertical: args.beam_position_vertical,
        beam_position_horizontal: args.beam_position_horizontal,
        beam_cross_section_area: args.beam_cross_section_area, beam_rho: args.beam_rho,
        protection_k: args.protection_k, protection_rho: args.protection_rho,
        protection_c: args.protection_c, protection_protected_perimeter: args.protection_protected_perimeter,
        solver_temperature_goal: args.solver_temperature_goal, solver_tol: args.solver_tol,
    };

    if !has_timber {
        let r = solve_teq_once(fire_load_density_base, &inp);
        TeqResult {
            fire_type: r.fire_type as i64, t1: r.t1, t2: r.t2, t3: r.t3,
            solver_steel_temperature_solved: r.solver_steel_temperature_solved,
            solver_time_critical_temp_solved: r.solver_time_critical_temp_solved,
            solver_protection_thickness: r.solver_protection_thickness,
            solver_iter_count: r.solver_iter_count,
            solver_time_equivalence_solved: r.solver_time_equivalence_solved,
            timber_exposed_duration: 0.0,
            timber_solver_iter_count: 0.0,
            timber_fire_load: f64::NAN,
        }
    } else {
        let room_floor_area = room_breadth * room_depth;
        let mut timber_solver_iter_count = -1.0_f64;
        let mut timber_exposed_duration = 0.0_f64;

        let (r, timber_exposed_duration_out, timber_fire_load_out, hit_iter_cap) = loop {
            timber_solver_iter_count += 1.0;
            let mut timber_fire_load = args.timber_burning_rate * timber_exposed_duration;
            if let Some(max) = args.timber_fire_load_max {
                timber_fire_load = timber_fire_load.min(max);
            }
            let fire_load_density = fire_load_density_base + timber_fire_load / room_floor_area;
            let r = solve_teq_once(fire_load_density, &inp);

            if timber_solver_iter_count >= args.timber_solver_ilim {
                break (r, f64::NAN, timber_fire_load, true);
            } else {
                let pt = r.solver_protection_thickness;
                if !(-f64::INFINITY < pt && pt < f64::INFINITY) {
                    break (r, pt, timber_fire_load, false);
                } else if (timber_exposed_duration - r.solver_time_equivalence_solved).abs() <= args.timber_solver_tol {
                    break (r, timber_exposed_duration, timber_fire_load, false);
                } else {
                    timber_exposed_duration = r.solver_time_equivalence_solved;
                }
            }
        };

        let (solver_te, solver_tct, solver_pt, solver_ic, solver_teq) = if hit_iter_cap {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        } else {
            (r.solver_steel_temperature_solved, r.solver_time_critical_temp_solved,
             r.solver_protection_thickness, r.solver_iter_count, r.solver_time_equivalence_solved)
        };

        TeqResult {
            fire_type: r.fire_type as i64, t1: r.t1, t2: r.t2, t3: r.t3,
            solver_steel_temperature_solved: solver_te,
            solver_time_critical_temp_solved: solver_tct,
            solver_protection_thickness: solver_pt,
            solver_iter_count: solver_ic,
            solver_time_equivalence_solved: solver_teq,
            timber_exposed_duration: timber_exposed_duration_out,
            timber_solver_iter_count,
            timber_fire_load: timber_fire_load_out,
        }
    }
}

/// Input parameters for `teq_main` (mirrors the Python signature).
#[derive(Debug, Clone)]
pub struct TeqMainArgs {
    pub beam_cross_section_area: f64,
    pub beam_position_vertical: f64,
    pub beam_position_horizontal: f64,
    pub beam_rho: f64,
    pub fire_time_duration: f64,
    pub fire_time_step: f64,
    pub fire_combustion_efficiency: f64,
    pub fire_hrr_density: f64,
    pub fire_load_density: f64,
    pub fire_mode: i32,
    pub fire_nft_limit: f64,
    pub fire_spread_speed: f64,
    pub fire_tlim: f64,
    pub protection_c: f64,
    pub protection_k: f64,
    pub protection_protected_perimeter: f64,
    pub protection_rho: f64,
    pub room_breadth: f64,
    pub room_depth: f64,
    pub room_height: f64,
    pub room_wall_thermal_inertia: f64,
    pub solver_temperature_goal: f64,
    pub solver_tol: f64,
    pub window_height: f64,
    pub window_width: f64,
    pub timber_burning_rate: f64,
    pub timber_fire_load_max: Option<f64>,
    pub timber_solver_tol: f64,
    pub timber_solver_ilim: f64,
}

impl Default for TeqMainArgs {
    fn default() -> Self {
        TeqMainArgs {
            beam_cross_section_area: 0.0, beam_position_vertical: 0.0, beam_position_horizontal: 0.0,
            beam_rho: 0.0, fire_time_duration: 0.0, fire_time_step: 0.0,
            fire_combustion_efficiency: 0.0, fire_hrr_density: 0.0, fire_load_density: 0.0,
            fire_mode: 0, fire_nft_limit: 0.0, fire_spread_speed: 0.0, fire_tlim: 0.0,
            protection_c: 0.0, protection_k: 0.0, protection_protected_perimeter: 0.0,
            protection_rho: 0.0, room_breadth: 0.0, room_depth: 0.0, room_height: 0.0,
            room_wall_thermal_inertia: 0.0, solver_temperature_goal: 0.0, solver_tol: 0.0,
            window_height: 0.0, window_width: 0.0, timber_burning_rate: 0.0,
            timber_fire_load_max: None, timber_solver_tol: 0.0, timber_solver_ilim: 0.0,
        }
    }
}
