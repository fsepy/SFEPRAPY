//! sfeprapy-wasm: Rust → WASM port of the sfeprapy equivalent-time-exposure calculation.
//!
//! The Python (`src/sfeprapy/`) is the source of truth; this crate is a faithful port
//! verified against the same fidelity oracle (d_p = 0.01555 m, teq ≈ 1964 s).

pub mod fire;
pub mod solver;
pub mod steel;
pub mod teq;

pub use teq::{teq_main, TeqMainArgs, TeqResult};

use wasm_bindgen::prelude::*;

/// Call `teq_main` from JS. Pass a JS object of the parameters; get a JS object back.
///
/// Field names match the Python `teq_main` kwargs. `timber_fire_load_max` may be null/omitted
/// (no cap). Returns an object with the `TeqResult` fields.
#[wasm_bindgen]
pub fn teq_main_js(kwargs: JsValue) -> Result<JsValue, JsValue> {
    let a: TeqMainArgsJs = serde_wasm_bindgen::from_value(kwargs)
        .map_err(|e| JsValue::from_str(&format!("deserialize error: {}", e)))?;

    let args = TeqMainArgs {
        beam_cross_section_area: a.beam_cross_section_area,
        beam_position_vertical: a.beam_position_vertical,
        beam_position_horizontal: a.beam_position_horizontal,
        beam_rho: a.beam_rho,
        fire_time_duration: a.fire_time_duration,
        fire_time_step: a.fire_time_step,
        fire_combustion_efficiency: a.fire_combustion_efficiency,
        fire_hrr_density: a.fire_hrr_density,
        fire_load_density: a.fire_load_density,
        fire_mode: a.fire_mode,
        fire_nft_limit: a.fire_nft_limit,
        fire_spread_speed: a.fire_spread_speed,
        fire_tlim: a.fire_tlim,
        protection_c: a.protection_c,
        protection_k: a.protection_k,
        protection_protected_perimeter: a.protection_protected_perimeter,
        protection_rho: a.protection_rho,
        room_breadth: a.room_breadth,
        room_depth: a.room_depth,
        room_height: a.room_height,
        room_wall_thermal_inertia: a.room_wall_thermal_inertia,
        solver_temperature_goal: a.solver_temperature_goal,
        solver_tol: a.solver_tol,
        window_height: a.window_height,
        window_width: a.window_width,
        timber_burning_rate: a.timber_burning_rate.unwrap_or(0.0),
        timber_fire_load_max: a.timber_fire_load_max,
        timber_solver_tol: a.timber_solver_tol.unwrap_or(1.0),
        timber_solver_ilim: a.timber_solver_ilim.unwrap_or(20.0),
    };

    let r = teq_main(&args);

    let out = TeqResultJs {
        fire_type: r.fire_type,
        t1: r.t1, t2: r.t2, t3: r.t3,
        solver_steel_temperature_solved: r.solver_steel_temperature_solved,
        solver_time_critical_temp_solved: r.solver_time_critical_temp_solved,
        solver_protection_thickness: r.solver_protection_thickness,
        solver_iter_count: r.solver_iter_count,
        solver_time_equivalence_solved: r.solver_time_equivalence_solved,
        timber_exposed_duration: r.timber_exposed_duration,
        timber_solver_iter_count: r.timber_solver_iter_count,
        timber_fire_load: r.timber_fire_load,
    };

    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("serialize error: {}", e)))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct TeqMainArgsJs {
    beam_cross_section_area: f64,
    beam_position_vertical: f64,
    beam_position_horizontal: f64,
    beam_rho: f64,
    fire_time_duration: f64,
    fire_time_step: f64,
    fire_combustion_efficiency: f64,
    fire_hrr_density: f64,
    fire_load_density: f64,
    fire_mode: i32,
    fire_nft_limit: f64,
    fire_spread_speed: f64,
    fire_tlim: f64,
    protection_c: f64,
    protection_k: f64,
    protection_protected_perimeter: f64,
    protection_rho: f64,
    room_breadth: f64,
    room_depth: f64,
    room_height: f64,
    room_wall_thermal_inertia: f64,
    solver_temperature_goal: f64,
    solver_tol: f64,
    window_height: f64,
    window_width: f64,
    timber_burning_rate: Option<f64>,
    timber_fire_load_max: Option<f64>,
    timber_solver_tol: Option<f64>,
    timber_solver_ilim: Option<f64>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct TeqResultJs {
    fire_type: i64,
    t1: f64, t2: f64, t3: f64,
    solver_steel_temperature_solved: f64,
    solver_time_critical_temp_solved: f64,
    solver_protection_thickness: f64,
    solver_iter_count: f64,
    solver_time_equivalence_solved: f64,
    timber_exposed_duration: f64,
    timber_solver_iter_count: f64,
    timber_fire_load: f64,
}
