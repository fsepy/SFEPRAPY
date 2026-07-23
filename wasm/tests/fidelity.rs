//! Fidelity tests: the Rust port must reproduce the Python reference results.
//! These are the same oracle cases used in `test/test_mcs0.py`.

use sfeprapy_wasm::solver::protection_thickness_2;
use sfeprapy_wasm::teq::{teq_main, TeqMainArgs};

/// Travelling-fire gas curve from the fsetools reference test (SI units, returns K).
/// This is the inline reproduction of `_trav_fire` from the Python test.
fn trav_fire(t: &[f64]) -> Vec<f64> {
    let t_0: f64 = 273.15 - 273.15; // C
    let q_f_d: f64 = 600e6 / 1e6;  // MJ/m2
    let hrrpua: f64 = 0.25e6 / 1e6; // MW/m2
    let (l, w) = (100.0_f64, 16.0_f64);
    let s: f64 = 0.012;
    let e_h: f64 = 3.0;
    let e_l: f64 = 50.0;
    let t_max: f64 = 1050.0 + 273.15 - 273.15; // C

    let time_step = t[1] - t[0];
    let t_burn = (q_f_d / hrrpua).max(900.0);
    let t_decay = t_burn.max(l / s);
    let t_lim = t_burn.min(l / s);
    let t_decay_ = (t_decay / time_step).round() * time_step;
    let mut t_lim_ = (t_lim / time_step).round() * time_step;
    if t_decay_ == t_lim_ { t_lim_ -= time_step; }

    let q_peak_scalar = (hrrpua * w * s * t_burn).min(hrrpua * w * l);

    let mut out = vec![0.0_f64; t.len()];
    for i in 0..t.len() {
        let ti = t[i];
        let q_growth = (hrrpua * w * s * ti) * if ti < t_lim_ { 1.0 } else { 0.0 };
        let q_peak = q_peak_scalar * if ti >= t_lim_ && ti <= t_decay_ { 1.0 } else { 0.0 };
        let mut q_decay = (q_peak_scalar - (ti - t_decay_) * w * s * hrrpua) * if ti > t_decay_ { 1.0 } else { 0.0 };
        if q_decay < 0.0 { q_decay = 0.0; }
        let q = (q_growth + q_peak + q_decay) * 1000.0;

        let mut l_fire_front = s * ti;
        if l_fire_front < 0.0 { l_fire_front = 0.0; }
        if l_fire_front > l { l_fire_front = l; }
        let mut l_fire_end = s * (ti - t_lim);
        if l_fire_end < 0.0 { l_fire_end = 0.0; }
        if l_fire_end > l { l_fire_end = l; }
        let l_fire_median = (l_fire_front + l_fire_end) / 2.0;

        let mut r = (e_l - l_fire_median).abs();
        if r == 0.0 { r = 0.001; }
        let ratio = r / e_h;
        let mut val = if ratio > 0.18 {
            (5.38 * (q / r).powf(2.0 / 3.0) / e_h) + t_0
        } else {
            (16.9 * q.powf(2.0 / 3.0) / e_h.powf(5.0 / 3.0)) + t_0
        };
        if val >= t_max { val = t_max; }
        out[i] = val + 273.15; // C -> K
    }
    out
}

#[test]
fn test_protection_thickness_fidelity() {
    // The fsetools oracle: travelling fire, goal 893.15 K -> d_p ≈ 0.01555 m.
    let t: Vec<f64> = (0..12600).map(|i| i as f64).collect();
    let fire_temp = trav_fire(&t);

    let r = protection_thickness_2(
        &t, &fire_temp, 7850.0, 0.017, 0.2, 800.0, 1700.0, 2.14,
        873.15 + 20.0, 0.1, 100, 0.0001, 0.0801, 0.00375,
    );
    println!("d_p = {:.5} m, T = {:.2} K, status = {}", r.d_p, r.t_a_max, r.status);
    assert!((r.t_a_max - (873.15 + 20.0)).abs() <= 0.1, "T mismatch: {}", r.t_a_max);
    assert!((r.d_p - 0.01555).abs() <= 1e-5, "d_p mismatch: {}", r.d_p);
}

#[test]
fn test_teq_scalar() {
    // The scalar case from test_mcs0.py -> teq ≈ 1964.4 s.
    let args = TeqMainArgs {
        fire_time_step: 1.0, fire_time_duration: 5.0 * 60.0 * 60.0,
        beam_cross_section_area: 0.017, beam_rho: 7850.0,
        beam_position_vertical: 2.5, beam_position_horizontal: 18.0,
        fire_combustion_efficiency: 0.8, fire_hrr_density: 0.25,
        fire_load_density: 420.0, fire_mode: 0, fire_nft_limit: 1050.0,
        fire_spread_speed: 0.01, fire_tlim: 0.333,
        protection_c: 1700.0, protection_k: 0.2,
        protection_protected_perimeter: 2.14, protection_rho: 800.0,
        room_breadth: 16.0, room_depth: 31.25, room_height: 3.0,
        room_wall_thermal_inertia: 720.0,
        solver_temperature_goal: 620.0 + 273.15, solver_tol: 0.01,
        window_height: 2.0, window_width: 57.6,
        timber_burning_rate: 0.0,
        timber_fire_load_max: None,
        timber_solver_ilim: 20.0, timber_solver_tol: 1.0,
    };
    let r = teq_main(&args);
    let teq = r.solver_time_equivalence_solved;
    println!("teq = {:.3} s ({:.1} min)", teq, teq / 60.0);
    assert!((teq - 1964.0).abs() < 5.0, "teq mismatch: {}", teq);
}

#[test]
fn test_teq_timber() {
    // Timber case: finite, positive, in a sane range.
    let mut args = TeqMainArgs {
        fire_time_step: 10.0, fire_time_duration: 18000.0,
        beam_cross_section_area: 0.017, beam_rho: 7850.0,
        beam_position_vertical: 3.1, beam_position_horizontal: 18.0,
        fire_combustion_efficiency: 0.8, fire_hrr_density: 0.25,
        fire_load_density: 420.0, fire_mode: 0, fire_nft_limit: 1323.15,
        fire_spread_speed: 0.01, fire_tlim: 0.333,
        protection_c: 1700.0, protection_k: 0.2,
        protection_protected_perimeter: 2.14, protection_rho: 800.0,
        room_breadth: 16.0, room_depth: 31.25, room_height: 3.1,
        room_wall_thermal_inertia: 720.0,
        solver_temperature_goal: 823.15, solver_tol: 1.0,
        window_height: 2.8, window_width: 14.4,
        timber_burning_rate: 30.8,
        timber_fire_load_max: None,
        timber_solver_ilim: 20.0, timber_solver_tol: 1.0,
    };
    let r = teq_main(&args);
    let teq = r.solver_time_equivalence_solved;
    println!("timber teq = {:.1} s, fire_type = {}, timber_iters = {}",
             teq, r.fire_type, r.timber_solver_iter_count);
    assert!(teq.is_finite(), "teq should be finite, got {}", teq);
    assert!(teq > 0.0);
    assert!(teq < 18000.0);

    // Also confirm no-timber on the same config is faster (smaller teq).
    args.timber_burning_rate = 0.0;
    let r2 = teq_main(&args);
    assert!(r2.solver_time_equivalence_solved < teq, "timber should increase teq");
}
