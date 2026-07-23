//! Design fire temperature curves. Port of `parametric_fire_temperature` and
//! `travelling_fire_temperature` from `sfeprapy._fsetools`.

/// Python 3 `round(x, 0)` uses banker's (half-to-even) rounding. Rust's `f64::round`
/// rounds half *away from zero*. They differ only at exact `.5` boundaries, but to
/// match the Python bit-for-bit we implement half-to-even here.
pub(crate) fn py_round(x: f64) -> f64 {
    let r = x.round();
    // If exactly at .5, pick the even neighbour.
    let frac = (x - r).abs();
    if frac == 0.5 {
        let floor = x.floor();
        let ceil = x.ceil();
        if floor as i64 % 2 == 0 { floor } else { ceil }
    } else {
        r
    }
}

// ---------------------------------------------------------------------------
// BS EN 1991-1-2 Annex A parametric fire
// ---------------------------------------------------------------------------

fn eq_3_12_t_g(t_star: f64, t_0: f64) -> f64 {
    1325.0 * (1.0 - 0.324 * (-0.2 * t_star).exp() - 0.204 * (-1.7 * t_star).exp()
        - 0.472 * (-19.0 * t_star).exp()) + t_0
}

fn eq_3_16_t_g(t_star_max: f64, t_max: f64, t_star: f64) -> f64 {
    if t_star_max <= 0.5 {
        t_max - 625.0 * (t_star - t_star_max)
    } else if t_star_max < 2.0 {
        t_max - 250.0 * (3.0 - t_star_max) * (t_star - t_star_max)
    } else {
        t_max - 250.0 * (t_star - t_star_max)
    }
}

fn eq_3_22_t_g(t_star_max: f64, t_max: f64, t_star: f64, gamma: f64, t_lim: f64) -> f64 {
    if t_star_max <= 0.5 {
        t_max - 625.0 * (t_star - gamma * t_lim)
    } else if t_star_max < 2.0 {
        t_max - 250.0 * (3.0 - t_star_max) * (t_star - gamma * t_lim)
    } else {
        t_max - 250.0 * (t_star - gamma * t_lim)
    }
}

/// EC parametric fire temperature [K]. Port of `parametric_fire_temperature`.
#[allow(clippy::too_many_arguments)]
pub fn parametric_fire_temperature(
    t: &[f64],          // [s]
    a_t: f64, a_f: f64, a_v: f64, h_eq: f64, q_fd: f64,
    lbd: f64, rho: f64, c: f64, t_lim: f64, t_0: f64,
) -> Vec<f64> {
    // Unit conversion SI -> local (matches Python exactly).
    let q_fd = q_fd / 1e6;      // J/m2 -> MJ/m2
    let t_lim = t_lim / 3600.0; // s -> hr
    let t_0 = t_0 - 273.15;     // K -> C

    let b = (lbd * rho * c).sqrt();
    let o = a_v * h_eq.powf(0.5) / a_t;
    let q_td = q_fd * a_f / a_t;
    let gamma = ((o / 0.04) / (b / 1160.0)).powi(2);
    let t_max = 0.0002 * q_td / o;

    let n = t.len();
    let mut t_g = vec![0.0_f64; n];

    if t_max >= t_lim {
        // ventilation controlled
        let t_star_max = gamma * t_max;
        let t_max_temp = eq_3_12_t_g(t_star_max, t_0);
        for i in 0..n {
            let t_hr = t[i] / 3600.0; // s -> hr
            let t_star = gamma * t_hr;
            let heating = eq_3_12_t_g(gamma * t_hr, t_0);
            let cooling = eq_3_16_t_g(t_star_max, t_max_temp, t_star);
            let mut val = heating.min(cooling);
            if val < t_0 { val = t_0; }
            t_g[i] = val + 273.15; // C -> K
        }
    } else {
        // fuel controlled
        let o_lim = 0.0001 * q_td / t_lim;
        let mut gamma_lim = ((o_lim / 0.04) / (b / 1160.0)).powi(2);
        if o > 0.04 && q_td < 75.0 && b < 1160.0 {
            let k = 1.0 + ((o - 0.04) / 0.04) * ((q_td - 75.0) / 75.0) * ((1160.0 - b) / 1160.0);
            gamma_lim *= k;
        }
        let t_star_max = gamma * t_max; // used in cooling
        let t_max_temp = eq_3_12_t_g(gamma_lim * t_lim, t_0);
        for i in 0..n {
            let t_hr = t[i] / 3600.0;
            let t_star_f = gamma_lim * t_hr;
            let heating = eq_3_12_t_g(t_star_f, t_0);
            let t_star = gamma * t_hr;
            let cooling = eq_3_22_t_g(t_star_max, t_max_temp, t_star, gamma, t_lim);
            let mut val = heating.min(cooling);
            if val < t_0 { val = t_0; }
            t_g[i] = val + 273.15;
        }
    }
    t_g
}

// ---------------------------------------------------------------------------
// Travelling fire
// ---------------------------------------------------------------------------

/// Travelling fire temperature [degC] (NOT SI; see param units below).
/// Port of `travelling_fire_temperature`. Scalar `beam_location_length_m` only.
#[allow(clippy::too_many_arguments)]
pub fn travelling_fire_temperature(
    t: &[f64],
    fire_load_density_mjm2: f64,
    fire_hrr_density_mwm2: f64,
    mut room_length_m: f64,
    mut room_width_m: f64,
    fire_spread_rate_ms: f64,
    beam_location_height_m: f64,
    beam_location_length_m: f64,
    fire_nft_limit_c: f64,
) -> Vec<f64> {
    let q_fd = fire_load_density_mjm2;
    let hrrpua = fire_hrr_density_mwm2;
    let s = fire_spread_rate_ms;
    let h_s = beam_location_height_m;
    let l_s = beam_location_length_m;
    if room_length_m < room_width_m {
        // 3-temp-swap idiom to match Python exactly.
        room_length_m += room_width_m;
        room_width_m = room_length_m - room_width_m;
        room_length_m -= room_width_m;
    }
    let l = room_length_m;
    let w = room_width_m;

    let t_burn = (q_fd / hrrpua).max(900.0);
    let t_decay = t_burn.max(l / s);
    let t_lim = t_burn.min(l / s);

    let time_interval_s = t[1] - t[0];
    let t_decay_ = py_round(t_decay / time_interval_s) * time_interval_s;
    let mut t_lim_ = py_round(t_lim / time_interval_s) * time_interval_s;
    if t_decay_ == t_lim_ {
        t_lim_ -= time_interval_s;
    }

    let n = t.len();
    let q_peak_scalar = (hrrpua * w * s * t_burn).min(hrrpua * w * l);

    let mut t_g = vec![0.0_f64; n];
    for i in 0..n {
        let ti = t[i];
        // Heat release rate
        let q_growth = (hrrpua * w * s * ti) * if ti < t_lim_ { 1.0 } else { 0.0 };
        let q_peak = q_peak_scalar * if ti >= t_lim_ && ti <= t_decay_ { 1.0 } else { 0.0 };
        let mut q_decay = (q_peak_scalar - (ti - t_decay_) * w * s * hrrpua) * if ti > t_decay_ { 1.0 } else { 0.0 };
        if q_decay < 0.0 { q_decay = 0.0; }
        let q = (q_growth + q_peak + q_decay) * 1000.0;

        // Fire-front positions. NOTE: l_fire_end uses unsnapped t_lim.
        let mut l_fire_front = s * ti;
        if l_fire_front < 0.0 { l_fire_front = 0.0; }
        if l_fire_front > l { l_fire_front = l; }
        let mut l_fire_end = s * (ti - t_lim);
        if l_fire_end < 0.0 { l_fire_end = 0.0; }
        if l_fire_end > l { l_fire_end = l; }
        let l_fire_median = (l_fire_front + l_fire_end) / 2.0;

        // Temperature
        let r = (l_s - l_fire_median).abs();
        let ratio = r / h_s;
        let mut val = if ratio > 0.18 {
            (5.38 * (q / r).powf(2.0 / 3.0) / h_s) + 20.0
        } else {
            // near-field branch (matches the second np.where overriding the first)
            (16.9 * q.powf(2.0 / 3.0) / h_s.powf(5.0 / 3.0)) + 20.0
        };
        if val >= fire_nft_limit_c { val = fire_nft_limit_c; }
        t_g[i] = val;
    }
    t_g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parametric_fire_basic() {
        // Smoke: EC parametric curve should rise above ambient and peak then cool.
        let t: Vec<f64> = (0..1801).map(|i| i as f64 * 10.0).collect();
        let tg = parametric_fire_temperature(&t, 963.5, 500.0, 40.32, 2.8, 420e6,
                                             720.0 * 720.0, 1.0, 1.0, 0.333, 293.15);
        assert!(tg.iter().cloned().fold(0.0_f64, f64::max) > 800.0, "should get hot");
        assert!(tg[0] > 270.0, "starts near ambient");
    }

    #[test]
    fn test_py_round_half_to_even() {
        assert_eq!(py_round(0.5), 0.0);  // even
        assert_eq!(py_round(1.5), 2.0);  // even
        assert_eq!(py_round(2.5), 2.0);  // even
        assert_eq!(py_round(0.4), 0.0);
        assert_eq!(py_round(0.6), 1.0);
    }
}
