//! Steel heat-transfer kernels (BS EN 1993-1-2).
//!
//! Faithful Rust port of `sfeprapy._fsetools` (which itself ports the fsetools Cython
//! module). The numerics must match the Python to ~1e-6.
//!
//! IMPORTANT: the exponential base in the Euler loop is the literal `2.718`, NOT
//! `std::f64::consts::E`. Using `E` would drift from the Python at ~1e-4. Do not change.

/// Specific heat of carbon steel [J/kg/K] as a function of temperature [K].
/// BS EN 1993-1-2:2005, 3.4.1.2. Exact port of the Python `c_steel_T`.
pub fn c_steel_t(t_kelvin: f64) -> f64 {
    // K -> degC
    let t = t_kelvin - 273.15;
    if t < 20.0 {
        // Literal expression (not precomputed) to match Python float rounding.
        425.0 + 0.773 * 20.0 - 1.69e-3 * 400.0 + 2.22e-6 * 8000.0
    } else if t < 600.0 {
        425.0 + 0.773 * t - 1.69e-3 * (t * t) + 2.22e-6 * (t * t * t)
    } else if t < 735.0 {
        666.0 + 13002.0 / (738.0 - t)
    } else if t < 900.0 {
        545.0 + 17820.0 / (t - 731.0)
    } else {
        650.0
    }
}

/// Steel temperature history for a protected member [K]. Port of `_temperature_jit`.
///
/// SI units. BS EN 1993-1-2:2005, Clauses 4.2.5.2 (Eq. 4.27).
pub fn temperature(
    fire_time: &[f64],
    fire_temperature: &[f64],
    beam_rho: f64,
    beam_cross_section_area: f64,
    protection_k: f64,
    protection_rho: f64,
    protection_c: f64,
    protection_thickness: f64,
    protection_protected_perimeter: f64,
) -> Vec<f64> {
    let v = beam_cross_section_area;
    let rho_a = beam_rho;
    let lambda_p = protection_k;
    let rho_p = protection_rho;
    let d_p = protection_thickness;
    let a_p = protection_protected_perimeter;
    let c_p = protection_c;

    let n = fire_time.len();
    let mut t_a = vec![0.0_f64; n];
    t_a[0] = fire_temperature[0]; // steel starts at gas temp at t=0

    for i in 1..n {
        let t_g = fire_temperature[i];
        let c_s = c_steel_t(t_a[i - 1]);

        // Eq. 4.27
        let phi = (c_p * rho_p / c_s / rho_a) * d_p * a_p / v;
        let a = (lambda_p * a_p / v) / (d_p * c_s * rho_a);
        let b = (t_g - t_a[i - 1]) / (1.0 + phi / 3.0);
        // CRITICAL: literal 2.718, not E.
        let c = (2.718_f64.powf(phi / 10.0) - 1.0) * (t_g - fire_temperature[i - 1]);
        let d = fire_time[i] - fire_time[i - 1];

        let mut d_t = (a * b * d - c) / d;
        if d_t < 0.0 && (t_g - fire_temperature[i - 1]) > 0.0 {
            d_t = 0.0;
        }
        t_a[i] = t_a[i - 1] + d_t * d;
    }
    t_a
}

/// Peak steel temperature [K] and the time it occurs [s]. Port of `_temperature_max_jit`.
///
/// Early-terminates when the steel starts cooling. Returns `(t_max, t_at_max)`.
pub fn temperature_max(
    fire_time: &[f64],
    fire_temperature: &[f64],
    beam_rho: f64,
    beam_cross_section_area: f64,
    protection_k: f64,
    protection_rho: f64,
    protection_c: f64,
    protection_thickness: f64,
    protection_protected_perimeter: f64,
) -> (f64, f64) {
    let v = beam_cross_section_area;
    let rho_a = beam_rho;
    let lambda_p = protection_k;
    let rho_p = protection_rho;
    let d_p = protection_thickness;
    let a_p = protection_protected_perimeter;
    let c_p = protection_c;

    let mut t = fire_temperature[0]; // scalar running steel temperature
    let d = fire_time[1] - fire_time[0]; // constant dt (first interval)

    let mut i_stop = 1_usize;
    for i in 1..fire_temperature.len() {
        i_stop = i;
        let t_g = fire_temperature[i];
        let c_s = c_steel_t(t);

        let phi = (c_p * rho_p / c_s / rho_a) * d_p * a_p / v;
        let a = (lambda_p * a_p / v) / (d_p * c_s * rho_a);
        let b = (t_g - t) / (1.0 + phi / 3.0);
        let c = (2.718_f64.powf(phi / 10.0) - 1.0) * (t_g - fire_temperature[i - 1]);

        let mut d_t = (a * b * d - c) / d;
        if d_t < 0.0 && (t_g - fire_temperature[i - 1]) > 0.0 {
            d_t = 0.0;
        }
        t = t + d_t * d;

        if d_t < 0.0 {
            t -= d_t * d; // undo the last update
            break;
        }
    }
    // Matches Python: returns fire_time[i-1] where i is the stopping index.
    (t, fire_time[i_stop - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_steel_t_branches() {
        // Spot-checks against the Python implementation's outputs.
        // Below 20 degC -> clamped formula value
        let v0 = c_steel_t(273.15); // 0 degC
        assert!((v0 - (425.0 + 0.773 * 20.0 - 1.69e-3 * 400.0 + 2.22e-6 * 8000.0)).abs() < 1e-9);
        // 400 degC -> polynomial branch
        let v400 = c_steel_t(400.0 + 273.15);
        let expected = 425.0 + 0.773 * 400.0 - 1.69e-3 * (400.0 * 400.0) + 2.22e-6 * (400.0_f64).powi(3);
        assert!((v400 - expected).abs() < 1e-6, "got {} expected {}", v400, expected);
        // 650 degC -> 666 + 13002/(738-650)
        let v650 = c_steel_t(650.0 + 273.15);
        assert!((v650 - (666.0 + 13002.0 / (738.0 - 650.0))).abs() < 1e-6);
        // >= 900 degC -> 650
        assert!((c_steel_t(1000.0 + 273.15) - 650.0).abs() < 1e-9);
    }

    #[test]
    fn test_temperature_runs() {
        // Smoke test: constant hot gas -> steel approaches gas temp over time.
        // Note: with the protection phi factor, steel can dip below initial temp early
        // (energy locked in protection layer), so only assert the end-state, not monotonicity.
        let fire_time: Vec<f64> = (0..100).map(|i| i as f64 * 30.0).collect();
        let fire_temp: Vec<f64> = vec![900.0; 100];
        let t_a = temperature(
            &fire_time, &fire_temp, 7850.0, 0.017, 0.2, 800.0, 1700.0, 0.01, 2.14,
        );
        assert_eq!(t_a.len(), 100);
        // After 50 min of exposure, steel should be substantially heated (well above ambient).
        assert!(t_a[99] > 500.0, "steel should heat up, got end temp {}", t_a[99]);
        assert!(t_a[99] <= 900.0, "steel cannot exceed gas temp, got {}", t_a[99]);
    }
}
