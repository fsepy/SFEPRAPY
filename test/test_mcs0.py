"""Tests for sfeprapy.

next-gen: there is no Monte Carlo orchestration any more, so the dist-dependent
tests sample stochastic inputs directly with ``scipy.stats`` and loop ``teq_main``.
The three tests below are:

* ``test_teq_scalar``          -- pure scalar check of ``teq_main`` (fast, no sampling).
* ``test_heat_transfer_fidelity``-- pins the vendored pure-Python heat-transfer port
                                    to a known fsetools reference result (fast).
* ``test_standard_case``        -- a small Monte Carlo run that samples ``EXAMPLE_INPUT``
                                    with ``scipy.stats`` and checks the time-equivalence CDF
                                    thresholds (slower; sample size kept modest because the
                                    pure-Python solver is the bottleneck).
"""

import copy

import numpy as np

from sfeprapy import (
    EXAMPLE_INPUT,
    teq_main,
)
from sfeprapy._fsetools import protection_thickness_2, temperature


# =====================================================================
# 1. Pure-function regression: teq_main with scalar inputs
# =====================================================================

def test_teq_scalar():
    """End-to-end smoke test of the deterministic time-equivalence calculation with
    fixed scalar inputs (no sampling). Verifies ``teq_main`` runs and returns a
    physically sensible, positive equivalent time exposure for an EC parametric fire."""
    import warnings

    warnings.filterwarnings("ignore")

    input_param = dict(
        fire_time_step=1., fire_time_duration=5. * 60 * 60, beam_cross_section_area=0.017,
        beam_position_vertical=2.5, beam_position_horizontal=18, beam_rho=7850.,
        fire_combustion_efficiency=0.8,
        fire_hrr_density=0.25, fire_load_density=420, fire_mode=0,
        fire_nft_limit=1050,
        fire_spread_speed=0.01, fire_tlim=0.333, protection_c=1700., protection_k=0.2,
        protection_protected_perimeter=2.14, protection_rho=800., room_breadth=16, room_depth=31.25,
        room_height=3,
        room_wall_thermal_inertia=720, solver_temperature_goal=620 + 273.15,
        solver_tol=0.01, window_height=2,
        window_width=57.6,
        timber_burning_rate=0,
        timber_solver_ilim=20,
        timber_solver_tol=1,
    )

    result = teq_main(**input_param)
    teq = result.solver_time_equivalence_solved

    print(f'Solved equivalent time exposure: {teq:.3f} s ({teq / 60:.1f} min)')
    print(f'  fire_type={result.fire_type}, protection_thickness={result.solver_protection_thickness:.5f} m')

    # Sanity: a 420 MJ/m2 fuel load in this compartment should give a positive,
    # finite equivalent time exposure well inside the 5 h (18000 s) fire duration.
    assert np.isfinite(teq)
    assert 0 < teq < input_param['fire_time_duration']


# =====================================================================
# 2. Fidelity of the vendored pure-Python heat-transfer port
#    (reference case lifted from fsetools' own test_protection_thickness)
# =====================================================================

def _trav_fire(t: np.ndarray):
    """Travelling-fire gas curve used by the fsetools reference test."""
    from sfeprapy._fsetools import travelling_fire_temperature
    # temperature_si is not vendored; reproduce it inline (SI units).
    T_0 = 273.15
    q_f_d = 600e6
    hrrpua = 0.25e6
    l, w = 100, 16
    s = 0.012
    e_h = 3
    e_l = 50
    T_max = 1050 + 273.15

    T_0 -= 273.15
    q_f_d /= 1e6
    hrrpua /= 1e6
    T_max -= 273.15

    time_step = t[1] - t[0]
    t_burn = max([q_f_d / hrrpua, 900.0])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])
    t_decay_ = round(t_decay / time_step, 0) * time_step
    t_lim_ = round(t_lim / time_step, 0) * time_step
    if t_decay_ == t_lim_:
        t_lim_ -= time_step

    Q_growth = (hrrpua * w * s * t) * (t < t_lim_)
    Q_peak = min([hrrpua * w * s * t_burn, hrrpua * w * l]) * (t >= t_lim_) * (t <= t_decay_)
    Q_decay = (max(Q_peak) - (t - t_decay_) * w * s * hrrpua) * (t > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.0

    l_fire_front = s * t
    l_fire_front[l_fire_front < 0] = 0.0
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (t - t_lim)
    l_fire_end[l_fire_end < 0] = 0.0
    l_fire_end[l_fire_end > l] = l
    l_fire_median = (l_fire_front + l_fire_end) / 2.0
    r = np.absolute(e_l - l_fire_median)
    r[r == 0] = 0.001

    T_g1 = (5.38 * np.power(Q / r, 2 / 3) / e_h) * ((r / e_h) > 0.18)
    T_g2 = (16.9 * np.power(Q, 2 / 3) / np.power(e_h, 5 / 3)) * ((r / e_h) <= 0.18)
    T_g = T_g1 + T_g2 + T_0
    T_g[T_g >= T_max] = T_max
    return T_g + 273.15  # C -> K


def test_heat_transfer_fidelity():
    t = np.arange(0, 210 * 60, 1, dtype=float)

    solver_d_p, solver_T_a_max, _, _, status = protection_thickness_2(
        fire_time=t,
        fire_temperature=_trav_fire(t),
        beam_rho=7850.,
        beam_cross_section_area=0.017,
        protection_k=0.2,
        protection_rho=800.,
        protection_c=1700.,
        protection_protected_perimeter=2.14,
        solver_temperature_goal=873.15 + 20,
        solver_temperature_goal_tol=0.1,
    )

    print(
        f'Solved protection thickness   {solver_d_p:<8.4} m\n'
        f'Solved max. steel temperature {solver_T_a_max - 273.15:<8.2f} C\n'
        f'Solver status                 {status}'
    )

    # Reference values from fsetools test_protection_thickness (solution on 05/10/2020)
    assert abs(solver_T_a_max - (873.15 + 20)) <= 0.1
    assert abs(solver_d_p - 0.01556) <= 1e-5

    # And the solved thickness must reproduce the peak temperature through the full history.
    T = temperature(
        fire_time=t,
        fire_temperature=_trav_fire(t),
        beam_rho=7850.,
        beam_cross_section_area=0.017,
        protection_k=0.2,
        protection_rho=800.,
        protection_c=1700.,
        protection_thickness=solver_d_p,
        protection_protected_perimeter=2.14,
    )
    assert abs(np.amax(T) - solver_T_a_max) < 1e-3


# =====================================================================
# 3. Small Monte Carlo run: sample EXAMPLE_INPUT with scipy.stats
# =====================================================================

# Kept modest because the pure-Python protection-thickness solver is the bottleneck.
N_SIM = 500


def _true_to_scipy(dist_name, p):
    """Convert 'natural' distribution parameters (mean/sd, lbound/ubound) to scipy.stats
    parameters. Mirrors the (deleted) sfeprapy.input_parser.TrueToScipy conversions."""
    import scipy.stats as st

    if dist_name in ('uniform_',):
        a, b = p['lbound'], p['ubound']
        if a > b:
            a, b = b, a
        return st.uniform(loc=a, scale=b - a)

    if dist_name in ('norm_',):
        return st.norm(loc=p['mean'], scale=p['sd'])

    if dist_name in ('gumbel_r_',):
        mean, sd = p['mean'], p['sd']
        alpha = 1.282 / sd
        u = mean - 0.5772 / alpha
        return st.gumbel_r(loc=u, scale=1.0 / alpha)

    if dist_name in ('lognorm_', 'lognorm_mod_'):
        mean, sd = p['mean'], p['sd']
        cov = sd / mean
        sigma_ln = np.sqrt(np.log(1 + cov ** 2))
        mu_ln = np.log(mean) - 0.5 * sigma_ln ** 2
        return st.lognorm(s=sigma_ln, loc=0, scale=np.exp(mu_ln))

    raise ValueError(f'Unsupported distribution {dist_name!r}')


def _sample(spec, n):
    """Sample `n` stratified values from an EXAMPLE_INPUT stochastic spec, reproducing the
    old DistFunc.sampling scheme: evenly spaced quantiles between cdf(lbound) and cdf(ubound),
    then shuffled."""
    spec = copy.deepcopy(spec)
    dist_name = spec.pop('dist')

    if dist_name == 'constant_':
        value = (spec.get('lbound', 0) + spec.get('ubound', 0)) / 2 if 'lbound' in spec and 'ubound' in spec \
            else spec.get('value', 0)
        return np.full(n, value, dtype=float)

    dist = _true_to_scipy(dist_name, spec)
    lbound = spec.get('lbound', None)
    ubound = spec.get('ubound', None)
    c_lo = dist.cdf(lbound) if lbound is not None else 1.0 / n
    c_hi = dist.cdf(ubound) if ubound is not None else 1.0 - 1.0 / n
    q = np.linspace(c_lo, c_hi, n)
    samples = dist.ppf(q)

    if dist_name == 'lognorm_mod_':
        samples = 1.0 - samples

    # guard against +/- inf at the tails
    if ubound is not None:
        samples[samples == np.inf] = ubound
        samples[samples > ubound] = ubound
    if lbound is not None:
        samples[samples == -np.inf] = lbound
        samples[samples < lbound] = lbound

    np.random.shuffle(samples)
    return samples


def _resolve_case(case_kwargs, n):
    """Turn an EXAMPLE_INPUT case dict into a list of `n` teq_main keyword dicts.

    Stochastic entries (``dict(dist=...)``) are sampled; scalar/array entries are
    broadcast; keys that are not teq_main parameters are dropped."""
    import inspect

    teq_params = set(inspect.signature(teq_main).parameters)

    # separate stochastic vs static, restricted to teq_main's accepted parameters
    sampled = {}
    static = {}
    for k, v in case_kwargs.items():
        if k not in teq_params:
            continue
        if isinstance(v, dict) and 'dist' in v:
            sampled[k] = _sample(v, n)
        else:
            static[k] = v

    cases = []
    for i in range(n):
        kw = dict(static)
        for k, arr in sampled.items():
            kw[k] = arr[i]
        cases.append(kw)
    return cases


def _make_cdf(teq_values, bin_width=0.2):
    """Reproduce MCSSingle.make_cdf: clip to [0, 18000] s, convert to minutes, bin into a PDF."""
    data = np.asarray(teq_values, dtype=float)
    data[data >= 18000.] = 17999.999
    data[data <= 0] = 1e-3
    data /= 60.0  # s -> min

    edges = np.arange(0, 300 + bin_width, bin_width)
    x = (edges[1:] + edges[:-1]) / 2
    y_pdf = np.histogram(data, edges)[0] / len(data)
    return x, np.cumsum(y_pdf)


def test_standard_case():
    import time

    expectations = {
        # case_name: (x_threshold_minutes, cdf_at_that_x ~ 0.8 within +/- 0.5)
        'CASE_1': (60.0, 0.8),  # ~60 min based on Kirby et al.
        'CASE_2_teq_phi': (64.5, 0.8),  # ~63 min based on a test run on 16th Aug 2022
        'CASE_3_timber': (81.0, 0.8),  # ~78 min based on a test run on 16th Aug 2022
    }

    for case_name, (x_thr, cdf_target) in expectations.items():
        case = copy.deepcopy(EXAMPLE_INPUT[case_name])
        kw_list = _resolve_case(case, N_SIM)

        t0 = time.time()
        teq_values = [teq_main(**kw).solver_time_equivalence_solved for kw in kw_list]
        dt = time.time() - t0

        x, y = _make_cdf(teq_values)
        observed = np.amax(y[x < x_thr])
        print(f'{case_name:<16} n={N_SIM:<6d} {dt:6.1f}s  '
              f'CDF(x<{x_thr:g} min)={observed:6.3f} (target ~{cdf_target} +/- 0.5)')

        assert abs(observed - cdf_target) <= 0.5


if __name__ == '__main__':
    test_teq_scalar()
    test_heat_transfer_fidelity()
    test_standard_case()
