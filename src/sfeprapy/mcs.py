"""Monte Carlo driver for :func:`sfeprapy.teq_main`.

This module is **opt-in** and deliberately kept out of the core package import path:
``import sfeprapy`` does not load it (it needs ``scipy`` for sampling, which is a test
extra, not a runtime dependency). Import it explicitly when you want to run a
probabilistic analysis::

    from sfeprapy.mcs import run_monte_carlo

The driver does two jobs the deterministic core does not:

1. **Sample** stochastic input parameters from ``scipy.stats`` distributions, using
   stratified (Latin-hypercube-style) quantile sampling for stable CDFs at small N.
2. **Batch** the resulting calls across processes, since Monte Carlo workloads
   (1e6+ calls) are embarrassingly parallel.

The distribution specs use the same ``dict(dist=..., mean=..., sd=..., lbound=...,
ubound=...)`` shape as :data:`sfeprapy.EXAMPLE_INPUT`.
"""

import copy
import inspect
import multiprocessing as mp
from typing import Dict, List, Optional

import numpy as np

from .calcs import teq_main

__all__ = ('run_monte_carlo', 'sample_case', 'monte_carlo_distribution',)


# ---------------------------------------------------------------------
# Distribution specs -> scipy.stats frozen distributions
# ---------------------------------------------------------------------

def _true_to_scipy(dist_name: str, p: dict):
    """Convert 'natural' distribution parameters (mean/sd, lbound/ubound) to a
    ``scipy.stats`` frozen distribution."""
    import scipy.stats as st

    if dist_name == 'uniform_':
        a, b = p['lbound'], p['ubound']
        if a > b:
            a, b = b, a
        return st.uniform(loc=a, scale=b - a)
    if dist_name == 'norm_':
        return st.norm(loc=p['mean'], scale=p['sd'])
    if dist_name == 'gumbel_r_':
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


def monte_carlo_distribution(spec: dict):
    """Return a ``scipy.stats`` frozen distribution for an EXAMPLE_INPUT-style spec.

    Useful if you want to draw your own samples (e.g. random instead of stratified)
    rather than using :func:`sample_case`.
    """
    spec = copy.deepcopy(spec)
    dist_name = spec.pop('dist')
    return _true_to_scipy(dist_name, spec)


def _sample_one(spec: dict, n: int) -> np.ndarray:
    """Sample ``n`` stratified values from an EXAMPLE_INPUT-style stochastic spec.

    Stratified (quantile) sampling: one sample at each of ``n`` evenly-spaced
    probabilities between ``cdf(lbound)`` and ``cdf(ubound)``, then shuffled. More
    stable CDFs at small N than independent random draws.
    """
    spec = copy.deepcopy(spec)
    dist_name = spec.pop('dist')

    if dist_name == 'constant_':
        if 'lbound' in spec and 'ubound' in spec:
            value = (spec['lbound'] + spec['ubound']) / 2
        else:
            value = spec.get('value', 0)
        return np.full(n, value, dtype=float)

    dist = _true_to_scipy(dist_name, spec)
    lbound = spec.get('lbound')
    ubound = spec.get('ubound')
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


def sample_case(case_kwargs: dict, n: int) -> List[dict]:
    """Turn one EXAMPLE_INPUT-style case dict into a list of ``n`` concrete
    ``teq_main`` keyword dicts.

    Stochastic entries (``dict(dist=...)``) are sampled; scalar entries are
    broadcast; keys that are not ``teq_main`` parameters are dropped.
    """
    teq_params = set(inspect.signature(teq_main).parameters)

    sampled: Dict[str, np.ndarray] = {}
    static: Dict[str, object] = {}
    for k, v in case_kwargs.items():
        if k not in teq_params:
            continue
        if isinstance(v, dict) and 'dist' in v:
            sampled[k] = _sample_one(v, n)
        else:
            static[k] = v

    cases = []
    for i in range(n):
        kw = dict(static)
        for k, arr in sampled.items():
            kw[k] = arr[i]
        cases.append(kw)
    return cases


# ---------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------

def _worker(kw: dict):
    """Top-level worker for multiprocessing (must be picklable)."""
    return teq_main(**kw).solver_time_equivalence_solved


def run_monte_carlo(
        case_kwargs: dict,
        n_simulations: int,
        n_proc: Optional[int] = None,
        chunksize: int = 1000,
        seed: Optional[int] = None,
) -> np.ndarray:
    """Run a Monte Carlo simulation and return the array of solved time-equivalence [s].

    Parameters
    ----------
    case_kwargs : dict
        An EXAMPLE_INPUT-style case dict. ``dict(dist=...)`` entries are sampled;
        scalar entries are broadcast.
    n_simulations : int
        Number of independent samples to draw / ``teq_main`` calls to make.
    n_proc : int, optional
        Worker process count. ``None`` (default) uses ``os.cpu_count()``;
        ``1`` runs in-process (useful for debugging or when ``numba`` is threaded).
    chunksize : int
        Tasks-per-process-dispatch chunk. Tuned for amortizing IPC overhead on
        large N; the default is fine for most workloads.
    seed : int, optional
        Seed for the sampling shuffle. Pass it for reproducible runs.

    Returns
    -------
    np.ndarray
        Shape ``(n_simulations,)`` array of ``solver_time_equivalence_solved`` [s].

    Notes
    -----
    Only ``solver_time_equivalence_solved`` is collected from each call -- this is the
    high-throughput path for 1e6+ call workloads where only the time-equivalence
    distribution matters. If you need the full ``TeqResult`` per call, call
    :func:`sample_case` and ``teq_main`` yourself (single-process) or adapt this
    driver's ``_worker``.
    """
    if seed is not None:
        np.random.seed(seed)

    cases = sample_case(case_kwargs, n_simulations)

    if n_proc is None:
        n_proc = mp.cpu_count()

    if n_proc <= 1 or n_simulations <= chunksize:
        # in-process: no IPC overhead, friendliest for numba's JIT warmup
        return np.fromiter((_worker(kw) for kw in cases), dtype=float, count=len(cases))

    with mp.Pool(processes=n_proc) as pool:
        results = pool.imap(_worker, cases, chunksize=chunksize)
        return np.fromiter(results, dtype=float, count=len(cases))
