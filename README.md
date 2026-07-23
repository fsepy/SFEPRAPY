# SFEPRAPY

SFEPRAPY provides a single routine, `teq_main`, that calculates the **equivalent time
exposure** of a protected steel element to the ISO 834 standard fire. Given a set of
physical input parameters (compartment geometry, ventilation, fire load, protection
properties, and a target steel failure temperature), it returns the equivalent time in
seconds.

This routine is the building block for probabilistic (Monte Carlo) structural fire
reliability analysis: sample the input parameters from their distributions, call
`teq_main` for each draw, and build a distribution of equivalent time exposures.
**Random sampling is deliberately not part of this repository** — the library is the
deterministic physics only. Callers sample inputs themselves (e.g. with `scipy.stats`);
see `test/test_mcs0.py::test_standard_case` for a worked example.

## Usage

```python
from sfeprapy import teq_main, EXAMPLE_INPUT

result = teq_main(**{
    'fire_time_step': 10.0, 'fire_time_duration': 18000,
    'beam_cross_section_area': 0.017, 'beam_rho': 7850,
    'beam_position_vertical': 3.1, 'beam_position_horizontal': 18,
    'fire_combustion_efficiency': 0.8,
    'fire_hrr_density': 0.25, 'fire_load_density': 420, 'fire_mode': 0,
    'fire_nft_limit': 1050, 'fire_spread_speed': 0.01, 'fire_tlim': 0.333,
    'protection_c': 1700, 'protection_k': 0.2, 'protection_protected_perimeter': 2.14, 'protection_rho': 800,
    'room_breadth': 16, 'room_depth': 31.25, 'room_height': 3, 'room_wall_thermal_inertia': 720,
    'solver_temperature_goal': 823.15, 'solver_tol': 1.0,
    'window_height': 2.8, 'window_width': 14.4,
    'timber_burning_rate': 0, 'timber_solver_ilim': 20, 'timber_solver_tol': 1,
})

# result is a TeqResult NamedTuple -- read fields by name
time_equivalence = result.solver_time_equivalence_solved  # [s]
```

`EXAMPLE_INPUT` documents every accepted parameter and its units; the stochastic entries
there (`dict(dist=..., ...)`) describe how each parameter would be sampled in a Monte
Carlo run, but are not consumed by `teq_main` itself.

## Probabilistic analysis (Monte Carlo)

For workloads of 1e6+ calls, `sfeprapy.mcs` provides an opt-in driver that samples the
stochastic inputs and batches the calls across processes. It needs `scipy` (for sampling),
so it's a separate import from the core:

```python
from sfeprapy import EXAMPLE_INPUT
from sfeprapy.mcs import run_monte_carlo

# Returns an (n_simulations,) array of solver_time_equivalence_solved [s]
teq = run_monte_carlo(EXAMPLE_INPUT['CASE_1'], n_simulations=10_000, n_proc=8, seed=42)
```

`run_monte_carlo` collects only `solver_time_equivalence_solved` (the high-throughput
path). If you need the full `TeqResult` per call, use `sfeprapy.mcs.sample_case` to get
the resolved kwargs and call `teq_main` directly.

## Installation

Python 3.8 or later. The runtime dependency is `numpy` only.

```sh
pip install --upgrade "git+https://github.com/fsepy/SfePrapy.git@next-gen"
```

`scipy` is not a runtime dependency -- it's only used by the test suite for stochastic
sampling. Install with `[test]` to run the tests, or `[fast]` for the numba speedup:

```sh
pip install -e ".[test,fast]"
```

### Performance (`[fast]`)

The steel-temperature solver is the hot path in Monte Carlo workloads. Installing the
optional `[fast]` extra (`numba`) JIT-compiles that kernel and gives roughly a **25×
speedup**; the package falls back to pure Python automatically if numba is unavailable
(e.g. in Pyodide/WASM, or on a Python version numba does not yet support).

## Browser demo (Pyodide / WebAssembly)

`demo/index.html` runs `teq_main` entirely in the browser via
[Pyodide](https://pyodide.org) (CPython compiled to WASM). The same Python source is
used -- nothing is ported. To try it locally, serve the `demo/` folder over HTTP (a
`file://` URL won't fetch the wheel), then open it in a browser:

```sh
python -m build --wheel --outdir demo/
python -m http.server --directory demo 8000
# open http://localhost:8000/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
