# SFEPRAPY

Structural fire engineering (Sfe) probabilistic reliability assessment (Pra) Python (py). It
calculates the equivalent time exposure to the ISO 834 standard fire for a protected steel
element, which can be used to assess an appropriate fire resistance rating using reliability
based methods.

## next-gen

This branch is a massive simplification. The library is now a **single pure function** --
`sfeprapy.mcs0.teq_main` -- plus its supporting helpers. All of the previous Monte Carlo
orchestration, parallel batching, custom distribution machinery, file I/O, and the CLI have
been removed. The fire and steel heat-transfer physics (previously in
[`fsetools`](https://github.com/fsepy/fsetools)) has been vendored into
`sfeprapy.mcs0._fsetools` as pure Python, so there are **no compiled dependencies**.

Pass `teq_main` sampled parameters, get a result tuple back.

```python
from sfeprapy.mcs0 import teq_main, EXAMPLE_INPUT

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

`EXAMPLE_INPUT` documents every accepted parameter and its units; stochastic entries there
(`dict(dist=..., ...)`) describe how each parameter would be sampled in a Monte Carlo run --
sampling itself is left to the caller (see `test/test_mcs0.py::test_standard_case` for a
`scipy.stats` example).

## Installation

Python 3.8 or later.

```sh
pip install --upgrade "git+https://github.com/fsepy/SfePrapy.git@next-gen"
```

### Development

The package uses a `src/` layout, so install it editable (with test deps) to run the
suite from a checkout:

```sh
pip install -e ".[test]"
pytest
```

## Browser demo (Pyodide / WebAssembly)

`demo/index.html` runs `teq_main` entirely in the browser via
[Pyodide](https://pyodide.org) (CPython compiled to WASM). The same Python source is
used -- nothing is ported. It loads `numpy`/`scipy` (built into Pyodide) and installs the
pure-Python sfeprapy wheel via `micropip`. The result surfaces as a plain JS object.

To try it locally, serve the `demo/` folder over HTTP (a `file://` URL won't fetch the
wheel), then open it in a browser:

```sh
# rebuild the wheel first if the package changed
python -m build --wheel --outdir demo/
python -m http.server --directory demo 8000
# open http://localhost:8000/
```

First load downloads the Pyodide runtime (~10 MB); subsequent runs are instant.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
