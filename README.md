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
    'index': 0,
    'fire_time_step': 10.0, 'fire_time_duration': 18000,
    'beam_cross_section_area': 0.017, 'beam_rho': 7850,
    'beam_position_vertical': 3.1, 'beam_position_horizontal': 18,
    'fire_combustion_efficiency': 0.8, 'fire_gamma_fi_q': 1,
    'fire_hrr_density': 0.25, 'fire_load_density': 420, 'fire_mode': 0,
    'fire_nft_limit': 1050, 'fire_spread_speed': 0.01, 'fire_t_alpha': 300, 'fire_tlim': 0.333,
    'protection_c': 1700, 'protection_k': 0.2, 'protection_protected_perimeter': 2.14, 'protection_rho': 800,
    'room_breadth': 16, 'room_depth': 31.25, 'room_height': 3, 'room_wall_thermal_inertia': 720,
    'solver_temperature_goal': 823.15, 'solver_max_iter': 20,
    'solver_thickness_lbound': 0.0001, 'solver_thickness_ubound': 0.04, 'solver_tol': 1.0,
    'window_height': 2.8, 'window_open_fraction': 0.8, 'window_width': 72,
    'window_open_fraction_permanent': 0, 'phi_teq': 1.0,
    'timber_charring_rate': 0.7, 'timber_exposed_area': 0, 'timber_hc': 13.2, 'timber_density': 400,
    'timber_solver_ilim': 20, 'timber_solver_tol': 1,
})

# result[16] is the solved equivalent time exposure [s]
time_equivalence = result[16]
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
