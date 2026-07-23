# Release

Checklist before release of new versions:

- Run tests in `test/`.
- Version is up to date in `src/sfeprapy/_version.py`.

## Version history

### next-gen VERSION: 0.9.0

**Breaking.** Massive simplification of the repository.

- Removed: Monte Carlo orchestration (`sfeprapy.mcs`), the custom distribution
  machinery (`sfeprapy.dists`, `sfeprapy.func.erf`), the input parser
  (`sfeprapy.input_parser`), the `mcs1` and `mcs2` modules, the CLI
  (`sfeprapy.cli`), file I/O helpers (`sfeprapy.func.xlsx`, `sfeprapy.func.csv`),
  the profiler, the demo notebooks, and stale CI config.
- Removed: runtime dependency on `fsetools`. The fire and steel heat-transfer
  physics used by `teq_main` have been vendored into `sfeprapy._fsetools` as
  pure Python (the former Cython heat-transfer module has been ported 1:1).
- Removed: `xlrd` and `openpyxl` dependencies.
- Added: `scipy` as a runtime dependency (used by the test suite for stochastic
  sampling; `teq_main` itself needs only `numpy`).
- Kept: `sfeprapy.teq_main` and its helpers (`decide_fire`,
  `evaluate_fire_temperature`, `solve_protection_thickness`,
  `solve_time_equivalence_iso834`) and `EXAMPLE_INPUT`, all behaviourally unchanged.

The public surface is now the single deterministic function `teq_main`.

### xx/xx/2020 VERSION: 0.7.2

- [ ] Added: 1D heat transfer module.
- [x] Improved: `cli` mcs0 template save as `.xlsx`.
- [x] Fixed: `cli` mcs0 template blank lines when saved as `.csv`.

### 01/02/2020 VERSION: 0.7.0

- New: GUI added to `sfeprapy.mcs0`, can be summoned in CLI by `sfeprapy mcs0 gui`.
- Improved: various improvements see repository commits.

### 28/10/2019 VERSION: 0.6.9

- New: repository feature, codecov integration.
- New: repository feature, travis integration.
- Improved: CLI commands simplified and updated.
- Improved: updated and some new test functions in \test directory.

### 23/10/2019 VERSION: 0.6.8

- New: use `sfeprapy` to trigger recently refreshed CLI.  Previously this is  `python -m sfeprapy.mcs0`.
- Improved: CLI, added figure and save template input file features. Use `sfeprapy -h` to find more information.
- Depreciated: `sfeprapy.mcs0` module can no longer be triggered directly as `python -m sfeprapy.mcs0` (i.e. codes are removed after `if __name__ == '__main__'` .
- Depreciated: `sfeprapy.mcs0` GUI to select input file, use CLI instead.

### 10/10/2019 VERSION: 0.6.7

- New: `sfeprapy.mcs0` added exposure time dependent timber charring rate.
- New: `sfeprapy.func.mcs_gen` added `ramp` input variable type for time dependent variables.
- New: `sfeprapy.mcs0` added plotting feature, to activate `python -m sfeprapy.mcs0 {fp} mp2 fig`
- Depreciated: `sfeprapy.mc0`. This module is imported/combined into `sfeprapy.mcs0` at version 0.6.
- Depreciated: `sfeprapy.mcs2`. This module is imported/combined into `sfeprapy.mcs0` at version 0.6.5.

### 12/09/2019 VERSION: 0.6.6

- New: system arguments added to `python -m sfeprapy.mcs0` command line call. For example, calling `python -m sfeprapy.mcs0 example_input.csv mp4` will run the problem definition file `example_input.csv` with 4 processes.

### 27/08/2019 VERSION: 0.6.5

- New: `phi_teq` is added in `sfeprapy.mcs0` to include Model Uncertainty factor, see [README](README) for details.
- New: `sfeprapy.mcs0` implemented timber fuel contribution.
- New: timber-related parameters added in `sfeprapy.mcs0`.
- Improved: `sfeprapy.mcs0` changed criteria for parametric fire when `fire_mode` is set to 3. `opening_factor` should be within 0.01 and 0.2 (instead 0.02 and 0.2) to compliant to UK NA to Eurocode 1991-1-2.
