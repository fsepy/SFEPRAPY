# RELEASE NOTES

## KNOWN ISSUES AND TASKS LIST

- [ ] sfeprapy.mc0 to remove intermediate json files, to run MCS for all cases in one go.
- [ ] sfeprapy.mc0.mc0_func_gen, combine all MCS input DataFrames for individual cases into one DataFrame. One more column is needed to record case names.
- [ ] sfeprapy.mc0.mc0 to refactor `path_work`, `input_master` and `config_master`, so the function handles inputs from Python rather than GUI file picker.
- [ ] sfeprapy.mc0.mc0 to implement MCS for multiple cases in one DataFrame, to run MCS on case to case basis in alphabetical order, to to_csv after individual cases are run, to able to set `is_live` for individual cases.

## VERSIONS

### XX/XX/2019 VERSION: 0.6.1

- New: `sfeprapy.func.mcs_gen` general purpose stochastic variable generator.
- New: `sfeprapy.func_mcs_obj` general purpose Monte Carlo Simulation object.
- New: `sfeprapy.mcs0` to implement `mcs_gen` and `mcs_obj` into the time equivalence calculation.
- New: `sfeprapy.mcs0` when `beam_loc` is removed, the most onerous location will be calculated and used based on specific fire curve.
- Improved: `sfeprapy.mc1` MCS routine is converted to an object, MonteCarloCase and MonteCarlo classes are provided to substitute existing factions.
- Fixed: `sfeprapy.mc1` convergence of protection thickness maybe failed to find.
- Fixed: `sfeprapy.dat.ec_3_1_2kyT` units fixed, used degree K but should be degree C.

### 15/04/2019 VERSION: 0.5

- New: `sfeprapy.pd6688.annex_b_equivalent_time_of_fire_exposure` PD 6688 equivalent time exposure calculation. Manual can be found in its docstring `annex_b_equivalent_time_of_fire_exposure.__doc__`.
- New: (WIP) `sfeprapy.mc1` new equivalent time exposure procedure.
- Improved: `sfeprapy.mc` optimised temperature dependent steel heat capacity routine, resulted in 65% less simulation time. Tested case shows 32.8 seconds reduced to 16.7 seconds for 1000 simulations on i7-7660U with 2 threads.
- Fixed: `sfeprapy.mc.mc_inputs_generator.py:mc_inputs_generator` eliminated nan values in sampled stochastic variables - it is discovered negative or positive values are sampled even with predefined boundary limits, these extreme values (i.e. ±inf) are replaced with prescribed (user defined) limits (i.e. lbound and ubound).

### 31/03/2019 VERSION: 0.4

- New: `sfeprapy.mc` figure size is customisable in config.json file.
- New: `sfeprapy.mc` figure x-axis limits (i.e. t_eq) is customisable in config.json file.
- Fixed: `sfeprapy.mc` final time equivalence 'stepping' issue due to tolerance of the solver is now fixed by replacing existing bisection by a secant.
- Improved: `sfeprapy.mc` input parameter (in master .csv file) `fire_mode` is added, replacing `fire_type_enforced`.
- Improved: `sfeprapy.mc` simulation time improved. Tested case shows 256 seconds reduced to 26 seconds for 1000 simulations on i7-3770k with 6 threads.
- Improved: `sfeprapy.mc` refreshed data flow - master .csv input file -> individual case *.json file -> individual case MC *_in.csv file -> individual case result *_out.csv file -> *.png files.

### 11/03/2019 VERSION: 0.3

- Improved: Updated the decision making for choosing between travelling fire and parametric fire, introduced q_td limit criteria, i.e. 50 <= q_td <= 1000.

### 11/03/2019 VERSION: 0.2

- SfePrapy is now in beta testing phase;
- New: Eurocode DIN Annex parametric fire curve is now available, force simulations to use only this fire curve by set `fire_type_enforced` to 2 (see new input file template); and
- New: Feature to find best fit distribution function - fire this tool by `python -m sfeprapy.dist_fit`.

### 07/03/2019 VERSION: 0.0.8

- Improved: Able to combine individual time equivalence curves into one with corresponding `probability_weight` (see new input file template).

### 17/02/2019 VERSION: 0.0.7

- New: Additional figure named `t_eq_merged.png` is produced to show merged time equivalency of all cases.

### 22/01/2019 VERSION: 0.0.6

- Fixed: Final time equivalence plot legend text is set to case name. Previously '_out' was included.
- Improved: Final time equivalence plot will be produced without running any simulation, e.g. all `is_live` is set to 0.

### 01/01/2019 VERSION: 0.0.5

- New: More configuration parameters can be defined in 'config.json' file located in the same folder as the selected input file;
- New: Optional MC simulation calculation for each case (as in the input *.csv file);
- New: beam_loc_z (beam element height), room_opening_permanent_ratio (permanent free window opening area);
- Improved: Plot background transparent.

### 31/10/2018 VERSION: 0.0.4

- Additional returned results from the Monte Carlo simulation tool `sfeprapy.time_equivalence_core.grouped_a_b`. Window opening factor `opening_factor` is added to be returned from the function.
- `sfeprapy.time_equivalence.app` is now able to main_args a single simulation. When 'simulations=1' is defined, all distributed variables are disabled and mean or mean(upper, lower) is used for stochastic parameters.
- New testing input file 'benchmark_file_1' is added for single simulation testing, all other parameters are identical to 'benchmark_file_0'. Benchmark files are moved to validation folder, contained in root directory.

### 21/08/2018 VERSION: 0.0.3

- Updated code relating simulation output result \*.p and \*res.csv files. This is to fix an issue which output fires do not align with input / output index numbering. The new \*.p and \*res.csv files are sorted by time equivalence. The new output files are significantly larger than previous versions due to more variables are being passed in and out from the grouped_a_b calculation function `sfeprapy.time_equivalence_core.grouped_a_b()`.
- Fire duration `fire_duration` is checked and reassigned if necessary so that the slowest travelling fire is able to travel the entire room `room_depth`. `fire_duration` defined in the input file will be the minimum fire duration.
- Verification procedures are added for part of the project, including parametric fire testing, travelling fire testing and Eurocode protected steel heat transfer.

### 15/08/2018 VERSION: 0.0.2

- Graphical folder select dialog is available;
- Fixed an issue associated with `sfeprapy.time_equivalence.main_args()` where would not ask for new input folder directory when main_args more than once without re-import the module;
- Fixed window opening fraction factor distribution. Previously the mean $\mu$ and standard deviation $\sigma$ are adopted based on $x$, however, the `scipy.stats.lognorm` module takes $\mu$ and $\sigma$ based on $ln(x)$. This has been corrected;

### 04/08/2018 VERSION: 0.0.1

- Renamed the packaged from `sfepy` to `sfeprapy` (Structural Fire Engineering Probabilistic Risk Assessment Python);
- Github repository created;
- Updated progress bar appearance in `sfeprapy.time_equivalence.main_args()`;
- Implemented new window opening fraction distribution `window_open_fraction`, linear distribution is now replaced by inverse truncated log normal distribution;
- Updated plot appearance; and
- Project now can be installed through `pip install sfeprapy`.

### 02/01/2018 VERSION: 0.0.0
- Implemented Latin hypercube sampling function, `pyDOE` external library is no longer required;
- Boundary for `q_fd`, defined as `q_fd_ubound` and `q_fd_lbound` (upper and lower limit);
- Now output plot for peak steel temperature according to input 'protection_thickness';
- Inputs arguments are packed in a pandas `DataFrame` object instead of a list;
- Automatically generate fires inline with selected percentile `select_fires_teq` ±tolerance `select_fires_teq_tol` and save as .png and .csv.
