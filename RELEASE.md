## RELEASE NOTES

### KNOWN ISSUES AND TASKS LIST

- [ ] Verification to travelling fire.

- [ ] Verification to Eurocode protected steel heat transfer.

- [ ] Verification to time_equivalence calculated intermediate variables. i.e. opening_factor.

- [ ] Verification to distribution functions.

- [ ] Make verification procedures.

- [ ] Build-in Eurocode time equivalence hand calculation.

- [ ] Work out the most onerous beam location, which would give the worst time equivalence result.

- [ ] Add PD 6688 time equivalence calculation.


### VERSIONS


**31/03/2019 VERSION: 0.4**

- New: `sfeprapy.mc` figure size is customisable in config.json file.
- New: `sfeprapy.mc` figure x-axis limits (i.e. t_eq) is customisable in config.json file.
- Fixed: `sfeprapy.mc` result $t_{eq}$ 'stepping' issue due to tolerance of the solver is now fixed by replacing the bisection solver with secant solver.
- Improved: `sfeprapy.mc` input parameter (in master .csv file) `fire_mode` is added, replacing `fire_type_enforced`.
- Improved: `sfeprapy.mc` simulation time improved. Tested case shows 256 seconds reduced to 26 seconds for 1000 simulations on i7-3770k with 6 threads.
- Improved: `sfeprapy.mc` refreshed data flow - master .csv input file -> individual case *.json file -> individual case MC *_in.csv file -> individual case result *_out.csv file -> *.png files.

**11/03/2019 VERSION: 0.3**

- Improved: Updated the decision making for choosing between travelling fire and parametric fire, introduced q_td limit criteria, i.e. 50 <= q_td <= 1000.

**11/03/2019 VERSION: 0.2**

- SfePrapy is now in beta testing phase;
- New: Eurocode DIN Annex parametric fire curve is now available, force simulations to use only this fire curve by set `fire_type_enforced` to 2 (see new input file template); and
- New: Feature to find best fit distribution function - fire this tool by `python -m sfeprapy.dist_fit`.

**07/03/2019 VERSION: 0.0.8**

- Improved: Capable of combine individual time equivalence curves into one with corresponding `probability_weight` (see new input file template).

**17/02/2019 VERSION: 0.0.7**

- New: Additional figure named `t_eq_merged.png` is produced to show merged time equivalency of all cases.

**22/01/2019 VERSION: 0.0.6**

- Fixed: Final time equivalence plot legend text is set to case name. Previously '_out' was included.
- Improved: Final time equivalence plot will be produced without running any simulation, e.g. all `is_live` is set to 0.

**01/01/2019 VERSION: 0.0.5**

- New: More configuration parameters can be defined in 'config.json' file located in the same folder as the selected input file;
- New: Optional MC simulation calculation for each case (as in the input *.csv file);
- New: beam_loc_z (beam element height), room_opening_permanent_ratio (permanent free window opening area);
- Improved: Plot background transparent.

**31/10/2018 VERSION: 0.0.4**

- Additional returned results from the Monte Carlo simulation tool `sfeprapy.time_equivalence_core.calc_time_equivalence`. Window opening factor `opening_factor` is added to be returned from the function.
- `sfeprapy.time_equivalence.app` is now able to run a single simulation. When 'simulations=1' is defined, all distributed variables are disabled and mean or mean(upper, lower) is used for stochastic parameters.
- New testing input file 'benchmark_file_1' is added for single simulation testing, all other parameters are identical to 'benchmark_file_0'. Benchmark files are moved to validation folder, contained in root directory.

**21/08/2018 VERSION: 0.0.3**

- Updated code relating simulation output result \*.p and \*res.csv files. This is to fix an issue which output fires do not align with input / output index numbering. The new \*.p and \*res.csv files are sorted by time equivalence. The new output files are significantly larger than previous versions due to more variables are being passed in and out from the main calculation function `sfeprapy.time_equivalence_core.calc_time_equivalence()`.
- Fire duration `fire_duration` is checked and reassigned if necessary so that the slowest travelling fire is able to travel the entire room `room_depth`. `fire_duration` defined in the input file will be the minimum fire duration.
- Verification procedures are added for part of the project, including parametric fire testing, travelling fire testing and Eurocode protected steel heat transfer.

**15/08/2018 VERSION: 0.0.2**

- Graphical folder select dialog is available;
- Fixed an issue associated with `sfeprapy.time_equivalence.run()` where would not ask for new input folder directory when run more than once without re-import the module;
- Fixed window opening fraction factor distribution. Previously the mean $\mu$ and standard deviation $\sigma$ are adopted based on $x$, however, the `scipy.stats.lognorm` module takes $\mu$ and $\sigma$ based on $ln(x)$. This has been corrected;

**04/08/2018 VERSION: 0.0.1**

- Renamed the packaged from `sfepy` to `sfeprapy` (Structural Fire Engineering Probabilistic Risk Assessment Python);
- Github repository created;
- Updated progress bar appearance in `sfeprapy.time_equivalence.run()`;
- Implemented new window opening fraction distribution `window_open_fraction`, linear distribution is now replaced by inverse truncated log normal distribution;
- Updated plot appearance; and
- Project now can be installed through `pip install sfeprapy`.

**02/01/2018 VERSION: 0.0.0**
- Implemented Latin hypercube sampling function, `pyDOE` external library is no longer required;
- Boundary for `q_fd`, defined as `q_fd_ubound` and `q_fd_lbound` (upper and lower limit);
- Now output plot for peak steel temperature according to input 'protection_thickness';
- Inputs arguments are packed in a pandas `DataFrame` object instead of a list;
- Automatically generate fires inline with selected percentile `select_fires_teq` ±tolerance `select_fires_teq_tol` and save as .png and .csv.
