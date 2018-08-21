## Notes

**KNOWN ISSUES**
- [ ] Test with travelling fire curve `travelling_fire` in `sfeprapy.func.temperature_fires`.
- [ ] Graphical user interface, with multiprocessing capability.
- [ ] Make validation procedures.
- [ ] Make verification procedures.
- [ ] Upgrade `sfeprapy.time_equivalence_core.mc_inputs_generator2_core` with `scipy.stats.norm.ppf` and `scipy.stats.gumbel_r.ppf` to generate Monte Carlo input samples.
- [ ] Add array input variable feature which input files will be automatically populated based on variables arrays.
- [ ] Generate report about generated random variables, i.e. to calculate miu and sigma and compare with input values.
- [ ] Some steel temperature seeking cases unsuccessful, cause unknown. Currently these items are filtered out for time equivalence calculation.
- [ ] Verification function of all distribution functions.
- [ ] Eurocode time equivalence hand calculation.

**21/08/2018 VERSION: 0.0.3**
- Updated code relating simulation output result \*.p and \*res.csv files. This is to fix an issue which output fires do not align with input / output index numbering. The new \*.p and \*res.csv files are sorted by time equivalence. The new ouput files are significantly larger than previous versions due to more variables are being passed in and out from the main calculation function `sfeprapy.time_equivalence_core.calc_time_equivalence()`.
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
