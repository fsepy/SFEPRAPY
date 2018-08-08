## Notes

**KNOWN ISSUES**
- [ ] Test with travelling fire curve `travelling_fire` in `sfeprapy.func.temperature_fires`
- [ ] Graphical user interface, with multiprocessing capability
- [ ] Make validation procedures
- [ ] Make verification procedures
- [ ] Publish on PyPI

**04/08/2018 VERSION: 0.0.1**
- Renamed the packaged from `sfepy` to `sfeprapy` (Structural Fire Engineering Probabilistic Risk Assessment Python);
- Github repository created;
- Updated progress bar appearance in `sfeprapy.time_equivalence.run()`;
- Implemented new window opening fraction distribution `window_open_fraction`, linear distribution is now replaced by inverse truncated log normal distribution;
- Updated plot appearance; and
- Project now can be installed through `pip install sfeprapy`.

**02/01/2018 VERSION: 0.0.0**
- Implemented Latin hypercube sampling function, `pyDOE` external library is no longer required;
- Boundary for `q_fd`, defined as `q_fd_ubound` and `q_fd_lbound` (upper and lower limit). DEPRECIATED;
- Now output plot for peak steel temperature according to input 'protection_thickness';
- Inputs arguments are packed in a pandas `DataFrame` object instead of a list;
- Automatically generate fires inline with selected percentile `select_fires_teq` ±tolerance `select_fires_teq_tol` and save as .png and .csv.
