# sfeprapy

Introduction is on the way.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

```
python>=3.6.0
matplotlib==2.2.2
numpy==1.15.0
pandas==0.23.3
scipy==1.1.0
seaborn==0.9.0
```

### Installing

Download the entire project folder to your computer, change current working directory to the project folder then install PraPy by using pip:

```
pip install .
```

The installation process should be able to install all required libraries listed in Prerequisites.

### Example

In Python, enter the following code will run the time equivalence (Monte Carlo) analysis which will ask for input files directory.
```python
from prapy import time_equivalence as app
app.run()
```

Copy your inputs file folder and paste into the terminal window, where the directory `D:\\test` should be a folder containing all input files ending with `.txt` which all input files will be run. Input file template can be found at the end of this section. 

```shell
Work directory: C:\test
```

After the correct input files folder being provided, time equivalence analysis will proceed as below.

```shell
Input file:              test1
Total simulations:       500
Number of threads:       4
######################## 100% (74.8)
Input file:              test2
Total simulations:       500
Number of threads:       4
######################## 100% (83.9)
>>>
```

Simulation output files will be saved under the input file directory.

## Authors

* **Ian Fu** - *fuyans@gmail.com*
* **Danny Hopkin** - *danny.hopkin@olssonfire.com*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Notes

**KNOWN ISSUES**
- [ ] Test with travelling fire curve `travelling_fire` in `sfeprapy.func.temperature_fires`
- [ ] Graphical user interface, with multiprocessing capability
- [ ] Make validation procedures
- [ ] Make verification procedures
- [ ] Publish on PyPI

**04/08/2018 VERSION: 0.0.1 (alpha)**
- Renamed the packaged from `sfepy` to `sfeprapy` (Structural Fire Engineering Probabilistic Risk Assessment Python);
- Github repository created;
- Updated progress bar appearance in `sfeprapy.time_equivalence.run()`;
- Implemented new window opening fraction distribution `window_open_fraction`, linear distribution is now replaced by inverse truncated log normal distribution;
- Updated plot appearance; and
- Code published on PyPI.

**02/01/2018 VERSION: 0.0.0 (pre-alpha)**
- Implemented Latin hypercube sampling function, `pyDOE` external library is no longer required;
- Boundary for `q_fd`, defined as `q_fd_ubound` and `q_fd_lbound` (upper and lower limit). DEPRECIATED;
- Now output plot for peak steel temperature according to input 'protection_thickness';
- Inputs arguments are packed in a pandas `DataFrame` object instead of a list;
- Automatically generate fires inline with selected percentile `select_fires_teq` ±tolerance `select_fires_teq_tol` and save as .png and .csv.
