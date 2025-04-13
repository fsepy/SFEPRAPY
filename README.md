# SFEPRAPY
[![GitHub version](https://badge.fury.io/gh/fsepy%2Fsfeprapy.svg)](https://github.com/fsepy/SfePrapy)

Structural fire engineering (SFE) probabilistic reliability assessment (PRA) Python (PY) is a probabilistic analysis
tool. It calculates equivalent of time exposure to ISO 834 standard fire and this can be used to assess the appropriate
fire resistance rating for structural elements using reliability based methods.

`sfeprapy` is under continuous development and actively used in research and real engineering design problems.

Legacy wiki can be found [here](https://github.com/fsepy/SfePrapy/wiki).

Documentation (WIP) can be found [here](https://sfeprapy-doc.readthedocs.io/en/latest/index.html)

A publication summarising the capabilities can be found [here](https://www.researchgate.net/publication/333202825_APPLICATION_OF_PYTHON_PROGRAMMING_LANGUAGE_IN_STRUCTURAL_FIRE_ENGINEERING_-_MONTE_CARLO_SIMULATION).

## Getting Started

### Installation

You can download and install the package locally or use `pip` install from GitHub (
requires [git](https://git-scm.com/downloads)):

 ```sh
 pip install --upgrade "git+https://github.com/fsepy/SFEPRAPY.git@master"
 ```


### Command line interface

`sfeprapy` command line interface (CLI) uses the current working directory to obtain and/or save files.

#### To get help

```sh
sfeprapy -h
```

#### To produce a `sfeprapy.mcs0` example input file

```sh
sfeprapy mcs0 template example_input.csv
```

#### To run `sfeprapy.mcs0` simulation

```sh
sfeprapy mcs0 -p 4 example_input.csv
```

`sfeprapy.mcs0` uses the [multiprocessing](https://docs.python.org/3.4/library/multiprocessing.html#module-multiprocessing) library to utilise full potential performance of multi-core CPUs. The `-p 4` defines 4 threads will be used in running the simulation, 1 is the default value.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
