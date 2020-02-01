# SfePrapy
[![GitHub version](https://badge.fury.io/gh/fsepy%2Fsfeprapy.svg)](https://github.com/fsepy/SfePrapy)
[![Updates](https://pyup.io/repos/github/fsepy/SfePrapy/shield.svg)](https://pyup.io/repos/github/fsepy/SfePrapy/)
[![Build Status](https://img.shields.io/travis/fsepy/SfePrapy.svg?branch=master&label=build%20(master)&style=flat)](https://travis-ci.org/fsepy/SfePrapy)
[![Build Status](https://img.shields.io/travis/fsepy/SfePrapy.svg?branch=dev&label=build%20(dev)&style=flat)](https://travis-ci.org/fsepy/SfePrapy)
[![codecov](https://codecov.io/gh/fsepy/SfePrapy/branch/dev/graph/badge.svg)](https://codecov.io/gh/fsepy/SfePrapy)

Structural fire engineering (Sfe) probabilistic reliability assessment (Pra) Python (py) is a probabilistic analysis tool. It calculates equivalent of time exposure to ISO 834 standard fire and this can be used to assess the appropriate fire resistance rating for structural elements using reliability based methods.

`sfeprapy` is evolving and actively used in research and real engineering design problems.

A publication summarising the capabilities can be found [here](https://www.researchgate.net/publication/333202825_APPLICATION_OF_PYTHON_PROGRAMMING_LANGUAGE_IN_STRUCTURAL_FIRE_ENGINEERING_-_MONTE_CARLO_SIMULATION).

## Getting Started

Documentation can be found in the [wiki](https://github.com/fsepy/SfePrapy/wiki).

### Installation

Chose one of the following installation path.

[Python](https://www.python.org/downloads/) 3.7 or later is required. [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section) is recommended for new starters, it includes Python and few useful packages including a package management tool pip (see below).

[pip](https://pypi.org/) is a package management system for installing and updating Python packages. pip comes with Python, so you get pip simply by installing Python. On Ubuntu and Fedora Linux, you can simply use your system package manager to install the `python3-pip` package. [The Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/installation/) provides some guidance on how to install Python on your system if it isn't already; you can also install Python directly from [python.org](https://www.python.org/getit/). You might want to [upgrade pip](https://pip.pypa.io/en/stable/installing/) before using it to install other programs.

1. to use `pip` install from PyPI:

    [![Downloads](https://pepy.tech/badge/sfeprapy)](https://pepy.tech/project/sfeprapy)

    ```sh
    pip install --upgrade sfeprapy
    ```

2. to use `pip` install from GitHub (requires [git](https://git-scm.com/downloads)):  

    *Note installing `SfePrapy` via this route will include the lastest commits/changes to the library.*  

    ```sh
    pip install --upgrade "git+https://github.com/fsepy/SfePrapy.git@master"
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

#### To produce a figure (once a `sfeprapy.mcs0` simulation is complete)

```sh
sfeprapy mcs0 figure mcs.out.csv
```

## Authors

**Ian Fu** - *ian.fu@ofrconsultants.com*  
**Danny Hopkin** - *danny.hopkin@ofrconsultants.com*  
**Ieuan Rickard** - *ieuan.rickard@ofrconsultants.com*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
