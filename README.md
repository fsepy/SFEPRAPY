# SFEPraPy

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

### Installation

#### Installation through `pip`

The package can be installed through `pip` by the following command.
```python
pip install sfeprapy
```

#### Alternative local installation
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

Copy your inputs file folder directory and paste into the terminal window, where the directory `D:\\test` should be a folder containing all input files ending with `.txt` which all input files will be run. Input file template can be found at `problem_definition` folder. 

```shell
Work directory: C:\test
```

After correct folder directory being provided, time equivalence analysis will proceed as below.

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
The code will identify all input files contained in the folder directory provided and run them one by one until all input files are complete. Simulation output files will be saved under the input file directory.

## Authors

* **Ian Fu** - *fuyans@gmail.com*
* **Danny Hopkin** - *danny.hopkin@olssonfire.com*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
