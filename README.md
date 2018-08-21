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

#### Installation using `pip install`

`sfeprapy` is available on PyPI and therefore can be installed by using `pip install`. Please ensure you have `pip` packaged available. It is recommended to install Anaconda Python distribution as it packs few very useful libraries, including `pip`. Type the following command in a command-line terminal will install let you `sfeprapy`.
```python
pip install sfeprapy
```

#### Local installation
For any reasons you do not want to install `sfeprapy` as a Python library, download the entire project folder to your computer, change current working directory to the project folder then install `sfeprapy` by using pip:

```
pip install .
```

The installation process should be able to install all required libraries listed in Prerequisites.

### Usage

In Python, enter the following code will run the time equivalence (Monte Carlo) analysis which will ask for input files directory.
```python
from sfeprapy import time_equivalence as app
app.run()
```

A dialog box will pop out for you to select folders containing problem definition files. The dialog box will only let you select one folder at a time, but you can select more folders after you click **Select Folder** button. Click **Cancel** to finish selecting folders and the time equivalence analysis will proceed. The program will be paused when all problem definition files are complete. Example console output is shown below.

```
Input file:              test1
Total simulations:       500
Number of threads:       4
######################## 74.8s
Input file:              test2
Total simulations:       500
Number of threads:       4
######################## 83.9s
Press Enter to let me disappear
```

`sfeprapy.time_equivalence.run()` will identify all input files contained in the folder directory and run them one by one until all input files are complete. Simulation output files will be saved under the input file directory.

## Authors

* **Yan Fu** - *fuyans@gmail.com*
* **Danny Hopkin** - *danny.hopkin@olssonfire.com*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
