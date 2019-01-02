# SfePraPy

A probabilistic analysis tool that estimates the structural reliability for a given scenario (such as enclosure geometry, building type, window areas etc.) against equivalent time exposure to the ISO 834 fire curve.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Python 3.6.0 or later is required.

### Installation

#### pip

pip is a package management system for installing and updating Python packages. pip comes with Python, so you get pip simply by installing Python. On Ubuntu and Fedora Linux, you can simply use your system package manager to install the `python3-pip` package. [_The Hitchhiker's Guide to Python_ provides some guidance on how to install Python on your system if it isn't already](https://docs.python-guide.org/starting/installation/); you can also [install Python directly from python.org](https://www.python.org/getit/). You might want to [upgrade pip](https://pip.pypa.io/en/stable/installing/) before using it to install other programs.

SfePraPy uses Python, and as of July 2018 SfePraPy only works in Python3. 

1.	If you are using Windows with Python version 3.3 or higher, use the [Python Launcher for Windows](https://docs.python.org/3/using/windows.html?highlight=shebang#python-launcher-for-windows) to use `pip` with Python version 3:
    ```bash
    py -3 -m pip install sfeprapy
    ```
2.	If your system has a `python3` command (standard on Unix-like systems), install with:
    ```bash
    python3 -m pip install sfeprapy
    ```
3.	You can also just use the `python` command directly, but this will use the _current_ version of Python in your environment:
    ```bash
    python -m pip install sfeprapy
    ```

#### Local
For any reasons you do not want to install `sfeprapy` as a Python library, download the entire project folder to your computer (e.g. Desktop), change current working directory to the project folder then install `sfeprapy` by using pip:

```bash
pip install .
```

The installation process should be able to install all required libraries listed in Prerequisites.

### Usage

To run SfePraPy Monte Carlo simulation app in a terminal window:

```sh
sfeprapymc
```

To run SfePraPy Monte Carlo simulation in python:

```python
from sfeprapy import time_equivalence as app
app.run()
```

A dialog box will pop out for you to select folders containing problem definition files. The dialog box will only let you select one folder at a time, but you can select more folders after you click **Select Folder** button. Click **Cancel** to finish selecting folders and the time equivalence analysis will proceed. The program will be paused when all problem definition files are complete. Example console output is shown below.

```
Work directory:          C:/Users/ian/Desktop/sfeprapy_test
Input file:              time_equivalence_version_0.0.3_1.txt
Total simulations:       1000
Number of threads:       8
████████████████████████ 115.1s
Press Enter to finish
```

The app will identify all input files contained in the folder directory and run them one by one until all input files are complete. Simulation output files will be saved under the input file directory.

## Authors

* **Yan Fu** - *fuyans@gmail.com*
* **Danny Hopkin** - *danny.hopkin@olssonfire.com*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
