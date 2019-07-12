# SfePrapy

A probabilistic analysis tool that estimates the structural reliability for a given scenario (such as enclosure geometry, building type, window areas etc.) against equivalent time exposure to the ISO 834 fire curve.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Python 3.6.0 or later is required.

### Installation

#### pip

pip is a package management system for installing and updating Python packages. pip comes with Python, so you get pip simply by installing Python. On Ubuntu and Fedora Linux, you can simply use your system package manager to install the `python3-pip` package. [_The Hitchhiker's Guide to Python_ provides some guidance on how to install Python on your system if it isn't already](https://docs.python-guide.org/starting/installation/); you can also [install Python directly from python.org](https://www.python.org/getit/). You might want to [upgrade pip](https://pip.pypa.io/en/stable/installing/) before using it to install other programs.

SfePrapy uses Python, and as of July 2018 SfePrapy only works in Python3. 

1.	If you are using Windows with Python version 3.3 or higher, use the [Python Launcher for Windows](https://docs.python.org/3/using/windows.html?highlight=shebang#python-launcher-for-windows) to use `pip` with Python version 3:
    ```bash
    py -3 -m pip install SfePrapy
    ```
2.	If your system has a `python3` command (standard on Unix-like systems), install with:
    ```bash
    python3 -m pip install SfePrapy
    ```
3.	You can also just use the `python` command directly, but this will use the _current_ version of Python in your environment:
    ```bash
    python -m pip install SfePrapy
    ```

### Usage

#### `sfeprapy.mc` Monte Carlo Simulation for Structural PRA

There are a number of components packed within SfePrapy and the most important tool is the PRA Monte Carlo Simulation for Structural PRA. Enter the following code in your terminal window to fire up the tool.

```sh
python -m SfePrapy.mc
```

Alternatively, the following code will To main_args SfePrapy Monte Carlo simulation routine in Python:

```python
>> from sfeprapy.mc.__main__ import main_args
>> main_args()
```

Then the tool will ask to select an input file (in *.csv format, definition see following sections) and simulation starts after selecting an input file. Simulation progress will be displayed as below.

```
CASE: CASE A
NO. OF THREADS: 7
NO. OF SIMULATION: 1000
████████████████████████ 151.3s 
```

The app will identify all input files contained in the folder directory and main_args them one by one until all input files are complete. Simulation output files will be saved under the input file directory.

##### Input Parameters

Example input files can be found in example_input.

The input parameters have been organised on this sheet and the units provided. The key parameters to adjust for each modelling case have been highlighted red. Other parameters will need to be adjusted for specific cases. Completing this table produces the output in the layout required for the csv input file on the sheet "Input for SfePrapy". This can then be copied into the csv file used as the input for SfePrapy - see misc folder for template. The third sheet of this document provides an example input file which if saved as a csv holds the data for three cases but would only main_args the first two (see "is_live" variable).

| PARAMETER                           | DESCRIPTION                                                  |
| ----------------------------------- | ------------------------------------------------------------ |
| **Model Settings**                  |                                                              |
| is_live                             | This is the trigger to tell SfePrapy to main_args the model for this set of input values. In the input CSV there can be multiple cases |
| fire_type_enforced                  | Integer.<br />0 - EC parametric fire only;<br />1 - Travelling fire only;<br />2 - EC parametric fire, German Annex;<br />3 - Option 0 and 1 as above; and<br />4 - Option 2 and 2 as above. |
| simulations                         | Integer. The number of simulations that will   be main_args. A sensitivity analysis should be carried out to determine the   appropriate number of simulations |
|                                     |                                                              |
| **Compartment Geometry**            |                                                              |
| room_breadth                        | Float, in [m]. Breadth of room                             |
| room_depth                          | Float, in [m]. Depth of room                               |
| room_height                         | Float, in [m]. Height of room                              |
| room_window_width                   | Float, in [m]. Total width of windows for   compartment    |
| room_window_height                  | Float, in [m]. Height of window (top of window to   bottom of window, not height from floor) |
| beam_loc_z                          | Float, in [m]. Height of test beam within the   compartment. This can be altered to assess the influence of height in tall   compartments. Need to assess worst case height for columns. |
| beam_loc_ratio_lbound               | Float. Minimum beam location relative to   compartment length for TFM - Linear distribution |
| beam_loc_ratio_ubound               | Float. Maximum beam location relative to   compartment length for TFM - Linear distribution |
| **Windows/Natural Vent**            |                                                              |
| room_opening_fraction_std           | Float. Glazing fall-out fraction - 1-lognorm   - standard dev. |
| room_opening_fraction_mean          | Float. Glazing fall-out fraction - 1-lognorm   - mean        |
| room_opening_fraction_ubound        | Float. Glazing fall-out fraction - 1-lognorm   - upper limit |
| room_opening_fraction_lbound        | Float. Glazing fall-out fraction - 1-lognorm   - lower limit |
| room_opening_permanent_fraction     | Float. Use this to force a ratio of open   windows. If there is a vent to the outside this can be included here. |
| **Fire inputs**                     |                                                              |
| fire_tlim                           | Float, in [hour]<br/>Time for maximum gas temperature in case of fuel controlled fire. <br />Slow: 25/60 <br />Medium: 20/60 <br />Fast: 15/60<br />(Annex A EN 1991-1-2) |
| time_step                           | Float, in [s]. Time step used for the model.   Suggested time step 30 s to balance results accuracy and computation time. |
| time_duration                       | Float, in [s]. End of simulation. This should be set so that output data is produced allowing the target reliability to be determined. Not always necessary to main_args for extended periods. Normally set it to 4 hours and longer period of time for greater room length in order for travelling fire to propagate the entire room |
| room_wall_thermal_inertia           | Float, in [J/m²/K/√s]. Compartment thermal inertia |
| fire_qfd_mean                       | Float, in [MJ/m²]. Fire load density - Gumbel distribution - mean |
| fire_qfd_std                        | Float, in [MJ/m²]. Fire load density - Gumbel distribution - standard dev |
| fire_qfd_lbound                     | Fire load density - Gumbel   distribution - lower limit      |
| fire_qfd_ubound                     | Float, in [MJ/m²]. Fire load density - Gumbel   distribution - upper limit |
| fire_hrr_density                    | Float, in [MW/m²]. Heat release rate. This should be selected based on the fuel. See literature for typical values for different occupancies. |
| fire_spread_lbound                  | Float, in [MJ/m²]. Min spread rate for TFM - Linear distribution |
| fire_spread_ubound                  | Float, in [m/s]. Max spread rate for TFM - Linear distribution |
| fire_nft_mean                       | Float, in [°C]. TFM near field temperature - Norm   distribution - mean |
| fire_com_eff_lbound                 | Min combustion efficiency - Linear distribution              |
| fire_com_eff_ubound                 | Max combustion efficiency - Linear distribution              |
| **Section Properties**              |                                                              |
| beam_cross_section_area             | Float, in [m²]. Cross sectional area of beam               |
| beam_rho                            | Float, in [kg/m³]. Density of beam              |
| beam_temperature_goal               | Float, in [K]. Beam (steel) failure temperature in   kelvin for goal seek |
| beam_protection_protected_perimeter | Float, in [m]. Heated perimeter                            |
| beam_protection_thickness           | Float, in [m]. Thickness of protection                     |
| beam_protection_k                   | Float, in [W/m/K]. Protection conductivity    |
| beam_protection_rho                 | Float, in [kg/m³]. Density of protection to beam |
| beam_protection_c                   | Float, in [J/kg/K]. Specific heat of protection |

A configuration file (.json) is also required to main_args the simulation

`config.json` contains configurations for `SfePrapy.mc`.

```json
{
  "n_proc": 7,
}
```

Activate Python virtual environment and main_args SfePrapy.

```bash
python -m SfePrapy.mc
```

##Limitations of SfePrapy

###Eurocode Parametric Fire

#### Maximum floor geometry

500 m^2

Travelling Fire

#### Beam/Column temperature measurement location

this is conservatively the same as ceiling height of the compartment. 

## Authors

* **Yan Fu** - *fuyans@gmail.com*
* **Danny Hopkin** - *danny.hopkin@ofrconsultants.com*
* **Ieuan Rickard** - *ieuan.rickard@ofrconsultants.com*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
