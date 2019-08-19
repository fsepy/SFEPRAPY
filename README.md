# SfePrapy

A probabilistic analysis tool that estimates the structural reliability for a given scenario (such as enclosure geometry, building type, window areas etc.) against equivalent time exposure to the ISO 834 fire curve.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Python 3.7 or later is required.

### Installation

#### pip

pip is a package management system for installing and updating Python packages. pip comes with Python, so you get pip simply by installing Python. On Ubuntu and Fedora Linux, you can simply use your system package manager to install the `python3-pip` package. [The Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/installation/) provides some guidance on how to install Python on your system if it isn't already; you can also install Python directly from [python.org](https://www.python.org/getit/). You might want to [upgrade pip](https://pip.pypa.io/en/stable/installing/) before using it to install other programs.

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

#### MCS for structural PRA method 0: `sfeprapy.mcs0`

Introduction of `sfeprapy.mcs0` is current NOT covered here but will provide notebook tutorials in near future.

To run `sfeprapy.mcs0` from source (or when it is installed via, i.e. pip):

```sh
python -m SfePrapy.mcs0
```

A window will be popped up asking for input a input / problem definition file. The input file should be in '.csv' or '.xlsx' format. Structure of the input file is addressed in the following paragraphs.

When a configuration file with the name 'config.json' is provided in the same directory, the program will attempt to read it and use parameters contained therein. Number of threads (multi-processing) along with few other parameters can be defined in this file.

Once the input file is selected, the program will take the lead and run calculations until all simulations are complete. Results will be saved in the same folder. Conveniently, the software has the feature of displaying the progress and some statistics as per below.

```
CASE                    : Standard Case 1
NO. OF THREADS          : 4
NO. OF SIMULATIONS      : 1000
100%|███████████████████| 1000/1000 [00:09<00:00, 86.15it/s]
fire_type               : {0: 23, 1: 977}
beam_position_horizontal: -1.000    27.534    28.646   
fire_combustion_efficien: 0.800     0.900     1.000    
fire_hrr_density        : 0.240     0.250     0.260    
fire_load_density       : 10.000    420.399   1500.000 
fire_nft_limit          : 623.150   1323.150  2023.152 
fire_spread_speed       : 0.004     0.011     0.019     

CASE                    : Standard Case 2
NO. OF THREADS          : 4
NO. OF SIMULATIONS      : 1000
100%|███████████████████| 1000/1000 [00:10<00:00, 77.85it/s]
fire_type               : {0: 10, 1: 990}
beam_position_horizontal: -1.000    27.943    28.646   
fire_combustion_efficien: 0.800     0.900     1.000    
fire_hrr_density        : 0.240     0.250     0.260    
fire_load_density       : 10.000    420.399   1500.000 
fire_nft_limit          : 623.150   1323.150  2023.152 
fire_spread_speed       : 0.004     0.011     0.019   
```

##### Input Parameters

Example input template can be found at:

-  `sfeprapy.mcs0.EXAMPLE_CONFIG_DICT`: Example configuration file, dict object;
-  `sfeprapy.mcs0.EXAMPLE_INPUT_DICT`: Example input file, dict object; and
-  `sfeprapy.mcs0.EXAMPLE_INPUT_CSV`: Example input file, str in csv format.

The following table summerises the parameters that are required by `sfeprapy.mcs0` module.

| PARAMETERS                       | DESCRIPTION                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| **Model Settings**               |                                                              |
| `case_name`                      | Str. An unique name indicating a case. This maybe used in post-processing when combining time equivalence results. |
| `fire_mode`                      | Integer. To define what design fires to use:<br />0 - EC parametric fire only;<br />1 - Travelling fire only;<br />2 - EC parametric fire, German Annex;<br />3 - Option 0 and 1 as above; and<br />4 - Option 2 and 2 as above. |
| `n_simulations`                  | Integer. The number of simulations that will be running. A sensitivity analysis should be carried out to determine the   appropriate number of simulations. |
| `probability_weight`             | Float. The fire occurance probability weight of this specific case (i.e. compartment) among all cases (i.e. entire building).<br />`sfeprapy.mcs0` does not use this parameter for any calculation, it is only used during post-processing phrase. |
| **Compartment Geometry**         |                                                              |
| `room_breadth`                   | Float, in [m]. Breadth of room (greater dimension).          |
| `room_depth`                     | Float, in [m]. Depth of room (shorter dimension).            |
| `room_height`                    | Float, in [m]. Height of room (floor slab to ceiling slab).  |
| `window_width`                   | Float, in [m]. Total width of all opening areas for a compartment. |
| `window_height`                  | Float, in [m]. Weighted height of all opening areas.         |
| `beam_position_vertical`         | Float, in [m]. Height of test structure element within the compartment for TFM. This can be altered to assess the influence of height in tall compartments. Need to assess worst case height for columns. |
| `beam_position_horizontal`       | Float, in [m]. Minimum beam location relative to compartment length for TFM - Linear distribution |
| **Windows/Natural Vent**         |                                                              |
| `window_open_fraction`           | Float. Glazing fall-out fraction.                            |
| `window_open_fraction_permanent` | Float. Use this to force a ratio of open   windows. If there is a vent to the outside this can be included here. |
| **Fire inputs**                  |                                                              |
| `fire_tlim`                      | Float, in [hour]<br/>Time for maximum gas temperature in case of fuel controlled fire. <br />Slow: 25/60 <br />Medium: 20/60 <br />Fast: 15/60<br />(Annex A EN 1991-1-2) |
| `fire_time_step`                 | Float, in [s]. Time step used for the model, all fire time-temperature curves and heat transfer calculation. This is recommended to be less than 30 s. |
| `fire_time_duration`             | Float, in [s]. End of simulation. This should be set so that output data is produced allowing the target reliability to be determined. Not always necessary to main_args for extended periods. Normally set it to 4 hours and longer period of time for greater room length in order for travelling fire to propagate the entire room. |
| `room_wall_thermal_inertia`      | Float, in [J/m²/K/√s]. Compartment lining thermal inertia.   |
| `fire_load_density`              | Float, in [MJ/m²]. Fire load density. This should be selected based on occupancy charateristics. See literature for typical values for different occupancies. |
| `fire_hrr_density`               | Float, in [MW/m²]. Heat release rate. This should be selected based on the fuel. See literature for typical values for different occupancies. |
| `fire_spread_speed`              | Float, in [MJ/m²]. Min spread rate for TFM.                  |
| `fire_nft_limit`                 | Float, in [°C]. TFM near field temperature.                  |
| `fire_combustion_efficiency`     | Float, in [-]. Combustion efficiency.                        |
| `fire_gamma_fi_q`                | Float, in [-]. The partial factor for EC fire (German Annex). |
| `fire_t_alpha`                   | Float, in [s]. The fire growth factor.                       |
|                                  |                                                              |
| **Section Properties**           |                                                              |
| `beam_cross_section_area`        | Float, in [m²]. Cross sectional area of beam                 |
| `beam_rho`                       | Float, in [kg/m³]. Density of beam                           |
| `beam_temperature_goal`          | Float, in [K]. Beam (steel) failure temperature in   kelvin for goal seek |
| `protection_protected_perimeter` | Float, in [m]. Heated perimeter                              |
| beam_protection_thickness        | Float, in [m]. Thickness of protection                       |
| `protection_k`                   | Float, in [W/m/K]. Protection conductivity                   |
| `protection_rho`                 | Float, in [kg/m³]. Density of protection to beam             |
| `protection_c`                   | Float, in [J/kg/K]. Specific heat of protection              |
| **Solver Settings**              |                                                              |
| `solver_temperature_goal`        | Float, in [K]. The temperature to be solved for. This is critical temperature of the beam structural element, i.e. 550 or 620 °C. |
| `solver_max_iter`                | Float. The maximum iteration for the solver to find convergence. Suggest 20 as most (if not all) cases converge in less than 20 iterations. |
| `solver_thickness_lbound`        | Float. The smallest value that the solved protection thickness can be. |
| `solver_thickness_ubound`        | Float. The greatest value that the solved protection thickness can be. |
| `solver_tol`                     | Float, in [K]. Tolerance of the temperature to be solved for. Set to 1 means convergence will be satisfied when the solved value is within `solver_temperature_goal`-1 and `solver_temperature_goal`+1. |

##Limitations

Todo

## Authors

* **Yan Fu** - *fuyans@gmail.com*
* **Danny Hopkin** - *danny.hopkin@ofrconsultants.com*
* **Ieuan Rickard** - *ieuan.rickard@ofrconsultants.com*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
