import numpy as np


def fire(
        time,
        temperature_initial
):

    # INPUTS CHECK
    time = np.array(time, dtype=float)
    time[time < 0] = np.nan

    # SI UNITS -> EQUATION UNITS
    temperature_initial -= 273.15  # [K] -> [C]
    time /= 60.  # [s] - [min]

    # CALCULATE TEMPERATURE BASED ON GIVEN TIME
    temperature = 345. * np.log10(time * 8. + 1.) + temperature_initial
    temperature[temperature == np.nan] = temperature_initial

    # EQUATION UNITS -> SI UNITS
    time *= 60.  # [min] -> [s]
    temperature += 273.15  # [C] -> [K]

    return temperature
