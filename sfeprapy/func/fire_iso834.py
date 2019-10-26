import numpy as np


def fire(time, temperature_initial):

    # INPUTS CHECK
    time = np.array(time, dtype=float)
    time[time < 0] = np.nan

    # SI UNITS -> EQUATION UNITS
    temperature_initial -= 273.15  # [K] -> [C]
    time /= 60.0  # [s] - [min]

    # CALCULATE TEMPERATURE BASED ON GIVEN TIME
    temperature = 345.0 * np.log10(time * 8.0 + 1.0) + temperature_initial
    temperature[temperature == np.nan] = temperature_initial

    # EQUATION UNITS -> SI UNITS
    time *= 60.0  # [min] -> [s]
    temperature += 273.15  # [C] -> [K]

    return temperature


if __name__ == "__main__":
    fire_time = np.arange(0, 2 * 60 * 60 + 1, 1)
    fire_temperature = fire(fire_time, 273.15 + 20)

    print(
        fire_temperature[fire_time == 5 * 60],
        fire_temperature[fire_time == 10 * 60],
        fire_temperature[fire_time == 15 * 60],
        fire_temperature[fire_time == 20 * 60],
        fire_temperature[fire_time == 25 * 60],
    )
