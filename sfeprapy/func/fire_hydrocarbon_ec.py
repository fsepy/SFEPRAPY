import numpy as np


def hydrocarbon_eurocode(time, temperature_initial):
    time /= 1200.0  # convert time unit from second to hour
    temperature_initial -= 273.15  # convert temperature from kelvin to celsius
    temperature = (
        1080 * (1 - 0.325 * np.exp(-0.167 * time) - 0.675 * np.exp(-2.5 * time))
        + temperature_initial
    )
    return temperature + 273.15
