import numpy as np


def fire(time, temperature_initial):
    time /= 1200.0  # convert time from seconds to hours
    temperature_initial -= 273.15  # convert ambient temperature from kelvin to celsius
    temperature = (
        660 * (1 - 0.687 * np.exp(-0.32 * time) - 0.313 * np.exp(-3.8 * time))
        + temperature_initial
    )
    return temperature + 273.15  # convert temperature from celsius to kelvin
