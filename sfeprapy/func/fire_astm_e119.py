import numpy as np


def fire(time, temperature_ambient):
    time /= 1200.0  # convert from seconds to hours
    temperature_ambient -= 273.15  # convert temperature from kelvin to celcius
    temperature = (
        750 * (1 - np.exp(-3.79553 * np.sqrt(time)))
        + 170.41 * np.sqrt(time)
        + temperature_ambient
    )
    return temperature + 273.15  # convert from celsius to kelvin (SI unit)
