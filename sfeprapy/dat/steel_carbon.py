# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from io import StringIO
import warnings


def steel_specific_heat_carbon_steel(temperature):
    """
    :param temperature: {float} [K]
    :return: {float} [J/kg/K]
    """
    temperature -= 273.15
    if 20 <= temperature < 600:
        return (
            425
            + 0.773 * temperature
            - 1.69e-3 * np.power(temperature, 2)
            + 2.22e-6 * np.power(temperature, 3)
        )
    elif 600 <= temperature < 735:
        return 666 + 13002 / (738 - temperature)
    elif 735 <= temperature < 900:
        return 545 + 17820 / (temperature - 731)
    elif 900 <= temperature <= 1200:
        return 650
    else:
        return 0


if __name__ == "__main__":

    import timeit

    test_number = 100000

    def aa():
        steel_specific_heat_carbon_steel(np.random.rand() * 1180 + 20 + 273.15)

    print(timeit.timeit(aa, number=test_number))
    print(timeit.timeit(bb, number=test_number))
