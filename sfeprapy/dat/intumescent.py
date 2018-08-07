# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os


def thermal(property_name):
    dir_package = os.path.dirname(os.path.abspath(__file__))

    dir_files = {
        "thermal conductivity (thomas2002)":
            "k_1_T_intumescent_thomas2002.csv"
    }

    # read file
    data = pd.read_csv("/".join([dir_package, dir_files[property_name]]), delimiter=",", dtype=float)
    x, y = data.values[:, 0], data.values[:, 1]

    return interp1d(x, y)


# if __name__ == "__main__":
#     temperature = np.arange(20.,1200.+0.5,0.5) + 273.15
#     prop = temperature * 0
#     for i,v in enumerate(temperature):
#         prop[i] = steel_specific_heat_carbon_steel(v)
#     df = pd.DataFrame(
#         {
#             "Temperature [K]": temperature,
#             "Specific Heat Capacity [J/kg/K]": prop
#         }
#     )
