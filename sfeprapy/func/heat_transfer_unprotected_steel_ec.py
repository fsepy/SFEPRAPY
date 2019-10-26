# -*- coding: utf-8 -*-
import numpy as np


def unprotected_steel_eurocode(
    time,
    temperature_ambient,
    perimeter_section,
    area_section,
    perimeter_box,
    density_steel,
    c_steel_T,
    h_conv,
    emissivity_resultant,
):
    """
    SI UNITS FOR INPUTS AND OUTPUTS.
    :param time:
    :param temperature_ambient:
    :param perimeter_section:
    :param area_section:
    :param perimeter_box:
    :param density_steel:
    :param c_steel_T:
    :param h_conv:
    :param emissivity_resultant:
    :return:
    """

    # Create steel temperature change array s
    temperature_rate_steel = time * 0.0
    temperature_steel = time * 0.0
    heat_flux_net = time * 0.0
    c_s = time * 0.0

    k_sh = (
        0.9 * (perimeter_box / area_section) / (perimeter_section / area_section)
    )  # BS EN 1993-1-2:2005 (e4.26a)
    F = perimeter_section
    V = area_section
    rho_s = density_steel
    h_c = h_conv
    sigma = 56.7e-9  # todo: what is this?
    epsilon = emissivity_resultant

    time_, temperature_steel[0], c_s[0] = iter(time), temperature_ambient[0], 0.0
    next(time_)
    for i, v in enumerate(time_):
        i += 1
        T_f, T_s_ = (
            temperature_ambient[i],
            temperature_steel[i - 1],
        )  # todo: steel specific heat
        c_s[i] = c_steel_T(temperature_steel[i - 1] + 273.15)

        # BS EN 1993-1-2:2005 (e4.25)
        a = h_c * (T_f - T_s_)
        b = sigma * epsilon * (np.power(T_f, 4) - np.power(T_s_, 4))
        c = k_sh * F / V / rho_s / c_s[i]
        d = time[i] - time[i - 1]

        heat_flux_net[i] = a + b

        temperature_rate_steel[i] = c * (a + b) * d
        temperature_steel[i] = temperature_steel[i - 1] + temperature_rate_steel[i]

    return temperature_steel, temperature_rate_steel, heat_flux_net, c_s
