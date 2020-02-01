# -*- coding: utf-8 -*-
import numpy as np


def protected_steel_eurocode(
    time,
    temperature_ambient,
    rho_steel,
    area_steel_section,
    k_protection,
    rho_protection,
    c_protection,
    thickness_protection,
    perimeter_protected,
    terminate_when_cooling=False,
    terminate_max_temperature=np.inf,
):
    """
    SI UNITS!
    This function calculate the temperature curve of protected steel section based on BS EN 1993-1-2:2005, Section 4
    . Ambient (fire) time-temperature data must be given, as well as the parameters specified below.
    :param time: ndarray, [s], time evolution.
    :param temperature_ambient: ndarray, [K], imposed temperature evolution.
    :param rho_steel: float, [kg/m3], steel density.
    :param area_steel_section: float, [m2], steel cross section area.
    :param k_protection: float, [K/kg/m], protection thermal conductivity.
    :param rho_protection: float, [kg/m3], protection density.
    :param c_protection: float, [J/K/kg], protection thermal capacity.
    :param thickness_protection: float, [m], protection thickness.
    :param perimeter_protected: float, [m], protected perimeter.
    :param terminate_when_cooling: bool, [-], if True then terminate and return values when first peak steel
    temperature is observed.
    :return temperature_steel: ndarray, [K], is calculated steel temperature.
    """
    # todo: 4.2.5.2 (2) - thermal properties for the insulation material
    # todo: revise BS EN 1993-1-2:2005, Clauses 4.2.5.2

    # BS EN 1993-1-2:2005, 3.4.1.2
    # return self.__make_property("c")

    def c_steel_T(temperature):

        temperature -= 273.15
        if temperature < 20:
            # warnings.warn('Temperature ({:.1f} °C) is below 20 °C'.format(temperature))
            return (
                425 + 0.773 * 20 - 1.69e-3 * np.power(20, 2) + 2.22e-6 * np.power(20, 3)
            )
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
        elif temperature > 1200:
            # warnings.warn('Temperature ({:.1f} °C) is greater than 1200 °C'.format(temperature))
            return 650
        else:
            # warnings.warn('Temperature ({:.1f} °C) is outside bound.'.format(temperature))
            return 0

    V = area_steel_section
    rho_a = rho_steel
    lambda_p = k_protection
    rho_p = rho_protection
    d_p = thickness_protection
    A_p = perimeter_protected
    c_p = c_protection

    temperature_steel = time * 0.0
    temperature_rate_steel = time * 0.0
    specific_heat_steel = time * 0.0

    # Check time step <= 30 seconds. [BS EN 1993-1-2:2005, Clauses 4.2.5.2 (3)]
    # time_change = gradient(time)
    # if np.max(time_change) > 30.:
    # raise ValueError("Time step needs to be less than 30s: {0}".format(np.max(time)))

    flag_heating_started = False

    temperature_steel[0] = temperature_ambient[
        0
    ]  # initially, steel temperature is equal to ambient
    temperature_ambient_ = iter(temperature_ambient)  # skip the first item
    next(temperature_ambient_)  # skip the first item
    for i, T_g in enumerate(temperature_ambient_):
        i += 1  # actual index since the first item had been skipped.
        try:
            specific_heat_steel[i] = c_steel_T(temperature_steel[i - 1])
        except ValueError:
            specific_heat_steel[i] = specific_heat_steel[i - 1]
            # print(temperature_steel[i-1])

        # Steel temperature equations are from [BS EN 1993-1-2:2005, Clauses 4.2.5.2, Eq. 4.27]
        phi = (c_p * rho_p / specific_heat_steel[i] / rho_a) * d_p * A_p / V

        a = (lambda_p * A_p / V) / (d_p * specific_heat_steel[i] * rho_a)
        b = (T_g - temperature_steel[i - 1]) / (1.0 + phi / 3.0)
        c = (np.exp(phi / 10.0) - 1.0) * (T_g - temperature_ambient[i - 1])
        d = time[i] - time[i - 1]

        temperature_rate_steel[i] = (
            a * b * d - c
        ) / d  # deviated from e4.27, converted to rate [s-1]

        temperature_steel[i] = temperature_steel[i - 1] + temperature_rate_steel[i] * d

        if (temperature_rate_steel[i] > 0 and flag_heating_started is False) and time[
            i
        ] > 1800:
            flag_heating_started = True

        # Terminate steel temperature calculation if necessary
        if (
            terminate_when_cooling
            and flag_heating_started
            and temperature_rate_steel[i] < 0
        ):
            break
        elif flag_heating_started and terminate_max_temperature < temperature_steel[i]:
            break

        # NOTE: Steel temperature can be in cooling phase at the beginning of calculation, even the ambient temperature
        #       (fire) is hot. This is
        #       due to the factor 'phi' which intends to address the energy locked within the protection layer.
        #       The steel temperature is forced to be increased or remain as previous when ambient temperature and
        #       its previous temperature are all higher than the current calculated temperature.
        #       A better implementation is perhaps to use a 1-D heat transfer model.

        # DEPRECIATED 26 MAR 2019
        # if temperature_steel[i] < temperature_steel[i-1] or temperature_steel[i] < temperature_ambient[i]:
        #     temperature_rate_steel[i] = 0
        #     temperature_steel[i] = temperature_steel[i-1]

    return temperature_steel


def protected_steel_eurocode_max_temperature(
    time,
    temperature_ambient,
    rho_steel,
    area_steel_section,
    k_protection,
    rho_protection,
    c_protection,
    thickness_protection,
    perimeter_protected,
    terminate_check_wait_time=3600,
    terminate_max_temperature=np.inf,
):
    """
    LIMITATIONS:
    Constant time interval throughout
    Only one maxima

    SI UNITS!
    This function calculate the temperature curve of protected steel section based on BS EN 1993-1-2:2005, Section 4
    . Ambient (fire) time-temperature data must be given, as well as the parameters specified below.
    :param time:                    {ndarray} [s]
    :param temperature_ambient:     {ndarray} [K]
    :param rho_steel:               {float} [kg/m3]
    :param area_steel_section:      {float} [m2]
    :param k_protection:            {float} [K/kg/m]
    :param rho_protection:          {float} [kg/m3]
    :param c_protection:            {float} [J/K/kg]
    :param thickness_protection:    {float} [m]
    :param perimeter_protected:     {float} [m]
                                                            temperature is observed.
    :return time:                   {ndarray, float} [s]
    :return temperature_steel:      {ndarray, float} [K]
    :return data_all:               {Dict} [-]
    """
    # todo: 4.2.5.2 (2) - thermal properties for the insulation material
    # todo: revise BS EN 1993-1-2:2005, Clauses 4.2.5.2

    def c_steel_T(temperature):

        temperature -= 273.15
        if temperature < 20:
            # warnings.warn('Temperature ({:.1f} °C) is below 20 °C'.format(temperature))
            return (
                425 + 0.773 * 20 - 1.69e-3 * np.power(20, 2) + 2.22e-6 * np.power(20, 3)
            )
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
        elif temperature > 1200:
            # warnings.warn('Temperature ({:.1f} °C) is greater than 1200 °C'.format(temperature))
            return 650
        else:
            # warnings.warn('Temperature ({:.1f} °C) is outside bound.'.format(temperature))
            return 0

    V = area_steel_section
    rho_a = rho_steel
    lambda_p = k_protection
    rho_p = rho_protection
    d_p = thickness_protection
    A_p = perimeter_protected
    c_p = c_protection

    # temperature_steel = time * 0.
    # temperature_rate_steel = time * 0.
    # specific_heat_steel = time * 0.

    # Check time step <= 30 seconds. [BS EN 1993-1-2:2005, Clauses 4.2.5.2 (3)]
    # time_change = gradient(time)
    # if np.max(time_change) > 30.:
    # raise ValueError("Time step needs to be less than 30s: {0}".format(np.max(time)))

    flag_heating_started = False

    # T_ini = temperature_ambient[0]  # temperature at beginning
    T = temperature_ambient[0]  # current steel temperature
    # T_max = -1  # maximum steel temperature
    # c = c_steel_T(293.15)
    d = time[1] - time[0]

    for i in range(1, len(temperature_ambient)):

        T_g = temperature_ambient[i]

        c_s = c_steel_T(T)

        # Steel temperature equations are from [BS EN 1993-1-2:2005, Clauses 4.2.5.2, Eq. 4.27]
        phi = (c_p * rho_p / c_s / rho_a) * d_p * A_p / V

        a = (lambda_p * A_p / V) / (d_p * c_s * rho_a)
        b = (T_g - T) / (1.0 + phi / 3.0)
        c = (np.exp(phi / 10.0) - 1.0) * (T_g - temperature_ambient[i - 1])

        dT = (a * b * d - c) / d  # deviated from e4.27, converted to rate [s-1]

        T += dT * d

        if not (T >= 0 or T <= 0):
            print("d")

        if not flag_heating_started:
            if time[i] >= terminate_check_wait_time:
                if dT > 0:
                    flag_heating_started = True

        # Terminate early if maximum temperature is reached
        elif flag_heating_started:
            if T > terminate_max_temperature:
                break
            if dT < 0:
                T -= dT * d
                break
    # print(T)
    return T


if __name__ == "__main__":
    from sfeprapy.func.fire_iso834 import fire
    import numpy as np

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-paper")
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_ylabel("Temperature [$^{℃}C$]")
    ax.legend().set_visible(True)
    ax.grid(color="k", linestyle="--")

    rho = 7850
    t = np.arange(0, 700, 0.1)
    T = fire(t, 20 + 273.15)
    ax.plot(t / 60, T - 273.15, "k", label="ISO 834 fire")

    list_dp = np.arange(0.0001, 0.01 + 0.002, 0.002)

    for d_p in list_dp:
        T_s = protected_steel_eurocode(
            time=t,
            temperature_ambient=T,
            rho_steel=rho,
            area_steel_section=0.017,
            k_protection=0.2,
            rho_protection=800,
            c_protection=1700,
            thickness_protection=d_p,
            perimeter_protected=2.14,
        )
        plt.plot(
            t / 60,
            T_s - 273.15,
            label="Protection thickness {:2.0f} mm".format(d_p * 1000),
        )

    plt.legend(loc=4)
    plt.tight_layout()
    plt.show()
