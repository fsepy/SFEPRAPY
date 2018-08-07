# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d


def unprotected_steel_eurocode(
        time,
        temperature_ambient,
        perimeter_section,
        area_section,
        perimeter_box,
        density_steel,
        c_steel_T,
        h_conv,
        emissivity_resultant
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
    temperature_rate_steel = time * 0.
    temperature_steel = time * 0.
    heat_flux_net = time * 0.
    c_s = time * 0.

    k_sh = 0.9 * (perimeter_box / area_section) / (perimeter_section / area_section)  # BS EN 1993-1-2:2005 (e4.26a)
    F = perimeter_section
    V = area_section
    rho_s = density_steel
    h_c = h_conv
    sigma = 56.7e-9  # todo: what is this?
    epsilon = emissivity_resultant

    time_, temperature_steel[0], c_s[0] = iter(time), temperature_ambient[0], 0.
    next(time_)
    for i, v in enumerate(time_):
        i += 1
        T_f, T_s_ = temperature_ambient[i], temperature_steel[i-1]  # todo: steel specific heat
        c_s[i] = c_steel_T(temperature_steel[i - 1] + 273.15)

        # BS EN 1993-1-2:2005 (e4.25)
        a = h_c * (T_f - T_s_)
        b = sigma * epsilon * (np.power(T_f,4) - np.power(T_s_,4))
        c = k_sh * F / V / rho_s / c_s[i]
        d = time[i] - time[i-1]

        heat_flux_net[i] = a + b

        temperature_rate_steel[i] = c * (a + b) * d
        temperature_steel[i] = temperature_steel[i-1] + temperature_rate_steel[i]

    return temperature_steel, temperature_rate_steel, heat_flux_net, c_s


def protected_steel_eurocode(
        time,
        temperature_ambient,
        rho_steel,
        c_steel_T,
        area_steel_section,
        k_protection,
        rho_protection,
        c_protection,
        thickness_protection,
        perimeter_protected,
        is_terminate_peak=False
):
    """
    SI UNITS!
    This function calculate the temperature curve of protected steel section based on BS EN 1993-1-2:2005, Section 4
    . Ambient (fire) time-temperature data must be given, as well as the parameters specified below.
    :param time:                    {ndarray} [s]
    :param temperature_ambient:     {ndarray} [K]
    :param rho_steel:               {float} [kg/m3]
    :param c_steel_T:               {Func} [J/kg/K]
    :param area_steel_section:      {float} [m2]
    :param k_protection:            {float} [K/kg/m]
    :param rho_protection:          {float} [kg/m3]
    :param c_protection:            {float} [J/K/kg]
    :param thickness_protection:    {float} [m]
    :param perimeter_protected:     {float} [m]
    :param is_terminate_peak:       {bool} [-]              True will terminate and return values when first peak steel temperature is observed.
    :return time:                   {ndarray, float} [s]
    :return temperature_steel:      {ndarray, float} [K]
    :return data_all:               {Dict} [-]
    """
    # todo: 4.2.5.2 (2) - thermal properties for the insulation material
    # todo: revise BS EN 1993-1-2:2005, Clauses 4.2.5.2

    V = area_steel_section
    rho_a = rho_steel
    lambda_p = k_protection
    rho_p = rho_protection
    d_p = thickness_protection
    A_p = perimeter_protected
    c_p = c_protection

    temperature_steel = time * 0.
    temperature_rate_steel = time * 0.
    specific_heat_steel = time * 0.

    # Check time step <= 30 seconds. [BS EN 1993-1-2:2005, Clauses 4.2.5.2 (3)]
    # time_change = gradient(time)
    # if np.max(time_change) > 30.:
    # raise ValueError("Time step needs to be less than 30s: {0}".format(np.max(time)))

    temperature_steel[0] = temperature_ambient[0]  # initially, steel temperature is equal to ambient
    temperature_ambient_ = iter(temperature_ambient)  # skip the first item
    next(temperature_ambient_)  # skip the first item
    for i, T_g in enumerate(temperature_ambient_):
        i += 1  # actual index since the first item had been skipped.
        specific_heat_steel[i] = c_steel_T(temperature_steel[i - 1])

        # Steel temperature equations are from [BS EN 1993-1-2:2005, Clauses 4.2.5.2, Eq. 4.27]
        phi = (c_p * rho_p / specific_heat_steel[i] / rho_a) * d_p * A_p / V

        a = (lambda_p*A_p/V) / (d_p * specific_heat_steel[i] * rho_a)
        b = (T_g-temperature_steel[i-1]) / (1.+phi/3.)
        c = (np.exp(phi/10.)-1.) * (T_g-temperature_ambient[i-1])
        d = time[i] - time[i-1]

        temperature_rate_steel[i] = (a * b * d - c) / d  # deviated from e4.27, converted to rate [s-1]

        temperature_steel[i] = temperature_steel[i-1] + temperature_rate_steel[i] * d

        # NOTE: Steel temperature can be decrease at the begining, even the ambient temperature (fire) is hot. This is
        #       due to the factor 'phi' which intends to address the energy locked within the protection layer.
        #       The steel temperature is forced to be increased or remain as previous when ambient temperature and
        #       its previous temperature are all higher than the current calculated temperature.
        #       A better implementation is perhaps to use a 1-D heat transfer model.

        if temperature_steel[i] < min(temperature_steel[i-1], temperature_ambient[i]):
            temperature_rate_steel[i] = 0
            temperature_steel[i] = temperature_steel[i-1]

        if is_terminate_peak and temperature_rate_steel[i] < 0:
            data_all = {"temperature fire [K]": temperature_ambient,
                        "temperature rate steel [K/s]": temperature_rate_steel,
                        "specific heat steel [J/kg/K]": specific_heat_steel}
            return time, temperature_steel, data_all

    data_all = {
        "temperature steel [K]": temperature_steel,
        "temperature rate steel [K/s]": temperature_rate_steel,
        "specific heat steel [J/kg/K]": specific_heat_steel
    }

    return time, temperature_steel, data_all


def protected_steel_bs13381(
        time, temperature_ambient,
        lambda_pt_T, d_p, A_p, V, c_a_T, rho_a_T,
        time_ubound=10800, time_step_lbound=30.
):
    """
    :param time:                {ndarray}   [s]         Time array incorporating temperature_ambient, forming a time-temperature curve
    :param temperature_ambient: {ndarray}   [s]         Temperature array incorporating time, forming a time-temperature curve
    :param lambda_pt_T:         {func}      [W/m/K]     Thermal conductivity of the protective material (temperature dependent)
    :param d_p:                 {float}     [m]         Dry film thickness of reactive product
    :param A_p:                 {float}     [m]         Perimeter of protected
    :param V:                   {float}     [m2]        Area of steel cross section
    :param c_a_T:               {func}      [J/kg/K]    Specific heat capacity of steel (temperature dependent)
    :param rho_a_T:             {func}      [kg/m3]     Density of steel (temperature dependent)
    :param time_ubound:         {float}     [s]         The calculation time
    :param time_step_lbound:    {float}     [s]         The user-defined minimum time step, however, a smaller time step will be used with E.2
    :return:
    """

    # DEFINE FUNCTIONS: There are several functions defined in BS EN 13381-8:2013, ANNEX E, E.3. They are defined in
    # advance for simplification.
    # [BS EN 13381-8:2013, ANNEX E, Equation E.1]
    def _theta_at(lambda_pt, d_p, A_p, V, c_a, rho_a, theta_t, theta_at, dt):
        return (lambda_pt/d_p) * (A_p/V) * (1/c_a/rho_a) * (theta_t-theta_at) * dt

    # [BS EN 13381-8:2013, ANNEX E, Equation E.2]
    def _dt(c_a, rho_a, lambda_pt, d_p, A_p, V):
        return 0.8 * (c_a * rho_a / lambda_pt * d_p) * (V / A_p)

    # [BS EN 13381-8:2013, ANNEX E, Equation E.3]
    def _lambda_pt(d_p, V, A_p, c_a, rho_a, theta_t, theta_at, dt, d_theta_at):
        return d_p * (V / A_p) * c_a * rho_a * (1 / ((theta_t-theta_at)*dt)) * d_theta_at

    # [BS EN 13381-8:2013, ANNEX E, Equation E.4]
    def _theta_pt(theta_t, theta_t_, theta_at, theta_at_):
        return ((theta_t_+theta_t)/2. + (theta_at_+theta_at)/2.) / 2.

    # [BS EN 13381-8:2013, ANNEX E, Equation E.5]
    def _d_theta_at(c_a, rho_a, lambda_ave, d_p, A_p, V, theta_t, theta_at, dt):
        return (1/(c_a+rho_a)) * (lambda_ave/d_p) * (A_p / V) * (theta_t - theta_at) * dt

    # [BS EN 13381-8:2013, ANNEX E, Equation E.6]
    def _lambda_char(lambda_ave, K, sigma):
        return lambda_ave + K * sigma

    # Note: only equation E.1 and E.2 are currently being used as it is assumed the conductivity of protection layer
    #       is known.

    _temperature_gas = interp1d(time, temperature_ambient)

    # Instantiate output containers
    time_ = [0]
    time_rate = [0]
    temperature_steel = [temperature_ambient[0]]
    temperature_rate_steel = [0]
    conductivity_protection = [np.nan]

    # Start iterative calculation
    t = 0
    i = 0
    while t < time_ubound:
        i += 1

        # Determine conditions
        T_g = _temperature_gas(t)

        # Determine thermal properties
        c_a = c_a_T(T_g)
        rho_a = rho_a_T(T_g)
        lambda_pt = lambda_pt_T(T_g)
        conductivity_protection[i] = lambda_pt

        # Determine time step
        dt = _dt(c_a, rho_a, lambda_pt, d_p, A_p, V)
        dt = min([dt, time_step_lbound])
        time_rate.append(dt)

        # Determine current time
        time_.append(time_[i-1] + dt)
        t = time_[i]

        # Determine temperature change rate of the steel
        lambda_ave, theta_t, theta_at = lambda_pt, T_g, temperature_steel[i-1]
        temperature_rate_steel.append(_d_theta_at(c_a, rho_a, lambda_ave, d_p, A_p, V, theta_t, theta_at, dt))
        temperature_steel.append(temperature_steel[i-1] + temperature_rate_steel[i])

    # Convert lists to ndarray
    time_, time_rate = np.asarray(time_), np.asarray(time_rate)
    temperature_steel, temperature_rate_steel = np.asarray(temperature_steel), np.asarray(temperature_rate_steel)
    data_trivial = {""}

    return time_, temperature_steel, time_rate, temperature_rate_steel
