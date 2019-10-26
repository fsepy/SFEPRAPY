# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d


def protected_steel_bs13381(
    time: np.ndarray,
    temperature_ambient: np.ndarray,
    lambda_pt_T,
    d_p: float,
    A_p: float,
    V: float,
    c_a_T,
    rho_a_T,
    time_ubound: float = 10800,
    time_step_lbound: float = 30.0,
):
    """
    This function estimates the temperature of given protected steel structural element. The calculation procedure is
    in accordance with BS EN 13381-8:2013 "Test methods for determining the contribution to the fire resistance of
    structural members - Part 8: Applied reactive protection to steel members".
    :param time:                {ndarray}   [s]         Time array incorporating temperature_ambient, forming a time-temperature curve
    :param temperature_ambient: {ndarray}   [s]         Temperature array incorporating time, forming a time-temperature curve
    :param lambda_pt_T:         {ky2T}      [W/m/K]     Thermal conductivity of the protective material (temperature dependent)
    :param d_p:                 {float}     [m]         Dry film thickness of reactive product
    :param A_p:                 {float}     [m]         Perimeter of protected
    :param V:                   {float}     [m2]        Area of steel cross section
    :param c_a_T:               {ky2T}      [J/kg/K]    Specific heat capacity of steel (temperature dependent)
    :param rho_a_T:             {ky2T}      [kg/m3]     Density of steel (temperature dependent)
    :param time_ubound:         {float}     [s]         The calculation time
    :param time_step_lbound:    {float}     [s]         The user-defined minimum time step, a smaller time step will be used with Equation E.2
    :return:
    """

    # HELPER FUNCTIONS
    # [BS EN 13381-8:2013, ANNEX E, Equation E.1]
    def _theta_at(lambda_pt, d_p, A_p, V, c_a, rho_a, theta_t, theta_at, dt):
        return (
            (lambda_pt / d_p)
            * (A_p / V)
            * (1 / c_a / rho_a)
            * (theta_t - theta_at)
            * dt
        )

    # [BS EN 13381-8:2013, ANNEX E, Equation E.2]
    def _dt(c_a, rho_a, lambda_pt, d_p, A_p, V):
        return 0.8 * (c_a * rho_a / lambda_pt * d_p) * (V / A_p)

    # [BS EN 13381-8:2013, ANNEX E, Equation E.3]
    def _lambda_pt(d_p, V, A_p, c_a, rho_a, theta_t, theta_at, dt, d_theta_at):
        return (
            d_p
            * (V / A_p)
            * c_a
            * rho_a
            * (1 / ((theta_t - theta_at) * dt))
            * d_theta_at
        )

    # [BS EN 13381-8:2013, ANNEX E, Equation E.4]
    def _theta_pt(theta_t, theta_t_, theta_at, theta_at_):
        return ((theta_t_ + theta_t) / 2.0 + (theta_at_ + theta_at) / 2.0) / 2.0

    # [BS EN 13381-8:2013, ANNEX E, Equation E.5]
    def _d_theta_at(c_a, rho_a, lambda_ave, d_p, A_p, V, theta_t, theta_at, dt):
        return (
            (1 / (c_a + rho_a))
            * (lambda_ave / d_p)
            * (A_p / V)
            * (theta_t - theta_at)
            * dt
        )

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
        time_.append(time_[i - 1] + dt)
        t = time_[i]

        # Determine temperature change rate of the steel
        lambda_ave, theta_t, theta_at = lambda_pt, T_g, temperature_steel[i - 1]
        temperature_rate_steel.append(
            _d_theta_at(c_a, rho_a, lambda_ave, d_p, A_p, V, theta_t, theta_at, dt)
        )
        temperature_steel.append(temperature_steel[i - 1] + temperature_rate_steel[i])

    # Convert lists to ndarray
    time_, time_rate = np.asarray(time_), np.asarray(time_rate)
    temperature_steel, temperature_rate_steel = (
        np.asarray(temperature_steel),
        np.asarray(temperature_rate_steel),
    )

    return time_, temperature_steel, time_rate, temperature_rate_steel
