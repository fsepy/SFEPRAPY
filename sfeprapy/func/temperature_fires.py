# -*- coding: utf-8 -*-
import numpy as np


def parametric_eurocode1(A_t, A_f, A_v, h_eq, q_fd, lambda_, rho, c, t_lim, time_end=7200, time_step=1, time_start=0, time_padding = (0, 0),temperature_initial=293.15, is_more_return=False):
    """Function Description: (SI UNITS ONLY)
    This function calculates the time-temperature curve according to Eurocode 1 part 1-2, Appendix A.
    :param A_t:
    :param A_f:
    :param A_v:
    :param h_eq:
    :param q_fd:
    :param lambda_:
    :param rho:
    :param c:
    :param t_lim:
    :param time_end:
    :param time_step:
    :param time_start:
    :param time_padding:
    :param temperature_initial:
    :return t:
    :return T_g:
    """
    # Reference: Eurocode 1991-1-2; Jean-Marc Franssen, Paulo Vila Real (2010) - Fire Design of Steel Structures

    # UNITS: SI -> Equations
    q_fd /= 1e6  # [J/m2] -> [MJ/m2]
    t_lim /= 3600  # [s] -> [hr]
    time_end /= 3600  # [s] -> [hr]
    time_step /= 3600  # [s] -> [hr]
    time_start /= 3600  # [s] -> [hr]
    temperature_initial -= 273.15  # [K] -> [C]

    # ACQUIRING REQUIRED VARIABLES
    t = np.arange(time_start, time_end, time_step, dtype=float)

    b = (lambda_ * rho * c) ** 0.5
    O = A_v * h_eq**0.5 / A_t
    q_td = q_fd * A_f / A_t
    Gamma = ((O/0.04)/(b/1160))**2

    t_max = 0.0002*q_td/O

    # check criteria
    if not 50 <= q_td <= 1000:
        print("q_td = {:4.1f} not in range [50, 1000]".format(q_td))

    # CALCULATION
    def _temperature_heating(t_star, temperature_initial):
        # eq. 3.12
        T_g = 1325 * (1 - 0.324*np.exp(-0.2*t_star) - 0.204*np.exp(-1.7*t_star) - 0.472*np.exp(-19*t_star))
        T_g += temperature_initial
        return T_g

    def _temperature_cooling_vent(t_star_max, T_max, t_star):  # ventilation controlled
        # eq. 3.16
        if t_star_max <= 0.5:
            T_g = T_max - 625 * (t_star - t_star_max)
        elif 0.5 < t_star_max < 2.0:
            T_g = T_max - 250 * (3 - t_star_max) * (t_star - t_star_max)
        elif 2.0 <= t_star_max:
            T_g = T_max - 250 * (t_star - t_star_max)
        else: T_g = np.nan
        return T_g

    def _temperature_cooling_fuel(t_star_max, T_max, t_star, Gamma, t_lim):  # fuel controlled
        # eq. 3.22
        if t_star_max <= 0.5:
            T_g = T_max - 625 * (t_star - Gamma * t_lim)
        elif 0.5 < t_star_max < 2.0:
            T_g = T_max - 250 * (3 - t_star_max) * (t_star - Gamma * t_lim)
        elif 2.0 <= t_star_max:
            T_g = T_max - 250 * (t_star - Gamma * t_lim)
        else: T_g = np.nan
        return T_g

    def _variables(t, Gamma, t_max):
        t_star = Gamma * t
        t_star_max = Gamma * t_max
        return t_star, t_star_max

    def _variables_2(t, t_lim, q_td, b, O):
        O_lim = 0.0001 * q_td / t_lim
        Gamma_lim = ((O_lim/0.04)/(b/1160))**2

        if O > 0.04 and q_td < 75 and b < 1160:
            k = 1 + ((O-0.04)/(0.04)) * ((q_td-75)/(75)) * ((1160-b)/(1160))
            Gamma_lim *= k

        t_star_ = Gamma_lim * t
        t_star_max_ = Gamma_lim * t_lim
        return t_star_, t_star_max_

    t_star, t_star_max = _variables(t, Gamma, t_max)

    if t_max >= t_lim:  # ventilation controlled fire
        T_max = _temperature_heating(t_star_max, temperature_initial)
        T_heating_g = _temperature_heating(Gamma * t, temperature_initial)
        T_cooling_g = _temperature_cooling_vent(t_star_max, T_max, t_star)
        fire_type = "ventilation controlled"
    else:  # fuel controlled fire
        t_star_, t_star_max_ = _variables_2(t, t_lim, q_td, b, O)
        T_max = _temperature_heating(t_star_max_, temperature_initial)
        T_heating_g = _temperature_heating(t_star_, temperature_initial)
        T_cooling_g = _temperature_cooling_fuel(t_star_max, T_max, t_star, Gamma, t_lim)
        fire_type = "fuel controlled"

    T_g = np.minimum(T_heating_g, T_cooling_g)
    T_g[T_g < 0] = 0

    data_all = {"fire_type": fire_type}

    # UNITS: Eq. -> SI
    t *= 3600
    T_g += 273.15

    if is_more_return:
        return t, T_g, data_all
    else:
        return t, T_g


def standard_fire_iso834(
        time,
        temperature_initial
):
    # INPUTS CHECK
    time = np.array(time, dtype=float)
    time[time < 0] = np.nan

    # SI UNITS -> EQUATION UNITS
    temperature_initial -= 273.15  # [K] -> [C]
    time /= 60.  # [s] - [min]

    # CALCULATE TEMPERATURE BASED ON GIVEN TIME
    temperature = 345. * np.log10(time * 8. + 1.) + temperature_initial
    temperature[temperature == np.nan] = temperature_initial

    # EQUATION UNITS -> SI UNITS
    time *= 60.  # [min] -> [s]
    temperature += 273.15  # [C] -> [K]

    return time, temperature


def standard_fire_astm_e119(
        time,
        temperature_ambient
):
    time /= 1200.  # convert from seconds to hours
    temperature_ambient -= 273.15  # convert temperature from kelvin to celcius
    temperature = 750 * (1 - np.exp(-3.79553 * np.sqrt(time))) + 170.41 * np.sqrt(time) + temperature_ambient
    return temperature + 273.15  # convert from celsius to kelvin (SI unit)


def hydrocarbon_eurocode(
        time,
        temperature_initial
):
    time /= 1200.  # convert time unit from second to hour
    temperature_initial -= 273.15  # convert temperature from kelvin to celsius
    temperature = 1080 * (1 - 0.325 * np.exp(-0.167 * time) - 0.675 * np.exp(-2.5 * time)) + temperature_initial
    return temperature + 273.15


def external_fire_eurocode(
        time,
        temperature_initial
):
    time /= 1200.  # convert time from seconds to hours
    temperature_initial -= 273.15  # convert ambient temperature from kelvin to celsius
    temperature = 660 * (1 - 0.687 * np.exp(-0.32 * time) - 0.313 * np.exp(-3.8 * time)) + temperature_initial
    return temperature + 273.15  # convert temperature from celsius to kelvin


def travelling_fire(
        T_0,
        q_fd,
        RHRf,
        l,
        w,
        s,
        # A_v,
        # h_eq,
        h_s,
        l_s,
        temperature_max=1050,
        time_ubound=10800,
        time_step=1):
    """
    :param T_0: [float][K] Initial temperature.
    :param q_fd: [float][J m2] Fire load density.
    :param RHRf: [float][W m2] Heat release rate density
    :param l: [float][m] Compartment length
    :param w: [float][m] Compartment width
    :param s: [float][m/s] Fire spread speed
    # :param A_v: [float][m2] Ventilation area
    # :param h_eq: [float][m] Weighted ventilation height
    :param h_s: [float][m] Vertical distance between element to fuel bed.
    :param l_s: [float][m] Horizontal distance between element to fire front.
    :param time_ubound: [float][s] Maximum time for the curve.
    :param time_step: [float][s] Static time step.
    :return time: [ndarray][s] An array representing time incorporating 'temperature'.
    :return temperature: [ndarray][K] An array representing temperature incorporating 'time'.
    """

    # SETTINGS
    time_lbound = 0

    # UNIT CONVERSION TO FIT EQUATIONS
    T_0 -= 273.15
    q_fd /= 1e6
    RHRf /= 1e6

    # MAKE TIME ARRAY
    time = np.arange(time_lbound, time_ubound+time_step, time_step)

    # fire_load_density_MJm2=900
    # heat_release_rate_density_MWm2=0.15
    # length_compartment_m=150
    # width_compartment_m=17.4
    # fire_spread_rate_ms=0.012
    # area_ventilation_m2=190
    # height_ventilation_opening_m=3.3
    # height_fuel_to_element_m=3.5
    # length_element_to_fire_origin_m=105

    # workout burning time etc.
    t_burn = max([q_fd / RHRf, 900.])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    t_decay_ = round(t_decay/time_step, 0) * time_step
    t_lim_ = round(t_lim/time_step, 0) * time_step
    if t_decay_ == t_lim_: t_lim_ -= time_step

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (RHRf * w * s * time) * (time < t_lim_)
    Q_peak = min([RHRf * w * s * t_burn, RHRf * w * l]) * (time >= t_lim_) * (time <= t_decay_)
    Q_decay = (max(Q_peak) - (time-t_decay_) * w * s * RHRf) * (time > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.

    # workout the distance between fire_curve midian to the structural element r
    l_fire_front = s * time
    l_fire_front[l_fire_front < 0] = 0.
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (time - t_lim)
    l_fire_end[l_fire_end < 0] = 0.
    l_fire_end[l_fire_end > l] = l
    l_fire_median = (l_fire_front + l_fire_end) / 2.
    r = np.absolute(l_s - l_fire_median)
    r[r == 0] = 0.001  # will cause crash if r = 0

    # workout the far field temperature of gas T_g
    T_g1 = (5.38 * np.power(Q / r, 2 / 3) / h_s) * ((r/h_s) > 0.18)
    T_g2 = (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5/3)) * ((r/h_s) <= 0.18)
    T_g = T_g1 + T_g2 + T_0

    T_g[T_g >= temperature_max] = temperature_max

    # UNIT CONVERSION TO FIT OUTPUT (SI)
    T_g += 273.15  # C -> K
    Q *= 10e6  # MJ -> J

    temperature = T_g

    data_trivial = {
        "heat release [J]": Q,
        "distance fire to element [m]": r
    }
    return time, temperature, data_trivial


def t_square(time, growth_factor, cap_hrr=0, cap_hrr_to_time=0, terminate_at_peak=False):
    if isinstance(growth_factor, str):
        growth_factor = str.lower(growth_factor)
        growth_dict = {"slow": 0.0029,
                       "medium": 0.0117,
                       "fast": 0.0469,
                       "ultra-fast": 0.1876}
        try:
            growth_factor = growth_dict[growth_factor]
        except KeyError:
            err_msg = "{} should be one of the following if not a number: {}"
            err_msg = err_msg.format("growth_factor", ", ".join(list(growth_dict.keys())))
            raise ValueError(err_msg)

    heat_release_rate = growth_factor * time ** 2 * 1000

    # cap hrr
    if cap_hrr:
        heat_release_rate[heat_release_rate>cap_hrr] = cap_hrr

    if cap_hrr_to_time:
        heat_release_rate[time>cap_hrr_to_time] = -1
        heat_release_rate[heat_release_rate==-1] = max(heat_release_rate)

    return heat_release_rate


if __name__ == "__main__":
    time = np.arange(0, )

    t_square()