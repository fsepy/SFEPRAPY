import numpy as np


def fire(
        t,
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
        rich_return=False
):
    """
    :param t: [ndarray][s] An array representing time incorporating 'temperature'.
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
    :return temperature: [ndarray][K] An array representing temperature incorporating 'time'.
    :return Q: [ndarray][W m2] Time dependent HRR.
    :return r: [ndarray][m] distance between fire centre to structural element.
    """

    # SETTINGS

    # UNIT CONVERSION TO FIT EQUATIONS
    T_0 -= 273.15
    q_fd /= 1e6
    RHRf /= 1e6

    #
    time_step = t[1] - t[0]

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
    Q_growth = (RHRf * w * s * t) * (t < t_lim_)
    Q_peak = min([RHRf * w * s * t_burn, RHRf * w * l]) * (t >= t_lim_) * (t <= t_decay_)
    Q_decay = (max(Q_peak) - (t-t_decay_) * w * s * RHRf) * (t > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.

    # workout the distance between fire_curve median to the structural element r
    l_fire_front = s * t
    l_fire_front[l_fire_front < 0] = 0.
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (t - t_lim)
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

    if rich_return:
        return temperature, Q, r
    else:
        return temperature
