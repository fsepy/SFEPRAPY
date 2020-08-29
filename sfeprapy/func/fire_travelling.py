# -*- coding: utf-8 -*-
import copy
from typing import Union

import numpy as np


def fire(
        t: np.array,
        fire_load_density_MJm2: float,
        fire_hrr_density_MWm2: float,
        room_length_m: float,
        room_width_m: float,
        fire_spread_rate_ms: float,
        beam_location_height_m: float,
        beam_location_length_m: Union[float, list, np.ndarray],
        fire_nft_limit_c: float,
        *_,
        **__,
):
    """
    This function calculates and returns a temperature array representing travelling fire. This function is NOT in SI.
    :param t: in s, is the time array
    :param fire_load_density_MJm2: in MJ/m2, is the fuel density on the floor
    :param fire_hrr_density_MWm2: in MW/m2, is the heat release rate density
    :param room_length_m: in m, is the room length
    :param room_width_m: in m, is the room width
    :param fire_spread_rate_ms: in m/s, is fire spread speed
    :param beam_location_height_m: in m, is the beam lateral distance to fire origin
    :param beam_location_length_m: in m, is the beam height above the floor
    :param fire_nft_limit_c: in deg.C, is the maximum near field temperature
    :param opening_fraction: in -, is the ventilation opening proportion between 0 to 1
    :param opening_width_m: in m, is ventilation opening width
    :param opening_height_m: in m, is ventilation opening height
    :return T_g: in deg.C, is calculated gas temperature
    """

    # re-assign variable names for equation readability
    q_fd = fire_load_density_MJm2
    HRRPUA = fire_hrr_density_MWm2
    s = fire_spread_rate_ms
    h_s = beam_location_height_m
    l_s = beam_location_length_m
    l = room_length_m
    w = room_width_m
    if l < w:
        l += w
        w = l - w
        l -= w

    # work out ventilation conditions

    # a_v = opening_height_m * opening_width_m * opening_fraction
    # Qv = 1.75 * a_v * np.sqrt(opening_height_m)

    # workout burning time etc.
    t_burn = max([q_fd / HRRPUA, 900.0])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    time_interval_s = t[1] - t[0]
    t_decay_ = round(t_decay / time_interval_s, 0) * time_interval_s
    t_lim_ = round(t_lim / time_interval_s, 0) * time_interval_s
    if t_decay_ == t_lim_:
        t_lim_ -= time_interval_s

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (HRRPUA * w * s * t) * (t < t_lim_)
    Q_peak = (
            min([HRRPUA * w * s * t_burn, HRRPUA * w * l]) * (t >= t_lim_) * (t <= t_decay_)
    )
    Q_decay = (max(Q_peak) - (t - t_decay_) * w * s * HRRPUA) * (t > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.0

    # workout the distance between fire median to the structural element r
    l_fire_front = s * t
    l_fire_front[l_fire_front < 0] = 0
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (t - t_lim)
    l_fire_end[l_fire_end < 0] = 0.0
    l_fire_end[l_fire_end > l] = l
    l_fire_median = (l_fire_front + l_fire_end) / 2.0

    # workout the far field temperature of gas T_g
    if isinstance(l_s, float) or isinstance(l_s, int):
        r = np.absolute(l_s - l_fire_median)
        T_g = np.where((r / h_s) > 0.8, (5.38 * np.power(Q / r, 2 / 3) / h_s) + 20.0, 0)
        T_g = np.where(
            (r / h_s) <= 0.8,
            (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5 / 3)) + 20.0,
            T_g,
        )
        T_g[T_g >= fire_nft_limit_c] = fire_nft_limit_c
        return T_g
    elif isinstance(l_s, np.ndarray) or isinstance(l_s, list):
        l_s_list = copy.copy(l_s)
        T_g_list = list()
        for l_s in l_s_list:
            r = np.absolute(l_s - l_fire_median)
            T_g = np.where(
                (r / h_s) > 0.8, (5.38 * np.power(Q / r, 2 / 3) / h_s) + 20.0, 0
            )
            T_g = np.where(
                (r / h_s) <= 0.8,
                (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5 / 3)) + 20.0,
                T_g,
            )
            T_g[T_g >= fire_nft_limit_c] = fire_nft_limit_c
            T_g_list.append(T_g)
        return T_g_list
    else:
        raise TypeError('Unknown type of parameter "l_s": {}'.format(type(l_s)))


def fire_backup(
        t: np.ndarray,
        T_0: float,
        q_fd: float,
        hrrpua: float,
        l: float,
        w: float,
        s: float,
        e_h: float,
        e_l: float,
        T_max: float = 1323.15,
):
    """
    :param t: ndarray, [s] An array representing time incorporating 'temperature'.
    :param T_0: float, [K] ,Initial temperature.
    :param q_fd: float, [J/m2], Fire load density.
    :param hrrpua: float, [W/m2], Heat release rate density.
    :param l: float, [m], Compartment length.
    :param w: float, [m], Compartment width.
    :param s: float, [m/s], Fire spread speed.
    :param e_h: float, [m], Vertical distance between element to fuel bed.
    :param e_l: float, [m], Horizontal distance between element to fire front.
    :return temperature: [K] An array representing temperature incorporating 'time'.
    """

    # UNIT CONVERSION TO FIT EQUATIONS
    T_0 -= 273.15
    q_fd /= 1e6
    hrrpua /= 1e6
    T_max -= 273.15

    # workout time step
    time_step = t[1] - t[0]

    # workout burning time etc.
    t_burn = max([q_fd / hrrpua, 900.0])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    t_decay_ = round(t_decay / time_step, 0) * time_step
    t_lim_ = round(t_lim / time_step, 0) * time_step
    if t_decay_ == t_lim_:
        t_lim_ -= time_step

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (hrrpua * w * s * t) * (t < t_lim_)
    Q_peak = (
            min([hrrpua * w * s * t_burn, hrrpua * w * l]) * (t >= t_lim_) * (t <= t_decay_)
    )
    Q_decay = (max(Q_peak) - (t - t_decay_) * w * s * hrrpua) * (t > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.0

    # workout the distance between fire_curve median to the structural element r
    l_fire_front = s * t
    l_fire_front[l_fire_front < 0] = 0.0
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (t - t_lim)
    l_fire_end[l_fire_end < 0] = 0.0
    l_fire_end[l_fire_end > l] = l
    l_fire_median = (l_fire_front + l_fire_end) / 2.0
    r = np.absolute(e_l - l_fire_median)
    r[r == 0] = 0.001  # will cause crash if r = 0

    # workout the far field temperature of gas T_g
    T_g1 = (5.38 * np.power(Q / r, 2 / 3) / e_h) * ((r / e_h) > 0.18)
    T_g2 = (16.9 * np.power(Q, 2 / 3) / np.power(e_h, 5 / 3)) * ((r / e_h) <= 0.18)
    T_g = T_g1 + T_g2 + T_0

    T_g[T_g >= T_max] = T_max

    # UNIT CONVERSION TO FIT OUTPUT (SI)
    T_g = T_g + 273.15  # C -> K
    Q *= 10e6  # MJ -> J

    return T_g


def example_plot_interflam():
    time = np.arange(0, 210 * 60, 30)
    list_l = [25, 50, 100, 150]

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-paper")
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_ylabel("Temperature [$℃$]")

    for length in list_l:
        temperature = fire(
            t=time,
            fire_load_density_MJm2=600,
            fire_hrr_density_MWm2=0.25,
            room_length_m=length,
            room_width_m=16,
            fire_spread_rate_ms=0.012,
            beam_location_height_m=3,
            beam_location_length_m=length / 2,
            fire_nft_limit_c=1050,
        )

        ax.plot(time / 60, temperature, label="Room length {:4.0f} m".format(length))

    ax.legend(loc=4).set_visible(True)
    ax.set_xlim((0, 180))
    ax.set_ylim((0, 1400))
    ax.grid(color="k", linestyle="--")
    plt.tight_layout()
    plt.savefig(fname="fire-travelling.png", dpi=300)


def _test_fire_backup():
    import numpy as np

    time = np.arange(0, 22080, 30)
    list_l = [50, 100, 150]

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-paper")
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_ylabel("Temperature [$℃$]")

    for l in list_l:
        temperature = fire_backup(
            t=time,
            T_0=293.15,
            q_fd=900e6,
            hrrpua=0.15e6,
            l=l,
            w=17.4,
            s=0.012,
            e_h=3.5,
            e_l=l / 2,
        )
        ax.plot(time / 60, temperature - 273.15)

    ax.legend().set_visible(True)
    ax.set_xlim((0, 120))
    ax.grid(color="k", linestyle="--")
    plt.tight_layout()
    # plt.show()


def _test_fire():
    time = np.arange(0, 210 * 60, 30)
    list_l = [25, 50, 100, 150]

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-paper")
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_ylabel(u"Temperature [$℃$]")

    for length in list_l:
        temperature = fire(
            t=time,
            fire_load_density_MJm2=600,
            fire_hrr_density_MWm2=0.25,
            room_length_m=length,
            room_width_m=16,
            fire_spread_rate_ms=0.012,
            beam_location_height_m=3,
            beam_location_length_m=length / 2,
            fire_nft_limit_c=1050,
        )

        ax.plot(time / 60, temperature, label="Room length {:4.0f} m".format(length))

    ax.legend(loc=4).set_visible(True)
    ax.set_xlim((-10, 190))
    ax.grid(color="k", linestyle="--")
    plt.tight_layout()
    # plt.show()


def _test_fire_multiple_beam_location():
    time = np.arange(0, 210 * 60, 30)
    length = 100

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-paper")
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_ylabel("Temperature [$℃$]")

    temperature_list = fire(
        t=time,
        fire_load_density_MJm2=600,
        fire_hrr_density_MWm2=0.25,
        room_length_m=length,
        room_width_m=16,
        fire_spread_rate_ms=0.012,
        beam_location_height_m=3,
        beam_location_length_m=np.linspace(0, length, 12)[1:-1],
        fire_nft_limit_c=1050,
    )

    for temperature in temperature_list:
        ax.plot(time / 60, temperature, label="Room length {:4.0f} m".format(length))

    ax.legend(loc=4).set_visible(True)
    ax.set_xlim((-10, 190))
    ax.grid(color="k", linestyle="--")
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    _test_fire()
    _test_fire_multiple_beam_location()
    # example_plot_interflam()
