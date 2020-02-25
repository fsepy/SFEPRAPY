# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
import pandas as pd


def Q_star_H(
    HRR: float,
    height: float
):
    # Calcualtes non-dimensional HRR. HRR must be in [W] and height in [m]
    q_star = HRR / (1.11e06 * height **(5/2))
    return q_star

def Q_star_D(
    HRR: float,
    diameter: float
):
    # Calcualtes non-dimensional HRR. HRR must be in [W] and diameter in [m]
    q_star = HRR / (1.11e06 * diameter **(5/2))
    return q_star

def flame_ext_length(
    q_star: float,
    height: float
):
    # Calculates flame extension under the ceiling according to Annex C of EN 1991-1-2
    # q_star_H is dimensionless
    # height in [m]

    lh = (2.9 * height * (q_star**0.33)) - height
    return lh

def y_param(
    q_star_D: float,
    diameter: float,
    lh: float,
    radial_dist: float,
    height: float,
):
    # Calculates dimensionless parameter y for incident heat flux computation
    # Correlations per Annex C of EN 1991-1-2
    # q_star_D is dimensionless
    # remaining parameters in [m]

    # Calculate z'

    z = np.where(q_star_D < 1, 2.4 * diameter * ((q_star_D**(2/5))-(q_star_D**(2/3))),0)
    z = np.where(q_star_D >= 1,2.4 * diameter * (1-(q_star_D**(2/5))),z)

    # Calculate y

    y = (radial_dist + height + z) / (lh + height + z)

    return y

def fire(
    t: np.array,
    fire_load_density_MJm2: float,
    fire_hrr_density_MWm2: float,
    room_length_m: float,
    room_width_m: float,
    fire_spread_rate_ms: float,
    beam_location_height_m: float,
    beam_location_length_m: Union[float, list, np.ndarray],
    fire_nff_limit_kW: float
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

    # fTFM parameters
    fire_area = Q / (HRRPUA*1000)
    fire_dia = ((fire_area / np.pi)**0.5)*2
    q_star_H = Q_star_H(Q*1000, h_s)
    q_star_D = Q_star_D(Q * 1000, fire_dia)
    lh = flame_ext_length(q_star_H, h_s)
    lh[lh<0] = 0

    # Distance related parameters for proximity of ceiling point to flame
    r = np.absolute(l_s - l_fire_median)
    y = y_param(q_star_D,fire_dia,lh,r,h_s)

    # Heat flux computation for near field
    q_inc = np.where(y>0.5, 682 * np.exp(-3.4 * y),0)
    q_inc = np.where(y <= 0.5, fire_nff_limit_kW, q_inc)
    q_inc[q_inc > fire_nff_limit_kW] = fire_nff_limit_kW

    #Calculate far field heat flux
    q_inc_ff = 682 * np.exp(-3.4 * 1)
    q_inc = np.where(y > 1, q_inc_ff * (y**-3.7), q_inc)

    # Where HRR = 0 set Q_inc = 0
    q_inc = np.where(Q<=0,0,q_inc)

    return q_inc

def _test_fire():
    time = np.arange(0, 120 * 60, 10)
    list_l = [5, 10, 15, 20, 25, 30, 35]

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-paper")
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_xlim(0,90)
    ax.set_ylim(0,130)
    ax.set_ylabel('Incident heat flux [kW/m$^2$]')

    dict_data = {
        'time': time,
    }

    for beam_location_length_m in list_l:
        heat_flux = fire(
            t=time,
            fire_load_density_MJm2=760*0.8,
            fire_hrr_density_MWm2=0.5,
            room_length_m=40,
            room_width_m=16,
            fire_spread_rate_ms=0.018,
            beam_location_height_m=3,
            beam_location_length_m= beam_location_length_m,
            fire_nff_limit_kW=120,
        )

        dict_data['Ceiling location = '+ str(beam_location_length_m) + ' [m]'] = heat_flux

        ax.plot(time / 60, heat_flux, label="Ceiling position {:4.0f} m".format(beam_location_length_m))

    ax.legend(loc=4).set_visible(True)
    ax.grid(color="k", linestyle="--")
    plt.tight_layout()
    plt.show()

    df_data = pd.DataFrame.from_dict(dict_data)

    print(df_data)

    df_data.to_csv('test.csv')

if __name__ == "__main__":
    _test_fire()

