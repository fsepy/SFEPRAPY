# -*- coding: utf-8 -*-
import numpy as np

def travelling_fire(
        fire_load_density_MJm2,
        heat_release_rate_density_MWm2,
        length_compartment_m,
        width_compartment_m,
        fire_spread_rate_ms,
        height_fuel_to_element_m,
        length_element_to_fire_origin_m,
        time_start_s,
        time_end_s,
        time_interval_s,
        nft_max_C,
        win_width_m,
        win_height_m,
        open_fract
    ):
    # make time array
    t = np.arange(time_start_s, time_end_s, time_interval_s)

    # re-assign variable names for equation readability
    q_fd = fire_load_density_MJm2
    RHRf = heat_release_rate_density_MWm2
    l = max([length_compartment_m, width_compartment_m])
    w = min([length_compartment_m, width_compartment_m])
    s = fire_spread_rate_ms
    h_s = height_fuel_to_element_m
    l_s = length_element_to_fire_origin_m

    # work out ventilation conditions
    a_v = win_height_m * win_width_m * open_fract
    Qv = 1.75 * a_v * np.sqrt(win_height_m)

    # workout burning time etc.
    t_burn = max([q_fd / RHRf, 900.])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    t_decay_ = round(t_decay / time_interval_s, 0) * time_interval_s
    t_lim_ = round(t_lim / time_interval_s, 0) * time_interval_s
    if t_decay_ == t_lim_: t_lim_ -= time_interval_s

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (RHRf * w * s * t) * (t < t_lim_)
    Q_peak = min([RHRf * w * s * t_burn, RHRf * w * l]) * (t >= t_lim_) * (t <= t_decay_)
    Q_decay = (max(Q_peak) - (t - t_decay_) * w * s * RHRf) * (t > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.

    # workout the distance between fire midian to the structural element r
    l_fire_front = s * t
    l_fire_front[l_fire_front < 0] = 0.
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (t - t_lim)
    l_fire_end[l_fire_end < 0] = 0.
    l_fire_end[l_fire_end > l] = l
    l_fire_median = (l_fire_front + l_fire_end) / 2.
    r = np.absolute(l_s - l_fire_median)

    # workout the far field temperature of gas T_g

    T_g1 = (5.38 * np.power(Q / r, 2 / 3) / h_s) * (r / h_s > 0.18).astype(int)
    T_g2 = (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5/3)) * (r/h_s <= 0.18).astype(int)
    T_g = T_g1 + T_g2 + 20.
    T_g[T_g>=nft_max_C] = nft_max_C

    return t, T_g, Q, r
