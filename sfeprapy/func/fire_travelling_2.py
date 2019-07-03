import numpy as np


def fire(
        t_array_s:np.ndarray,
        T_0_K:float,
        q_fd:float,
        HRRD:float,
        l_m:float,
        w_m:float,
        s:float,
        h_s:float,
        l_s:float,
        temperature_max:float=1050.,
):
    """
    :param t_array_s: ndarray, [s] An array representing time incorporating 'temperature'.
    :param T_0_K: float, [K] ,Initial temperature.
    :param q_fd: float, [J/m2], Fire load density.
    :param HRRD: float, [W/m2], Heat release rate density.
    :param l_m: float, [m], Compartment length.
    :param w_m: float, [m], Compartment width.
    :param s: float, [m/s], Fire spread speed.
    :param h_s: float, [m], Vertical distance between element to fuel bed.
    :param l_s: float, [m], Horizontal distance between element to fire front.
    :return temperature: [K] An array representing temperature incorporating 'time'.
    """

    # DEPRECIATED 02/05/2019
    # :param A_v: [float][m2] Ventilation area
    # :param h_eq: [float][m] Weighted ventilation height

    # SETTINGS

    # UNIT CONVERSION TO FIT EQUATIONS
    T_0_K -= 273.15
    q_fd /= 1e6
    HRRD /= 1e6

    #
    time_step = t_array_s[1] - t_array_s[0]

    # workout burning time etc.
    t_burn = max([q_fd / HRRD, 900.])
    t_decay = max([t_burn, l_m / s])
    t_lim = min([t_burn, l_m / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    t_decay_ = round(t_decay/time_step, 0) * time_step
    t_lim_ = round(t_lim/time_step, 0) * time_step
    if t_decay_ == t_lim_: t_lim_ -= time_step

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (HRRD * w_m * s * t_array_s) * (t_array_s < t_lim_)
    Q_peak = min([HRRD * w_m * s * t_burn, HRRD * w_m * l_m]) * (t_array_s >= t_lim_) * (t_array_s <= t_decay_)
    Q_decay = (max(Q_peak) - (t_array_s - t_decay_) * w_m * s * HRRD) * (t_array_s > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.

    # workout the distance between fire_curve median to the structural element r
    l_fire_front = s * t_array_s
    l_fire_front[l_fire_front < 0] = 0.
    l_fire_front[l_fire_front > l_m] = l_m
    l_fire_end = s * (t_array_s - t_lim)
    l_fire_end[l_fire_end < 0] = 0.
    l_fire_end[l_fire_end > l_m] = l_m
    l_fire_median = (l_fire_front + l_fire_end) / 2.
    r = np.absolute(l_s - l_fire_median)
    r[r == 0] = 0.001  # will cause crash if r = 0

    # workout the far field temperature of gas T_g
    T_g1 = (5.38 * np.power(Q / r, 2 / 3) / h_s) * ((r/h_s) > 0.18)
    T_g2 = (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5/3)) * ((r/h_s) <= 0.18)
    T_g = T_g1 + T_g2 + T_0_K

    T_g[T_g >= temperature_max] = temperature_max

    # UNIT CONVERSION TO FIT OUTPUT (SI)
    T_g_K = T_g + 273.15  # C -> K
    Q *= 10e6  # MJ -> J

    return T_g_K


if __name__ == '__main__':

    import numpy as np

    time = np.arange(0, 22080, 30)
    list_l = [50, 100, 150]

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-paper')
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel('Time [minute]')
    ax.set_ylabel('Temperature [$^{\circ}C$]')

    for l in list_l:
        temperature = fire(
            t_array_s=time,
            T_0_K=293.15,
            q_fd=900e6,
            HRRD=0.15e6,
            l_m=l,
            w_m=17.4,
            s=0.012,
            h_s=3.5,
            l_s=l/2,
        )
        ax.plot(time / 60, temperature - 273.15)

    ax.legend().set_visible(True)
    ax.set_xlim((0,120))
    ax.grid(color='k', linestyle='--')
    plt.tight_layout()
    plt.show()
