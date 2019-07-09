import numpy as np


def fire(
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

    # DEPRECIATED 02/05/2019
    # :param A_v: [float][m2] Ventilation area
    # :param h_eq: [float][m] Weighted ventilation height

    # SETTINGS

    # UNIT CONVERSION TO FIT EQUATIONS
    T_0 -= 273.15
    q_fd /= 1e6
    hrrpua /= 1e6
    T_max -= 273.15

    # workout time step
    time_step = t[1] - t[0]

    # workout burning time etc.
    t_burn = max([q_fd / hrrpua, 900.])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    t_decay_ = round(t_decay/time_step, 0) * time_step
    t_lim_ = round(t_lim/time_step, 0) * time_step
    if t_decay_ == t_lim_: t_lim_ -= time_step

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (hrrpua * w * s * t) * (t < t_lim_)
    Q_peak = min([hrrpua * w * s * t_burn, hrrpua * w * l]) * (t >= t_lim_) * (t <= t_decay_)
    Q_decay = (max(Q_peak) - (t - t_decay_) * w * s * hrrpua) * (t > t_decay_)
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
    ax.set_xlim((0,120))
    ax.grid(color='k', linestyle='--')
    plt.tight_layout()
    plt.show()
