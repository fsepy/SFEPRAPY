# -*- coding: utf-8 -*-
import copy

# [Eq. 11.1 & 11.2]
def _theta_c(Q, r, h):
    """
    :param Q: [W] Rate of heat release from the fire.
    :param r: [m] Radial distance from the centre of the fire plume impingement.
    :param h: [m]
    :return theta_c: [K]
    """
    Q = copy.copy(Q) / 1000.  # Unit: [W] to [kW]

    # Condition for r/h
    if r/h > 0.18:
        theta_c = 5.38 * (Q/r)**(2/3) / h
    else:
        theta_c = 16.9 * Q**(2/3) / h ** (5/3)

    theta_c += 273.15  # Unit: [C] to [K]
    return theta_c


def _U(Q, r, h):
    """[Eq. 11.3 & 11.4]
    :param Q: [W] Rate of heat release from the fire.
    :param r: [m] Radial distance from the centre of the fire plume impingement.
    :param h: [m]
    :return U: [m/s]
    """
    Q = copy.copy(Q) / 1000.  # Unit: [W] to [kW]

    if r/h > 0.15:
        U = 0.195 * Q ** (1/3) * h ** (1/2) / r ** (5/6)
    else:
        U = 0.96 * Q / h ** (1/3)

    return U


def _dT_d_dt(U, T_g, T_d, RTI):
    """[Eq. 11.5]
    :param U: [m/s]
    :param T_g: [K]
    :param T_d: [K]
    :param RTI: [-]
    :return dT_d_dt: [K/s]
    """

    dT_d_dt = U**(1/2) * (T_g - T_d) / RTI

    return dT_d_dt


def _Q(alpha, t):
    """[Eq. 6.1]
    :param alpha:
    :param t:
    :return:
    """
    Q = alpha * t ** 2
    return Q


if __name__ == "__main__":
    import numpy as np

    # INPUTS
    time_start = 0
    time_step = 0.5
    time_end = 30 * 60

    alpha = 0.0117e3  # [W/s2]
    r = 2.82  # Estimation
    h = 6
    RTI = np.average([80, 200])
    T_d_activation = 273.15 + 68  # [K]

    # CONTAINERS
    time = np.arange(time_start, time_end, time_step)
    Q = time * 0
    T_g = time * 0
    T_d = time * 0

    # INITIAL CONDITIONS
    T_g[0] = 273.15
    T_d[0] = 273.15

    # CALCULATION
    # calculate heat release rate
    Q = _Q(alpha, time)
    # calculate jet speed near sprinkler
    U = _U(Q, r, h)

    # calculate temperature near sprinkler
    T_g = _theta_c(Q, r, h)

    # calculate sprinkler temperature
    iter_time = enumerate(time)
    next(iter_time)
    for i, t in iter_time:
        T_d[i] = T_d[i-1] + _dT_d_dt(U[i], T_g[i], T_d[i-1], RTI)

    # RE-EVALUATE FIRE FOR SUPPRESSION ACTIVATION
    Q[T_d >= T_d_activation] = -1
    T_g[T_d >= T_d_activation] = -1
    T_d[T_d >= T_d_activation] = -1
    Q[Q == -1] = np.max(Q)
    T_d[T_d == -1] = np.max(T_d)
    T_g[T_g == -1] = np.max(T_g)

    print("{:25}: {:7.2f} [min]".format("Sprinkler activated at", np.min(time[T_d == np.max(T_d)])/60.))
    print("{:25}: {:7.2f} [kW]".format("The maximum HRR is", np.max(Q)/1.e3))

    import matplotlib.pyplot as plt
    plt.figure(num=1)
    plt.subplot(211)
    plt.plot(time/60, Q/1e6, label='HRR')
    plt.ylabel("Heat Release Rate [MW]")
    plt.subplot(212)
    plt.plot(time/60, T_g-273.15, label="Temperature (near field)")
    plt.plot(time/60, T_d-273.15, label="Temperature (sprinkler)")
    plt.xlabel("Time [min]")
    plt.ylabel("Temperature [$\degree C$]")
    plt.legend()
    plt.show()
