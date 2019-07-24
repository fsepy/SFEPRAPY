# -*- coding: utf-8 -*-

import warnings
import numpy as np
import copy


def heat_transfer_general_1d_finite_difference(
        dt: float,
        k: np.ndarray,
        rho: np.ndarray,
        c: np.ndarray,
        T: np.ndarray,
        dx: np.ndarray,
        h0: float,
        T0: float,
        h1: float,
        T1: float
):
    """
    Calculates the current temperature for layers based on given thermal properties (i.e. k, rho, c, T and dx).
    This function returns a ndarray which contains temperatures of different layers, i.e. [T_0, T_1, T_2, ... T_n].

    NOTE:
    inert boundary - make inner_conductivity = 0.5 * dx[0] / k[0] or outer_conductivity = 0.5 * dx[n] / k[n]

    :param k:   [W/m/K]     thermal conductivity
    :param rho: [kg/m3]     density of boundary material
    :param c:   [J/kg/K]    specific heat of boundary material
    :param T:   [K]         temperature of boundary material (previous time step)
    :param dx:  [m]         virtual layers' thickness of boundary layer
    :param dt:  [s]         time step
    :param h0:  [W/m/K]     overall dQ/dt at j = -1 side (interior)
    :param T0:  [K]         temperature at j = -1 side (interior)
    :param h1:  [W/m/K]     overall dQ/dt at j = n+1 side (exterior)
    :param T1:  [K]         temperature of at j = n+1 side (exterior)
    :return:
    """
    # =================================================== PERMEABLE ====================================================
    # refactor parameters for readability

    k_ = 0.5 * (k[:1] + k[:-1])
    rho_ = 0.5 * (rho[1:] + rho[:-1])
    c_ = 0.5 * (c[1:] + c[:-1])

    T = T
    x_ = 0.5 * (dx[1:] + dx[:-1])
    h0 = h0
    T0 = T0
    h1 = h1
    T1 = T1

    # instantiate returning parameter T_i, current temperature at all virtual layers
    dT = np.zeros_like(T, dtype=float)

    # ====================================== FINITE DIFFERENCE METHOD CALCULATION ======================================

    j = 0
    left = h0 / rho[j] / c[j] / dx[j] * (T0 - T[j])
    right = k_[j] / rho_[j] / c_[j] / x_[j] * (T[j] - T[j+1])
    dT[j] = (left - right) * dt
    j = -1
    left = k_[j] / rho_[j] / c_[j] / x_[j] * (T[j] - T[j+1])
    right = h1 / rho[j] / c[j] / dx[j] * (T[j] - T1)
    dT[j] = (left - right) * dt

    print(dT[0], dT[-1])

    T[0] += dT[0]
    T[-1] += dT[-1]

    # left = k_[:-1] / rho_[:-1] / c_[:-1] / x_[:-1] * (T[:-2] - T[1:-1])
    # right = k_[1:] / rho_[1:] / c_[1:] / x_[1:] * (T[1:-1] - T[2:])
    # dTdt[1:-1] = left - right

    left = k_[:-1] / rho_[:-1] / c_[:-1] / (x_[:-1] ** 2) * (T[:-2] - T[1:-1])
    right = k_[1:] / rho_[1:] / c_[1:] / (x_[1:] ** 2) * (T[1:-1] - T[2:])
    dT[1:-1] = left - right

    return dT


def c_steel_T(temperature):

    temperature -= 273.15

    if temperature < 20:
        warnings.warn('Temperature ({:.1f} °C) is below 20 °C'.format(temperature))
        return 425 + 0.773 * 20 - 1.69e-3 * np.power(20, 2) + 2.22e-6 * np.power(20, 3)
    if 20 <= temperature < 600:
        return 425 + 0.773 * temperature - 1.69e-3 * np.power(temperature, 2) + 2.22e-6 * np.power(temperature, 3)
    elif 600 <= temperature < 735:
        return 666 + 13002 / (738 - temperature)
    elif 735 <= temperature < 900:
        return 545 + 17820 / (temperature - 731)
    elif 900 <= temperature <= 1200:
        return 650
    elif temperature > 1200:
        warnings.warn('Temperature ({:.1f} °C) is greater than 1200 °C'.format(temperature))
        return 650
    else:
        warnings.warn('Temperature ({:.1f} °C) is outside bound.'.format(temperature))
        return 0

from typing import Union


def steel_c_T_vectorised(T: np.ndarray):

    T = T - 273.15  # unit conversion: K -> deg.C

    c = np.where(T < 20, 425, 0)
    c = np.where((20 <= T) & (T <= 600), 425 + 0.773 * T - 1.69e-3 * np.power(T, 2) + 2.22e-6 * np.power(T, 3), c)
    c = np.where((600 < T) & (T <= 735), 666 + 13002 / (738 - T), c)
    c = np.where((735 < T) & (T <= 900), 545 + 17820 / (T - 731), c)
    c = np.where((900 < T) & (T <= 1200), 650, c)
    c = np.where(1200 < T, 651, c)

    return c


def k_steel_T(temperature):
    temperature -= 273.15
    if temperature < 20:
        warnings.warn('Temperature ({:.1f} °C) is below 20 °C'.format(temperature))
        return 54 - 3.33e-2 * 20
    if temperature < 800:
        return 54 - 3.33e-2 * temperature
    elif temperature <= 1200:
        return 27.3
    elif temperature > 1200:
        warnings.warn('Temperature ({:.1f} °C) is greater than 1200 °C'.format(temperature))
        return 27.3
    else:
        warnings.warn('Temperature ({:.1f} °C) is outside bound.'.format(temperature))
        return 0


def steel_k_T_vectorised(T: np.ndarray):

    T = T - 273.15  # unit conversion: K -> deg.C

    k = np.where(T < 20, 54, 0)
    k = np.where((20 <= T) & (T <= 800), 54 - 3.33e-2 * T, k)
    k = np.where((800 < T) & (T <= 1200), 27.3, k)
    k = np.where(1200 < T, 27.3, k)

    return k


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    # solid properties
    thickness = 0.1
    m = 20
    rho = 7850  # kg/m3
    c_func = c_steel_T  # J/kg/K
    k_func = k_steel_T  # W/K/m

    # boundary properties
    U0 = 1000 + 273.15
    h0 = 25
    U1 = 20 + 273.15
    h1 = 4
    emissivity = 0.7

    # time properties
    t0 = 0
    t1 = 600

    # construct vectors
    thickness += (thickness / m * 2)

    m += 2

    x = np.linspace(0, thickness, 2*m+1)[1::2]  # location of nodes

    dx = np.full((m,), thickness/m)  # thickness of nodes
    dx[[0, -1]] = 1  # gas nodes have unity thickness

    U = np.ones_like(dx) * (20 + 273.15)  # temperature of nodes

    rho = np.full_like(dx, rho)

    h = np.zeros_like(dx)  # convection coefficients
    h[0], h[-1] = h0, h1  # convection at gas nodes

    e = np.zeros_like(dx)  # radiation heat transfer coefficient
    e[[0, -1]] = 5.67e-8 * emissivity  # radiation at gas nodes

    # ==================================================================================================================
    # ONE-DIMENSIONAL FINITE DIFFERENCE START
    # ==================================================================================================================

    c = np.full_like(dx, 460)
    # k = np.full_like(dx, 30)

    dU_dt_ = list()  # results container - temperature change rate
    U_ = list()

    t = t0  # set simulation time
    while t <= t1:
        print(int(t))

        if int(t) == 300:
            print('d')

        c = steel_c_T_vectorised(U)
        # k = np.fromiter(map(k_steel_T, U), dtype=float) / dx
        k = steel_k_T_vectorised(U) / dx
        k[[0, -1]] = 0

        dt = 0.005  # constant time step

        # Temperatures at boundary (i.e. gas)
        # ===================================

        U[0], U[-1] = U0, U1  # boundary at constant temperature
        # Temperatures at (solid) outer nodes, only energy in and out, no into other solids
        # =================================================================================
        u0 = U[0] - U[1]
        u1 = U[1] - U[2]
        u0_4 = U[0]**4 - U[1]**4
        u1_4 = U[1]**4 - U[2]**4
        e0 = u0 * h[0] - u1 * h[1]  # convection
        e1 = u0 * k[0] - u1 * k[1]  # conduction
        e2 = u0_4 * e[0] - u1_4 * e[1]  # radiation
        e3 = rho[1] * c[1] * dx[1]  # stored
        dU_dt_0 = (e0 + e1 + e2) / e3
        U[1] = U[1] + dU_dt_0 * dt

        # Temperatures at intermediate nodes
        # ==================================

        for m in range(2, len(dx)-2, 1):
            du0 = U[m-1] - U[m]
            du1 = U[m] - U[m+1]
            e1 = du0 * k[m] - du1 * k[m]
            en = rho[m] * c[m] * dx[m]
            du_dt = e1 / en
            U[m] += du_dt * dt

        # u0 = U[1:-3] - U[2:-2]
        # u1 = U[2:-2] - U[3:-1]
        # e0 = u0 * h[1:-3] - u1 * h[3:-1]
        # e1 = u0 * k[2:-2] - u1 * k[2:-2]
        # e3 = rho[2:-2] * c[2:-2] * dx[2:-2] ** 2
        # dU_dt = (e0 + e1) / e3
        # U[2:-2] = U[2:-2] + dU_dt * dt

        # Temperatures at (solid) outer nodes, only energy in and out, no into other solids
        # =================================================================================

        u0 = U[-3] - U[-2]
        u1 = U[-2] - U[-1]
        u0_4 = U[-3]**4 - U[-2]**4
        u1_4 = U[-2]**4 - U[-1]**4
        e0 = u0 * h[-2] - u1 * h[-1]  # convection
        e1 = u0 * k[-2] - u1 * k[-1]  # conduction
        e2 = u0_4 * e[-2] - u1_4 * e[-1]  # radiation
        e3 = rho[-2] * c[-2] * dx[-2]  # stored
        dU_dt_1 = (e0 + e1 + e2) / e3
        U[-2] = U[-2] + dU_dt_1 * dt

        # dU_dt_.append(dU_dt)
        U_.append(copy.copy(U))

        # Increment time
        # ==============

        t += dt

    fig, ax1 = plt.subplots()
    # dU_dt_ = np.stack(dU_dt_, axis=0)
    # U_stacked = np.stack(U_, axis=0)
    # sns.heatmap(U_stacked[0::100, 1:-1], ax=ax1, xticklabels=np.round(x[1:-1], 2), yticklabels=[])
    for u in U_[::int(len(U_)/10)]:
        sns.lineplot(x[1:-1], u[1:-1]-273.15, ax=ax1)
    sns.lineplot(x[1:-1], U[1:-1]-273.15, ax=ax1, label=int(t))

    plt.tight_layout()
    plt.show()
