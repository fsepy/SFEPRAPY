# -*- coding: utf-8 -*-
import warnings

import numpy as np


def fire(
    t_array_s,
    A_w_m2,
    h_w_m2,
    A_t_m2,
    A_f_m2,
    t_alpha_s,
    b_Jm2s05K,
    q_x_d_MJm2,
    gamma_fi_Q=1.0,
    q_ref=1300,
    alpha=0.0117,
    hrrpua_MWm2=0.25,
):
    """This piece of code calculates a time dependent temperature array in accordance of Appendix AA in German Annex "
    Simplified natural fire model for fully developed room fires" to Eurocode 1991-1-2 (DIN EN 1991-1-2/NA:2010-12).
    Limitations are detailed in the Section AA.2 and they are:
      - minimum fuel load 100 MJ/sq.m
      - maximum fuel load 1,300 MJ/sq.m
      - maximum floor area 400 sq.m
      - maximum ceiling height 5 m
      - minimum vertical vent opening to floor area 12.5 %
      - maximum vertical vent opening to floor area 50 %

    PARAMETERS:
    :param t_array_s: numpy.array, in [s], time
    :param A_w_m2: float, [m2], is window opening area
    :param h_w_m2: float, [m2], is weighted window opening height
    :param A_t_m2: float, [m2], is total enclosure internal surface area, including openings
    :param A_f_m2: float, [m2], is total floor area
    :param t_alpha_s: float, [s], is the fire growth factor, Table BB.2, for residential/office t_alpha = 300 [s]
    :param b_Jm2s05K: float, [J/m2/s0.5/K], is the weighted heat storage capacity, see equation AA.31 and Table AA.1
    :param q_x_d_MJm2: float, [MJ/m2], is the design value for fire load density, same as q_f_d
    :param gamma_fi_Q: float, is the partial factor according to BB.5.3
    :param q_ref: float, [MJ/m2], is the reference upper bound heat release rate, 1300 [MJ/m2] for offices and residential
    :param alpha: float, [kW/s2], is the growth constant for t-square fire
    :param hrrpua_MWm2: float, [MW], is the heat release rate per unit area
    :return T: numpy.ndarray, in [K], is calculated gas temperature within fire enclosure

    SUPPLEMENTAL DATA:

    DIN EN 1991-1-2/NA:1010-12 (English translation)

    Table AA.1 - Influence groups as a function of heat storage capacity b
    | Line | Influence group | Heat storage capacity b [J/m2/s0.5/K] |
    |------|-----------------|---------------------------------------|
    |    1 |               1 |                                  2500 |
    |    2 |               2 |                                  1500 |
    |    3 |               3 |                                   750 |

    Table BB.2
    | Line |                 Occupancy                  | Fire propagation | t_alpha [s] | RHR_f [MW/m2] |
    |------|--------------------------------------------|------------------|-------------|---------------|
    |    1 | Residential building                       | medium           |         300 |          0.25 |
    |    2 | Office building                            | medium           |         300 |          0.25 |
    |    3 | Hospital (room)                            | medium           |         300 |          0.25 |
    |    4 | Hotel (room)                               | medium           |         450 |          0.25 |
    |    5 | Library                                    | medium           |         300 |          0.25 |
    |    6 | School (classroom)                         | medium           |         300 |          0.15 |
    |    7 | Shop, shopping centre                      | fast             |         150 |          0.25 |
    |    8 | Place of public assembly (theatre, cinema) | fast             |         150 |          0.50 |
    |    9 | Public transport                           | slow             |         600 |          0.25 |
    """

    # AA.1
    Q_max_v_k = 1.21 * A_w_m2 * np.sqrt(h_w_m2)  # [MW] AA.1

    # AA.2
    Q_max_f_k = hrrpua_MWm2 * A_f_m2  # [MW] AA.2

    # AA.3, Characteristic value of the maximum HRR, is the lower value of the Q_max_f_k and Q_max_v_k
    Q_max_k = min(Q_max_f_k, Q_max_v_k)  # [MW]

    # AA.5
    Q_max_v_d = gamma_fi_Q * Q_max_v_k  # [MW] AA.5
    Q_max_f_d = gamma_fi_Q * Q_max_f_k  # [MW] AA.6, gamma_fi_Q see BB.5.3

    # AA.6
    Q_max_d = gamma_fi_Q * Q_max_k

    # Work out fire type
    if Q_max_v_k == Q_max_k:
        fire_type = 0
    elif Q_max_f_k == Q_max_k:
        fire_type = 1
    else:
        fire_type = np.nan

    # print(fire_type)

    # AA.7 - AA.19: Calculate location of t and Q
    O = A_w_m2 * h_w_m2 ** 0.5 / A_t_m2
    Q_d = q_ref * A_f_m2  # [MJ], total fire load in the compartment
    if fire_type == 0:  # ventilation controlled fire

        # AA.7
        t_1 = t_alpha_s * np.sqrt(Q_max_v_d)  # [s] AA.7

        # AA.8
        T_1_v = -8.75 / O - 0.1 * b_Jm2s05K + 1175  # [deg.C]

        Q_1 = t_1 ** 3 / (3 * t_alpha_s ** 2)

        # AA.9
        Q_2 = 0.7 * Q_d - Q_1  # AA.9(a)
        t_2 = t_1 + Q_2 / Q_max_v_d  # [s] AA.9(b)

        # AA.10
        T_2_v = min(
            1134, (0.004 * b_Jm2s05K - 17) / O - 0.4 * b_Jm2s05K + 2175
        )  # [deg.C] AA.10

        # AA.11
        Q_3 = 0.3 * Q_d
        t_3 = t_2 + (2 * Q_3)

        # AA.12
        T_3_v = -5.0 / O - 0.16 * b_Jm2s05K + 1060  # [deg.C]

        T_1, T_2, T_3 = T_1_v, T_2_v, T_3_v

    elif fire_type == 1:  # fuel controlled fire

        # AA.19
        k = (
            (Q_max_f_d ** 2) / (A_w_m2 * h_w_m2 ** 0.5 * (A_t_m2 - A_w_m2) * b_Jm2s05K)
        ) ** (1 / 3)
        # print(h_w_m2, A_t_m2, A_w_m2, A_w_m2 * h_w_m2**0.5 * (A_t_m2-A_w_m2) * b_Jm2s05K)

        # AA.13
        t_1 = t_alpha_s * Q_max_f_d ** 0.5
        Q_1 = 1 / 3 * alpha * t_alpha_s ** 3
        Q_1 *= 1e-3

        # AA.14
        T_1_f = min(980, 24000 * k + 20)  # [deg.C]

        # AA.15
        Q_2 = 0.7 * Q_d - Q_1
        t_2 = t_1 + Q_2 / Q_max_f_d

        # AA.16
        T_2_f = min(1340, 33000 * k + 20)  # [deg.C]

        # AA.17
        Q_3 = 0.3 * Q_d
        t_3 = t_2 + (2 * Q_3) / Q_max_f_d

        # AA.18
        T_3_f = min(660, 16000 * k + 20)  # [deg.C]

        # AA.19
        # See above

        T_1, T_2, T_3 = T_1_f, T_2_f, T_3_f

    else:
        wmsg = "WTH, I do not know what type of fire ({}) this is, bugs?".format(
            fire_type
        )
        warnings.warn(wmsg)
        k = t_1 = T_1 = Q_2 = t_2 = Q_2_f = Q_3 = t_3 = Q_3_f = T_2 = T_3 = np.nan

    # Prerequisite for AA.20 and AA.21
    Q_1 = t_1 ** 3 / (3 * t_alpha_s ** 2)  # [MW]
    Q_x_d = q_x_d_MJm2 * A_f_m2

    if Q_1 < 0.7 * Q_x_d:
        # AA.20
        t_2_x = t_1 + (0.7 * Q_x_d - t_1 ** 3 / (3 * t_alpha_s ** 2)) / Q_max_d  # [s]

        # AA.21
        T_2_x = (T_2 - T_1) * ((t_2_x - t_1) / (t_2 - t_1)) ** 0.5 + T_1  # [deg.C]
    elif Q_1 >= 0.7:
        # AA.22
        t_1_x = (0.7 * Q_x_d * 3 * t_alpha_s ** 2) ** (1 / 3)  # [s]
        t_2_x = (0.7 * Q_x_d * 3 * t_alpha_s ** 2) ** (1 / 3)  # [s]

        # AA.23
        T_2_x = (T_1 - 20) / (t_1 ** 2) * t_1_x ** 2 + 20  # [deg.C]
    else:
        warnings.warn("Q_1 out of bound for AA.20 to AA.23.")
        t_1_x = t_2_x = T_2_x = np.nan

    # AA.25
    t_3_x = 0.6 * Q_x_d / Q_max_d + t_2_x  # [s]

    # AA.24
    T_3_x = T_3 * np.log10(t_3_x / 60 + 1) / np.log10(t_3 / 60 + 1)  # [deg.C]

    T = T_t(
        t_array_s,
        t_1,
        t_2,
        t_2_x,
        t_3_x,
        T_1,
        T_2_x,
        T_3_x,
        A_t_m2,
        A_w_m2,
        h_w_m2,
        t_alpha_s,
    )

    # CONVERT UNITS TO SI
    T += 273.15

    return T


# AA.26 - AA.28
def T_t(
    t, t_1, t_2, t_2_x, t_3_x, T_1, T_2_x, T_3_x, A_t, A_w, h_w, t_alpha, T_initial=20
):

    # Initialise container for return value
    T = np.zeros(len(t))

    # Check flash-over
    # AA.30
    Q_fo = 0.0078 * A_t + 0.378 * A_w * h_w ** 0.5  # [MW]
    # AA.29
    t_1_fo = (t_alpha ** 2 * Q_fo) ** 0.5  # [s]

    t_1 = min(t_1, t_1_fo)

    # AA.26
    t_1_ = np.logical_and(0 <= t, t <= t_1)

    T_1_ = (T_1 - 20) / t_1 ** 2 * t[t_1_] ** 2 + 20
    T[t_1_] = T_1_  # [deg.C]

    # AA.27
    # t_2_ = np.logical_and(t_1 <= t, t <= t_2)
    # T_2_ = (T_2_x - T_1) * ((t[t_2_] - t_1) / (t_2_x - t_1)) ** 0.5 + T_1
    # T[t_2_] = T_2_  # [deg.C]

    # AA.27 MOD
    t_2_ = np.logical_and(t_1 <= t, t <= t_2_x)
    T_2_ = (T_2_x - T_1) * ((t[t_2_] - t_1) / (t_2_x - t_1)) ** 0.5 + T_1
    T[t_2_] = T_2_  # [deg.C]

    # AA.28
    # t_3_ = t > t_2
    # T_3_ = (T_3_x - T_2_x) * ((t[t_3_] - t_2) / (t_3_x - t_2_x)) ** 0.5 + T_2_x
    # T[t_3_] = T_3_  # [deg.C]

    # AA.28 MOD
    t_3_ = t > t_2_x
    T_3_ = (T_3_x - T_2_x) * ((t[t_3_] - t_2_x) / (t_3_x - t_2_x)) ** 0.5 + T_2_x
    T[t_3_] = T_3_  # [deg.C]

    # No temperate below T_initial
    T[T < T_initial] = T_initial

    return T


def example_plot_interflam():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-paper")
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_ylabel("Temperature [$^{℃}C$]")

    # define geometry
    w = 16
    l = 32
    h = 3

    h_v = 2

    oo = (0.02, 0.04, 0.06, 0.10, 0.14, 0.20)  # desired opening factor

    def of2wv(of, w, l, h, h_v):
        # opening factor = Av * sqrt(hv) / At
        # Av = h_v * w_v
        # of * At / sqrt(h_v) = h_v * w_v
        # w_v = of * At / sqrt(h_v) / h_v
        return of * (2 * (w * l + l * h + h * w)) / h_v ** 0.5 / h_v

    w_v_ = [of2wv(o, w, l, h, h_v) for o in oo]

    print(w_v_)

    for w_v in w_v_:
        q_ref = 1300
        x = np.arange(0, 5 * 60 * 60 + 1, 1)
        y = fire(
            t_array_s=x,
            A_w_m2=h_v * w_v,
            h_w_m2=h_v,
            A_t_m2=2 * (l * w + w * h + h * l),
            A_f_m2=w * l,
            t_alpha_s=300,
            b_Jm2s05K=720,
            q_x_d_MJm2=600,
            gamma_fi_Q=1.0,
            q_ref=q_ref,
        )
        ax.plot(
            x / 60,
            y - 273.15,
            label="Opening Factor {:.2f}".format(
                (h_v * w_v) * h_v ** 0.5 / (2 * (w * h + h * l + l * w))
            ),
        )

    ax.legend().set_visible(True)
    ax.grid(color="k", linestyle="--")
    ax.set_ylim((0, 1400))
    ax.set_xlim((0, 300))
    plt.tight_layout()
    plt.savefig(fname="fire-ec_din.png", dpi=300)


def plot_variable_qfd():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-paper")
    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_ylabel("Temperature [$^{℃}C$]")

    # define geometry

    w = 16
    l = 32
    h = 3

    h_v = 2
    w_v = 5

    q_fd_ = [150, 300, 450, 600]

    for q_fd in q_fd_:
        q_ref = 1300
        x = np.arange(0, 5 * 60 * 60 + 1, 1)
        y = fire(
            t_array_s=x,
            A_w_m2=h_v * w_v,
            h_w_m2=h_v,
            A_t_m2=2 * (w * l + l * h + h * w),  # 80,
            A_f_m2=w * l,
            t_alpha_s=300,
            b_Jm2s05K=750,
            q_x_d_MJm2=q_fd,
            gamma_fi_Q=1.0,
            q_ref=q_ref,
        )
        ax.plot(
            x / 60,
            y - 273.15,
            label="Fuel load density {q_fd:4.0f} $MJ/m^2$".format(q_fd=q_fd),
        )

    ax.legend(loc=4).set_visible(True)
    ax.grid(color="k", linestyle="--")
    ax.set_ylim((-10, 1310))
    plt.tight_layout()
    plt.savefig(fname="fire-ec_din.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # plot_variable_qfd()
    example_plot_interflam()
