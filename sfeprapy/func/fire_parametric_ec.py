import numpy as np


def fire(
    t, A_t, A_f, A_v, h_eq, q_fd, lambda_, rho, c, t_lim, temperature_initial=293.15
):
    """Function Description: (SI UNITS ONLY)
    This function calculates the time-temperature curve according to Eurocode 1 part 1-2, Appendix A.
    :param t: numpy.ndarray, [s], time evolution.
    :param A_t: float, [m2], total surface area (including openings).
    :param A_f: float, [m2], floor area.
    :param A_v: float, [m2], opening area.
    :param h_eq: float, [m2], opening height.
    :param q_fd: float, [J/m2], fuel density.
    :param lambda_: float, [K/kg/m], lining thermal conductivity.
    :param rho: float, [kg/m3], lining density.
    :param c: float, [J/K/kg], lining thermal capacity.
    :param t_lim: float, [s], limiting time for the fire.
    :return T_g: numpy.ndarray, [s], temperature evolution.
    """
    # Reference: Eurocode 1991-1-2; Jean-Marc Franssen, Paulo Vila Real (2010) - Fire Design of Steel Structures

    # UNITS: SI -> Equations
    q_fd /= 1e6  # [J/m2] -> [MJ/m2]
    t_lim /= 3600  # [s] -> [hr]
    t = t / 3600  # [s] -> [hr]
    temperature_initial -= 273.15  # [K] -> [C]

    # ACQUIRING REQUIRED VARIABLES
    # t = np.arange(time_start, time_end, time_step, dtype=float)

    b = (lambda_ * rho * c) ** 0.5
    O = A_v * h_eq ** 0.5 / A_t
    q_td = q_fd * A_f / A_t
    Gamma = ((O / 0.04) / (b / 1160)) ** 2

    t_max = 0.0002 * q_td / O

    # check criteria
    # todo: raise warnings to higher level
    # if not 50 <= q_td <= 1000:
    #     if is_cap_q_td:
    #         msg = "q_td ({:4.1f}) EXCEEDED [50, 1000] AND IS CAPPED.".format(q_td, is_cap_q_td)
    #     else:
    #         msg = "q_td ({:4.1f}) EXCEEDED [50, 1000] AND IS UNCAPPED.".format(q_td)
    #     warnings.warn(msg)

    # CALCULATION
    def _temperature_heating(t_star, temperature_initial):
        # eq. 3.12
        T_g = 1325 * (
            1
            - 0.324 * np.exp(-0.2 * t_star)
            - 0.204 * np.exp(-1.7 * t_star)
            - 0.472 * np.exp(-19 * t_star)
        )
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
        else:
            T_g = np.nan
        return T_g

    def _temperature_cooling_fuel(
        t_star_max, T_max, t_star, Gamma, t_lim
    ):  # fuel controlled
        # eq. 3.22
        if t_star_max <= 0.5:
            T_g = T_max - 625 * (t_star - Gamma * t_lim)
        elif 0.5 < t_star_max < 2.0:
            T_g = T_max - 250 * (3 - t_star_max) * (t_star - Gamma * t_lim)
        elif 2.0 <= t_star_max:
            T_g = T_max - 250 * (t_star - Gamma * t_lim)
        else:
            T_g = np.nan
        return T_g

    def _variables(t, Gamma, t_max):
        t_star = Gamma * t
        t_star_max = Gamma * t_max
        return t_star, t_star_max

    def _variables_2(t, t_lim, q_td, b, O):
        O_lim = 0.0001 * q_td / t_lim
        Gamma_lim = ((O_lim / 0.04) / (b / 1160)) ** 2

        if O > 0.04 and q_td < 75 and b < 1160:
            k = 1 + ((O - 0.04) / (0.04)) * ((q_td - 75) / (75)) * ((1160 - b) / (1160))
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
    T_g[T_g < temperature_initial] = temperature_initial

    # UNITS: Eq. -> SI
    t *= 3600
    T_g += 273.15

    return T_g


def example_plot_interflam():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("ticks")
    sns.set_context(context="paper")

    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.set_xlabel("Time [minute]")
    ax.set_ylabel(r"Temperature [$^{\circ}C$]")

    # define geometry
    w = 16
    l = 32
    h = 3

    h_v = 2

    oo = (0.02, 0.04, 0.06, 0.10, 0.14, 0.20)  # desired opening factor

    def opening_factor_2_window_width(of, w, l, h, h_v):
        # opening factor = Av * sqrt(hv) / At
        # Av = h_v * w_v
        # of * At / sqrt(h_v) = h_v * w_v
        # w_v = of * At / sqrt(h_v) / h_v
        return of * (2 * (w * l + l * h + h * w)) / h_v ** 0.5 / h_v

    w_v_ = [opening_factor_2_window_width(o, w, l, h, h_v) for o in oo]

    print(w_v_)

    # Calculate fire curves
    for w_v in w_v_:
        x = np.arange(0, 5 * 60 * 60, 1)
        y = fire(
            t=x,
            A_v=h_v * w_v,
            A_t=2 * (w * h + h * l + l * w),
            A_f=w * l,
            h_eq=h_v,
            q_fd=600e6,
            lambda_=1,
            rho=1,
            c=720 ** 2,
            t_lim=20 * 60,
            temperature_initial=293.15,
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

    plt.savefig(fname="fire-ec.png", dpi=300)
    plt.show()


def _test_fire():
    """This is a test function to `fire` within this module `fire_parametric_ec`, it compares the function against
    Figure 7 in Holicky, M. et al [1].
    yan fu, 1 oct 2018

    REFERENCES
    [1] Holicky, M., Meterna, A., Sedlacek, G. and Schleich, J.B., 2005. Implementation of eurocodes, handbook 5, design
    of buildings for the fire situation. Leonardo da Vinci Pilot Project: Luxembourg."""

    import copy
    import pandas

    # LOAD VERIFICATION DATA
    # data are from the referenced document
    from io import StringIO

    verification_data = StringIO(
        """time_1,temperature_1,time_2,temperature_2,time_3,temperature_3,time_4,temperature_4,time_5,temperature_5,time_6,temperature_6,time_7,temperature_7
        20,689.1109391,20,689.1109391,20,689.1109391,20,843.9616854,20,741.3497451,20,601.424372,20,269.3348197
        30,782.7282474,30,782.7317977,30,782.7317977,30,946.9144525,30,827.5114674,30,726.7651987,30,430.1234077
        40,167.4009827,40,402.4756096,40,508.8188932,40,781.2034026,40,885.6845648,40,777.4756096,40,533.0726245
        50,20,50,20,50,236.7716603,50,540.8761379,50,818.8612125,50,813.2641976,50,604.3089737
        60,20,60,20,60,20,60,304.2731159,60,692.3328174,60,843.4522203,60,651.2880412
        70,20,70,20,70,20,70,69.53931579,70,565.8079725,70,780.3602113,70,685.2109576
        80,20,80,20,80,20,80,20,80,441.145249,80,719.1303236,80,709.8019654"""
    )
    verification_data = pandas.read_csv(
        verification_data, skip_blank_lines=True, skipinitialspace=True
    )

    # CALCULATE TIME TEMPERATURE CURVE BASED ON THE VERIFICATION DATA INPUTS

    # prepare inputs
    kws = dict(
        A_t=360,
        A_f=100,
        h_eq=1,
        q_fd=600e6,
        lambda_=1,
        rho=1,
        c=2250000,
        t_lim=20 * 60,
        t=np.arange(0, 2 * 60 * 60 + 1, 1),
        temperature_initial=293.15,
    )

    # define opening area
    A_v_list = [72, 50.4, 36.000000001, 32.4, 21.6, 14.4, 7.2]

    # calculate fire curves
    x1_list = []  # calculated time array
    y1_list = []  # calculated temperature array
    for i in A_v_list:
        y = fire(A_v=i, **copy.copy(kws))
        x = np.arange(0, 2 * 60 * 60 + 1, 1) + 10 * 60
        x1_list.append(x / 60)
        y1_list.append(y - 273.15)

    # COMPARE CALCULATED AND THE VERIFICATION DATA

    def r_square(x1, y1, x2, y2):
        from scipy.interpolate import interp1d
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            y2, interp1d(x1, y1)(x2)
        )
        return r_value ** 2

    for i in range(len(A_v_list)):
        x1 = x1_list[i]
        y1 = y1_list[i]
        x2 = verification_data["time_{}".format(int(i + 1))]
        y2 = verification_data["temperature_{}".format(int(i + 1))]
        assert r_square(x1, y1, x2, y2) > 0.99


if __name__ == "__main__":
    # exmaple_plot_interflam()
    _test_fire()
