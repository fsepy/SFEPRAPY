# -*- coding: utf-8 -*-
import numpy as np


def annex_b_equivalent_time_of_fire_exposure(
    q_fk, delta_1, m, k_b, A_f, H, A_vh, A_vv, delta_h
):
    """Calculates time equivalence of standard fire exposure in accordance with PD 6688-1-2 Annex B
    IAN FU 12 APRIL 2019

    :param q_fk: [MJ/m2] Fire load density
    :param delta_1: Active suppression factor
    :param m: Combustion factor
    :param k_b: Conversion factor
    :param A_f: [m2] Floor area of the compartment
    :param H: [m] Height of the compartment
    :param A_vh: [m2] Horizontal area
    :param A_vv: [m2] Vertical area
    :param delta_h: Height factor
    :return: Equivalent time exposure

    SUPPLEMENT INFO:
    Table A.4 - Design fire growth rates (from PD 7974-1:2003, Table 3)
    ╔══════════════════════════════════════════════════════════════════╦════════════╗
    ║ BUILDING                                                         ║ USE        ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Picture gallery                                                  ║ Slow       ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Passenger stations and termini for air, rail, road or sea travel ║ Slow       ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Classroom (school)                                               ║ Medium     ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Dwelling                                                         ║ Medium     ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Office                                                            ║ Medium     ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Hotel reception                                                  ║ Medium     ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Hotel bedroom                                                    ║ Medium     ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Hospital room                                                    ║ Medium     ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Library                                                          ║ Fast       ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Theatre (cinema)                                                 ║ Fast       ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Shop                                                             ║ Fast       ║
    ╠══════════════════════════════════════════════════════════════════╬════════════╣
    ║ Industrial storage or plant room                                 ║ Ultra-fast ║
    ╚══════════════════════════════════════════════════════════════════╩════════════╝

    Table A.5 - Heat release rate per unit area of fire for different occupancies (from PD 7974-1:2003)
    ╔═════════════╦══════════════════════════════════════════════╗
    ║ OCCUPANCY   ║ HEAT RELEASE RATE PER UNIT AREA, Q'' [kW/m2] ║
    ╠═════════════╬══════════════════════════════════════════════╣
    ║ Shops       ║ 550                                          ║
    ╠═════════════╬══════════════════════════════════════════════╣
    ║ Offices      ║ 290                                          ║
    ╠═════════════╬══════════════════════════════════════════════╣
    ║ Hotel rooms ║ 249                                          ║
    ╠═════════════╬══════════════════════════════════════════════╣
    ║ Industrial  ║ 86-620                                       ║
    ╚═════════════╩══════════════════════════════════════════════╝


    EXAMPLE:
    >>> kwargs = dict(q_fk=900, delta_1=0.61, m=1, k_b=0.09, H=4, A_f=856.5, A_vh=0, A_vv=235.2, delta_h=2)
    >>> print(annex_b_equivalent_time_of_fire_exposure(**kwargs))
    >>> 74.27814882871894
    """

    # B.1
    # Design fire load [MJ/m2]
    q_fd = q_fk * delta_1 * m

    # B.2
    # Vertical opening factor
    alpha_v = min([max([A_vv / A_f, 0.025]), 0.25])
    # horizontal opening factor
    alpha_h = A_vh / A_f
    # just a factor
    b_v = (
        12.5 * (1 + 10 * alpha_v - alpha_v ** 2)
        if (12.5 * (1 + 10 * alpha_v - alpha_v ** 2)) >= 10
        else np.nan
    )
    # total ventilation factor
    w_f = ((6 / H) ** 0.3) * ((0.62 + 90 * (0.4 - alpha_v) ** 4) / (1 + b_v * alpha_h))

    w_f = w_f if A_f >= 100 else np.nan

    return q_fd * k_b * w_f * delta_h


if __name__ == "__main__":
    input_params = dict(
        q_fk=900,
        delta_1=0.61,
        m=1,
        k_b=0.09,
        H=4,
        A_f=856.5,
        A_vh=0,
        A_vv=235.2,
        delta_h=2,
    )
    input_params = dict(
        q_fk=900,
        delta_1=1.0,
        m=1,
        k_b=0.09,
        H=4,
        A_f=877,
        A_vh=0,
        A_vv=98.35,
        delta_h=2,
    )
    res = annex_b_equivalent_time_of_fire_exposure(**input_params)
    print(res / 60)
