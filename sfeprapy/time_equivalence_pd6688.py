# -*- coding: utf-8 -*-


def time_equivalence_pd6688(q_fk, delta_1, m, k_b, H, A_f, A_vh, A_vv, H_v, delta_h):
    import numpy as np

    q_fd = q_fk * delta_1 * m
    alpha_v = min([max([A_vv / A_f, 0.025]), 0.25])
    alpha_h = A_vh / A_f
    b_v = 12.5 * (1 + 10 * alpha_v - alpha_v ** 2) if (12.5 * (1 + 10 * alpha_v - alpha_v ** 2)) >= 10 else np.nan
    w_f = ((0.62 + 90 * (0.4 - alpha_v) ** 4) / (1 + b_v * alpha_h))
    w_f = w_f if A_f >= 100 else np.nan

    return q_fd * k_b * w_f * delta_h


def test_time_equivalence_pd6688():

    input_params = {
        "q_fk": 900, "delta_1": 0.61, "m": 1, "k_b": 0.09, "H": 4, "A_f": 856.5, "A_vh": 0, "A_vv": 235.2, "H_v": 4, "delta_h": 2
    }

    res = time_equivalence_pd6688(**input_params)

    print(res)

