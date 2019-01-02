# -*- coding: utf-8 -*-

def least_square(x1, y1, x2, y2):
    import numpy as np
    from scipy.interpolate import interp1d

    f1 = interp1d(x1, y1)
    y2_interpolated = f1(x2)

    return np.sum(np.square(y2-y2_interpolated))


def test_eurocode_parametric_fire():
    """
    NAME: _test_eurocode_parametric_fire
    AUTHOR: yan fu
    VERSION: 0.0.1
    DATE: 1 Oct 2018
    DESCRIPTION:
    This function verifies sfeprapy.func.temperature_fires.parametric_eurocode1 based on figure 7 in the referenced
    document below.
    Holicky, M., Meterna, A., Sedlacek, G. and Schleich, J.B., 2005. Implementation of eurocodes, handbook 5, design of
    buildings for the fire situation. Leonardo da Vinci Pilot Project: Luxemboug.

    :return:
    """

    import copy, pandas, os
    import numpy as np
    from sfeprapy.func.temperature_fires import parametric_eurocode1 as fire

    kwargs = {"A_t": 360,
              "A_f": 100,
              # "A_v": 72,
              "h_eq": 1,
              "q_fd": 600e6,
              "lambda_": 1,
              "rho": 1,
              "c": 2250000,
              "t_lim": 20*60,
              "time_end": 2*60*60,
              "time_step": 1,
              "time_start": 0,
              "temperature_initial": 293.15}

    # define opening area
    A_v_list = [72, 50.4, 36.000000001, 32.4, 21.6, 14.4, 7.2]


    # Calculate fire curves
    x1_list = []  # calculated time array
    y1_list = []  # calculated temperature array
    for i in A_v_list:
        x, y = fire(A_v=i, **copy.copy(kwargs))
        x += 10*60
        x1_list.append(x/60)
        y1_list.append(y-273.15)

    # Validation data
    # Load data

    verification_data = pandas.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ec_parametric_fire_verification_data.csv'))

    # Calculate least square against the referenced data
    for i in range(len(A_v_list)):
        x1 = x1_list[i]
        y1 = y1_list[i]
        x2 = verification_data['time_{}'.format(int(i+1))]
        y2 = verification_data['temperature_{}'.format(int(i+1))]

        assert least_square(x1, y1, x2, y2) < 100

    return 0


# if __name__ == '__main__':
#     _test_eurocode_parametric_fire()
