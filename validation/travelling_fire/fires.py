# -*- coding: utf-8 -*-


def least_square(x1, y1, x2, y2):
    import numpy as np
    from scipy.interpolate import interp1d

    f1 = interp1d(x1, y1)
    y2_interpolated = f1(x2)

    return np.sum(np.square(y2-y2_interpolated))


def travelling_fire():
    from sfeprapy.func.temperature_fires import travelling_fire as fire

    time, temperature, data = fire(
        T_0=293.15,
        q_fd=900e6,
        RHRf=0.15e6,
        l=150,
        w=17.4,
        s=0.012,
        h_s=3.5,
        l_s=105,
        time_step=1200,
        time_ubound=22080
    )

    benchmark_time = [0, 1200, 2400, 3600, 4800, 6000, 7200, 8400, 9600, 10800, 12000, 13200, 14400, 15600, 16800, 18000, 19200, 20400, 21600, 22800]
    benchmark_temperature = [293, 374, 428, 481, 534, 592, 643, 722, 870, 1288, 1323, 1071, 773, 592, 454, 293, 293, 293, 293, 293]

    # Load data

    # Calculate least square against the referenced data
    x1 = time
    y1 = temperature
    x2 = benchmark_time
    y2 = benchmark_temperature

    print('Check least squares < 10: {}'.format(least_square(x1, y1, x2, y2) < 10))


if __name__ == "__main__":
    travelling_fire()
