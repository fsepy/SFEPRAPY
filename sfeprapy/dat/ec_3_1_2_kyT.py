import numpy as np


def func(T):
    ky = 0
    if T < 673.15:
        ky = 1
    elif T <= 773.15:
        ky = 1.00 + (0.78 - 1.00) / 100 * (T - 673.15)
    elif T <= 873.15:
        ky = 0.78 + (0.47 - 0.78) / 100 * (T - 773.15)
    elif T <= 973.15:
        ky = 0.47 + (0.23 - 0.47) / 100 * (T - 873.15)
    elif T <= 1073.15:
        ky = 0.23 + (0.11 - 0.23) / 100 * (T - 973.15)
    elif T <= 1173.15:
        ky = 0.11 + (0.06 - 0.11) / 100 * (T - 1073.15)
    elif T <= 1273.15:
        ky = 0.06 + (0.04 - 0.06) / 100 * (T - 1173.15)
    elif T <= 1373.15:
        ky = 0.04 + (0.02 - 0.04) / 100 * (T - 1273.15)
    elif T <= 1473.15:
        ky = 0.02 + (0.00 - 0.02) / 100 * (T - 1373.15)

    return ky


def func_v(T):

    ky = np.where(T <= 673.15, 1, 0)
    ky = np.where((673.15 < T) & (T <= 773.15), 1.00 + (0.78 - 1.00) / 100 * (T - 673.15), ky)
    ky = np.where((773.15 < T) & (T <= 873.15), 0.78 + (0.47 - 0.78) / 100 * (T - 773.15), ky)
    ky = np.where((873.15 < T) & (T <= 973.15), 0.47 + (0.23 - 0.47) / 100 * (T - 873.15), ky)
    ky = np.where((973.15 < T) & (T <= 1073.15), 0.23 + (0.11 - 0.23) / 100 * (T - 973.15), ky)
    ky = np.where((1073.15 < T) & (T <= 1173.15), 0.11 + (0.06 - 0.11) / 100 * (T - 1073.15), ky)
    ky = np.where((1173.15 < T) & (T <= 1273.15), 0.06 + (0.04 - 0.06) / 100 * (T - 1173.15), ky)
    ky = np.where((1273.15 < T) & (T <= 1373.15), 0.04 + (0.02 - 0.04) / 100 * (T - 1273.15), ky)
    ky = np.where((1373.15 < T) & (T <= 1473.15), 0.02 + (0.00 - 0.02) / 100 * (T - 1373.15), ky)
    ky = np.where(1473.15 < T, 0, ky)

    return ky


def func_test():

    T = np.linspace(0, 1600, 100000) + 273.15

    ky = []
    for i in T:
        ky.append(func(i))

    return T, ky


def func_vector_test():

    T = np.linspace(0, 1600, 100000) + 273.15
    ky = func_v(T)

    return T, ky


if __name__ == '__main__':
    from timeit import default_timer as timer
    import matplotlib.pyplot as plt

    t1 = timer()
    x1, y1 = func_test()
    t2 = timer()
    x2, y2 = func_vector_test()
    t3 = timer()

    print(t2-t1)
    print(t3-t2)

    fig, (ax1, ax2) = plt.subplots(2, 1, True)

    ax1.plot(x1, y1)
    ax2.plot(x2, y2)

    plt.show()
