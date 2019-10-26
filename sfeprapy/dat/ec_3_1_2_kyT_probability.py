import numpy as np
import scipy.stats as stats


def ky2T(T):
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


def ky2T_vectorised(T: np.ndarray):

    ky = np.where(T <= 673.15, 1, 0)
    ky = np.where(
        (673.15 < T) & (T <= 773.15), 1.00 + (0.78 - 1.00) / 100 * (T - 673.15), ky
    )
    ky = np.where(
        (773.15 < T) & (T <= 873.15), 0.78 + (0.47 - 0.78) / 100 * (T - 773.15), ky
    )
    ky = np.where(
        (873.15 < T) & (T <= 973.15), 0.47 + (0.23 - 0.47) / 100 * (T - 873.15), ky
    )
    ky = np.where(
        (973.15 < T) & (T <= 1073.15), 0.23 + (0.11 - 0.23) / 100 * (T - 973.15), ky
    )
    ky = np.where(
        (1073.15 < T) & (T <= 1173.15), 0.11 + (0.06 - 0.11) / 100 * (T - 1073.15), ky
    )
    ky = np.where(
        (1173.15 < T) & (T <= 1273.15), 0.06 + (0.04 - 0.06) / 100 * (T - 1173.15), ky
    )
    ky = np.where(
        (1273.15 < T) & (T <= 1373.15), 0.04 + (0.02 - 0.04) / 100 * (T - 1273.15), ky
    )
    ky = np.where(
        (1373.15 < T) & (T <= 1473.15), 0.02 + (0.00 - 0.02) / 100 * (T - 1373.15), ky
    )
    ky = np.where(1473.15 < T, 0, ky)

    return ky


def ky2T_probabilistic_vectorised(T: np.ndarray, epsilon_q: np.ndarray):

    k_y_2_T_bar = ky2T_vectorised(T)

    k_y_2_T_star = (k_y_2_T_bar + 1e-6) / 1.7

    epsilon = stats.norm.ppf(epsilon_q)

    # ky2T_probabilistic_vectorised = (a1 * exp(b1 * b2 * b3 * b4 * b5)) / (exp(b1 + b2 + b3 + b4 + b5) + c1)
    # ky2T_probabilistic_vectorised = (a1 * b6) / (b6 + c1)

    b1 = np.log(k_y_2_T_star / (1 - k_y_2_T_star))
    b2 = 0.412
    b3 = -0.81e-3 * T
    b4 = 0.58e-6 * (T ** 1.9)
    b5 = 0.43 * epsilon
    b6 = np.exp(b1 + b2 + b3 + b4 + b5)

    k_y_2_T_ = (1.7 * b6) / (b6 + 1)

    return k_y_2_T_


def func_test():

    T = np.linspace(0, 1600, 100000) + 273.15

    ky = []
    for i in T:
        ky.append(ky2T(i))

    return T, ky


def func_vector_test():

    T = np.linspace(0, 1600, 100000) + 273.15
    ky = ky2T_vectorised(T)

    return T, ky


def func_prob_vector_test():

    T = np.linspace(273.15 + 0, 273.15 + 1500, 5000)
    q = np.random.random_sample(len(T))

    ky = ky2T_probabilistic_vectorised(T, q)

    return T, ky


def _test_probabilistic():
    assert abs(ky2T_probabilistic_vectorised(0, 0.5) - 1.161499) <= 0.00001
    assert abs(ky2T_probabilistic_vectorised(673.15, 0.5) - 1.001560) <= 0.0001


if __name__ == "__main__":
    _test_probabilistic()
    from timeit import default_timer as timer
    import matplotlib.pyplot as plt

    t1 = timer()
    x1, y1 = func_test()
    t2 = timer()
    x2, y2 = func_vector_test()
    t3 = timer()
    x3, y3 = func_prob_vector_test()

    print(t2 - t1)
    print(t3 - t2)

    fig, ax3 = plt.subplots(figsize=(3.94 * 1.2, 2.76 * 1.2))

    ax3.scatter(x3 - 273.15, y3, c="grey", s=1, label="Random Sampled Points")
    ax3.plot(
        x3 - 273.15,
        ky2T_probabilistic_vectorised(x3, 0.5),
        "--k",
        label="$\epsilon$ Percentile 0.05, 0.5, 0.95",
    )

    print(ky2T_probabilistic_vectorised(673.15, 0.5))

    ax3.plot(x3 - 273.15, ky2T_probabilistic_vectorised(x3, 0.05), "--k")
    ax3.plot(x3 - 273.15, ky2T_probabilistic_vectorised(x3, 0.95), "--k")
    ax3.plot(x3 - 273.15, ky2T_vectorised(x3), "k", label="Eurocode $k_{y,\Theta}$")
    ax3.set_xlabel("Temperature [$^\circ C$]")
    ax3.set_ylabel("$k_{y,ach}$")

    plt.legend(loc=0, fontsize=9)
    plt.tight_layout()
    plt.show()
