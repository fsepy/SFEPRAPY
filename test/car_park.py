import numpy as np
from scipy import stats

# Fire growth coefficient, Q_dot_max [kW]
# Obtained from Table 5-4 in Mohd (2015)
peak_heat_release_rate_stats = dict(
    mini=dict(dist='weibull_min', c=5.19, scale=3809),
    light=dict(dist='weibull_min', c=1.66, scale=5078),
    compact=dict(dist='weibull_min', c=2.40, scale=4691),
    medium=dict(dist='weibull_min', c=3.18, scale=7688),
    heavy=dict(dist='weibull_min', c=4.25, scale=4588),
)

# Fire growth coefficient, alpha_peak [kW/min**2]
# Obtained from Table 5-4 in Mohd (2015)
fire_growth_stats = dict(
    mini=dict(dist='gamma', a=1.39, scale=11.86),
    light=dict(dist='gamma', a=1.23, scale=14.78),
    compact=dict(dist='gamma', a=1.18, scale=5.14),
    medium=dict(dist='gamma', a=2.24, scale=2.75),
    heavy=dict(dist='gamma', a=0.36, scale=159.18),
)

# Fire decay coefficient, beta_exp [min**-1]
# Obtained from Table 5-4 in Mohd (2015)
fire_decay_stats = dict(
    mini=dict(dist='weibull_min', c=0.93, scale=0.17),
    light=dict(dist='weibull_min', c=1.21, scale=0.11),
    compact=dict(dist='weibull_min', c=3.93, scale=0.08),
    medium=dict(dist='weibull_min', c=1.38, scale=0.11),
    heavy=dict(dist='weibull_min', c=2.51, scale=0.08),
)

# Vehicle weight, [kg]
# Table 3-1, ANSI classification of vehicles by curb weight
car_curb_weight_stats = dict(
    mini=dict(dist='uniform', loc=680, scale=906),
    light=dict(dist='uniform', loc=907, scale=1134),
    compact=dict(dist='uniform', loc=1135, scale=1360),
    medium=dict(dist='uniform', loc=1361, scale=1587),
    heavy=dict(dist='uniform', loc=1588, scale=1),
)

# Car width, [m]
# https://www.nimblefins.co.uk/cheap-car-insurance/average-car-dimensions
car_width_stats = dict(
    mini=dict(dist='uniform', loc=1.615, scale=1.670),  # city car
    light=dict(dist='uniform', loc=1.735, scale=1.805),  # hatchback
    compact=dict(dist='uniform', loc=1.735, scale=1.805),  # hatch back
    medium=dict(dist='uniform', loc=1.825, scale=1.842),  # saloon
    heavy=dict(dist='uniform', loc=1.900, scale=2.012),  # SUV
)


def car_fire_hrr(Q_dot_max: float, alpha_peak: float, beta_exp: float, t: np.ndarray):
    """
    :param Q_dot_max:
    :param alpha_peak:
    :param beta_exp:
    :param t:
    :return: [kW]
    """
    t = t / 60.  # s -> min

    Q_dot = np.zeros_like(t)
    t_max = (Q_dot_max / alpha_peak) ** 0.5
    t_end = t_max - np.log(50 / Q_dot_max) / beta_exp
    Q_dot[t <= t_max] = alpha_peak * (t[t <= t_max] ** 2)
    Q_dot[(t > t_max) & (t <= t_end)] = Q_dot_max * np.exp(beta_exp * (t_max - t[(t > t_max) & (t <= t_end)]))

    return Q_dot


def _test_car_fire_hrr(t=np.arange(0, 3600)):
    return car_fire_hrr(
        Q_dot_max=4000,
        alpha_peak=11,
        beta_exp=0.17,
        t=t
    )


def point_source_hrr(Q_dot, lambda_r, R):
    q_dot_fl = Q_dot * lambda_r / (4 * 3.1415926 * R ** 2)
    return q_dot_fl


def _test_point_source_hrr(Q_dot=_test_car_fire_hrr()):
    return point_source_hrr(
        Q_dot=Q_dot,
        lambda_r=0.6,
        R=1.2
    )


def flux_time_product(t, q_dot, q_dot_crit, n):
    q_dot_ave = (q_dot[1:] + q_dot[:-1]) / 2
    ftp_i = (np.abs(q_dot_ave - q_dot_crit) ** n) * (t[1:] - t[:-1])
    ftp = np.zeros_like(t)
    ftp[1:] = np.cumsum(ftp_i)
    return ftp


def _test_flux_time_product(t=np.arange(0, 3600), q_dot=_test_point_source_hrr()):
    ftp = flux_time_product(t=t, q_dot=q_dot, q_dot_crit=5.7, n=1.5)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(t, ftp)
    fig.show()
    return ftp


def car_time_to_ignition(t, q_dot, q_dot_crit, n, ftp):
    ftp_ = flux_time_product(t=t, q_dot=q_dot, q_dot_crit=q_dot_crit, n=n)
    if any(ftp_ >= ftp):
        t_ig = np.amin(t[ftp_ >= ftp])
    else:
        t_ig = 0
    return t_ig


def _test_car_time_to_ignition():
    t = np.arange(0, 3600)
    q_dot_crit = 5.7
    n = 1.5
    Q_dot_max = 4000
    alpha_peak = 11.
    beta_exp = 0.17
    lambda_r = 0.6
    R = 1.5
    ftp = 3258

    Q_dot = car_fire_hrr(Q_dot_max=Q_dot_max, alpha_peak=alpha_peak, beta_exp=beta_exp, t=t)
    q_dot = point_source_hrr(Q_dot=Q_dot, lambda_r=lambda_r, R=R)
    t_ig = car_time_to_ignition(t=t, q_dot=q_dot, q_dot_crit=q_dot_crit, n=n, ftp=ftp)
    print(t_ig)
    assert abs(t_ig - 437) <= 1.


def generate_rvs(n_samples: int, dist_param):
    dist_type = dist_param.pop('dist')
    try:
        dist = getattr(stats, dist_type)(**dist_param)
    except Exception as e:
        raise TypeError(f'Failed to generate random variables {dist_type}, {dist_param}, {e}')
    return dist.rvs(size=n_samples)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_samples = 1000
    arr_parking_width = np.linspace(2.2, 3.2, 1000)
    np.random.shuffle(arr_parking_width)

    # params = peak_heat_release_rate_stats['mini']
    # values = generate_rvs(n_samples=1000, dist_param=params)

    car_type_probability = dict(mini=.09, light=.22, compact=.27, medium=.27, heavy=.15)

    # ==========
    # Fire curve
    # ==========

    # hrr curve parameters based upon car type statistics
    arr_Q_dot_max = np.array([])
    arr_alpha_peak = np.array([])
    arr_beta_exp = np.array([])
    arr_car_width = np.array([])
    for car_type, prob in car_type_probability.items():
        n_samples_i = int(prob * n_samples)
        Q_dot_max = generate_rvs(n_samples=n_samples_i, dist_param=peak_heat_release_rate_stats[car_type].copy())
        alpha_peak = generate_rvs(n_samples=n_samples_i, dist_param=fire_growth_stats[car_type].copy())
        beta_exp = generate_rvs(n_samples=n_samples_i, dist_param=fire_decay_stats[car_type].copy())
        car_width = generate_rvs(n_samples=n_samples_i, dist_param=car_width_stats[car_type].copy())
        arr_Q_dot_max = np.hstack((arr_Q_dot_max, Q_dot_max))
        arr_alpha_peak = np.hstack((arr_alpha_peak, alpha_peak))
        arr_beta_exp = np.hstack((arr_beta_exp, beta_exp))
        arr_car_width = np.hstack((arr_car_width, car_width))

    print([f'{i:.2f}' for i in arr_alpha_peak])

    # make fire curves, find ignition time
    t = np.arange(0, 3600, 1)
    arr_t_ig = list()
    arr_spread_speed = list()
    for i in range(len(arr_Q_dot_max)):
        Q_dot_max = arr_Q_dot_max[i]
        alpha_peak = arr_alpha_peak[i]
        beta_exp = arr_beta_exp[i]
        car_width = arr_car_width[i]
        parking_width = arr_parking_width[i]
        hrr = car_fire_hrr(Q_dot_max=Q_dot_max, alpha_peak=alpha_peak, beta_exp=beta_exp, t=t)
        hrr_received = point_source_hrr(hrr, lambda_r=0.6, R=parking_width)
        t_ig = car_time_to_ignition(t=t, q_dot=hrr_received, q_dot_crit=3.1, n=2, ftp=21862)  # page 207
        arr_t_ig.append(t_ig)
        if t_ig > 0:
            arr_spread_speed.append(parking_width / t_ig)

    print(max(arr_t_ig), min(arr_t_ig), sum(arr_t_ig)/len(arr_t_ig))
    print(max(arr_spread_speed), min(arr_spread_speed), sum(arr_spread_speed)/len(arr_spread_speed))
    print(','.join([f'{i:.5f}' for i in arr_spread_speed]))

    # from sfeprapy.func.stats_dist_fit import fit
    # fig, ax = plt.subplots()
    #
    # list_dist, list_params, list_sse = fit(arr_spread_speed, ax = ax, distribution_list=0)
    #
    # # FINDING THE BEST FIT
    #
    # list_dist = np.asarray(list_dist)[np.argsort(list_sse)]
    # list_params = np.asarray(list_params)[np.argsort(list_sse)]
    # list_sse = np.asarray(list_sse)[np.argsort(list_sse)]
    #
    # print(
    #     "\n{:30.30}{}".format(
    #         "Distribution (sorted)",
    #         "Loss (Residual sum of squares) and distribution parameters",
    #     )
    # )
    #
    # for i, v in enumerate(list_dist):
    #     dist_name = v.name
    #     sse = list_sse[i]
    #     dist_params = ", ".join(["{:10.2f}".format(j) for j in list_params[i]])
    #     print(f"{dist_name:30.30}{sse:10.5E} - [{dist_params}]")
    #
    # dist_best = list_dist[0]
    # params_best = list_params[0]

    # PRODUCE FIGURES

    # ax_fitting.set_xlabel("Sample value")
    # ax_fitting.set_ylabel("PDF")
    # ax_fitting.legend().set_visible(True)
    # fig_fitting.savefig(os.path.join(dir_work, "distfit_fitting.png"))
    #
    # cdf_x_sampled = np.sort(samples)
    # cdf_y_sampled = np.linspace(0, 1, len(cdf_x_sampled))
    # cdf_y_fitted = np.linspace(0, 1, 1000)
    # cdf_x_fitted = dist_best.ppf(cdf_y_fitted, *params_best)
    # fig.show()