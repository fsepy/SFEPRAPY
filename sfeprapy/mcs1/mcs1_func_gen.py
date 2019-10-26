# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame as df
from scipy import stats
from scipy.interpolate import interp1d


def lognorm_parameters_true_to_inv(miu, sigma):
    """
    NAME: lognorm_parameters_true_to_inv
    VERSION: 0.0.1
    AUTHOR: Yan Fu, Ruben V. Coline
    DESCRIPTION:
    Converts the mean and standard deviation of distribution from x to ln(x).

    PARAMETERS:
    :param miu: True mean of the x
    :param sigma: True standard deviation of x
    :return: (miu, sigma) where miu and sigma are based on ln(x)

    USAGE:
    >>> print(lognorm_parameters_true_to_inv(0.2, 0.2))
    >>> (-1.9560115027140728, 0.8325546111576977)
    """
    cov = sigma / miu

    sigma_ln = np.sqrt(np.log(1 + cov ** 2))
    miu_ln = np.log(miu) - 1 / 2 * sigma_ln ** 2

    return miu_ln, sigma_ln


def lognorm_trunc_ppf(a, b, n_rv, sigma, loc, scale, cdf_y=None):
    """
    NAME: lognorm_trunc_ppf
    VERSION: 0.0.1
    AUTHOR: Yan Fu
    DATE: 3 Aug 2018
    DESCRIPTION:
    Truncated log normal distribution cumulative function density. Truncate and normalise  a log normal distribution
    function for a given boundary (a, b).

    PARAMETERS:
    :param a: float, Lower boundary
    :param b: float, Upper boundary
    :param n_rv: integer, total number of
    :param sigma: float, standard deviation of log normal distribution (for ln(x))
    :param loc: float, location of log normal distribution
    :param scale: float, scale of the log normal distribution
    :param cdf_y: array (1 dimension) or None. A set of numbers represent cumulative probability. If None the function
    will return the sampled values
    :return: array (1 dimension), set of numbers represent sampled values of truncated log normal distribution inline
    with 'cfd_y'

    USAGE:
    >>> import numpy as np
    >>> a = 0                   # lower boundary
    >>> b = 1                   # upper boundary
    >>> n_rv = 5                # number of random variables
    >>> sigma = 0.2             # standard deviation
    >>> miu = 0.2               # mean

    # Convert true mean and sigma to ln(x) based mean and sigma
    >>> miu, sigma = lognorm_parameters_true_to_inv(miu, sigma)

    >>> loc = np.exp(miu)       # distribution mean is 0.2 (i.e. loc = np.exp(miu))
    >>> scale = 1               # default
    >>> result = lognorm_trunc_ppf(a, b, n_rv, sigma, loc, scale)
    >>> print(result)
    [0.14142136 0.49653783 0.65783987 0.81969479 1.        ]
    """

    # Generate a linear spaced array inline with lower and upper boundary of log normal cumulative probability density.
    sampled_cfd = np.linspace(
        stats.lognorm.cdf(x=a, s=sigma, loc=loc, scale=scale),
        stats.lognorm.cdf(x=b, s=sigma, loc=loc, scale=scale),
        n_rv,
    )

    # Sample log normal distribution
    sampled = stats.lognorm.ppf(q=sampled_cfd, s=sigma, loc=loc, scale=scale)

    return sampled


def gumbel_parameter_converter(miu, sigma):
    """
    NAME: gumbel_parameter_converter
    VERSION: 0.0.1
    AUTHOR: Yan Fu, Ruben V. Coline
    DATE: 27 Sept 2018
    DESCRIPTION:
    This function is used in conjunction of the scipy.stats.gumbel distribution, converts mean and standard deviation
    (sigma) of samples x to location and scale which is used in scipy.stats.gumbel_r function.

    :param miu: mean value of samples x
    :param sigma: standard deviation of samples x
    :return: location and scale

    EXAMPLE:
    """

    # parameters Gumbel W&S
    alpha = 1.282 / sigma
    u = miu - 0.5772 / alpha

    # parameters Gumbel scipy
    scale = 1 / alpha
    loc = u

    return loc, scale


def gumbel_parameter_converter2(loc, scale):
    # parameters Gumbel W&S
    # alpha = 1.282 / sigma
    # u = miu - 0.5772 / alpha

    sigma = 1.282 / (1 / scale)
    miu = loc + 0.5772 / (1 / scale)

    # parameters Gumbel scipy
    # scale = 1 / alpha
    # loc = u

    return miu, sigma


def gumbel_r_trunc_ppf(a, b, n_rv, loc, scale, cdf_y=None):
    """
    NAME: gumbel_r_trunc_ppf
    VERSION: 0.0.1
    AUTHOR: Yan Fu
    DATE: 27 Sept 2018
    DESCRIPTION: Produces evenly sampled random variables based on gumbel distribution (tail to the x+ direction, i.e.
    median greater than mean). Truncation is possible via variables 'a' and 'b'. i.e. inversed cumulative density
    function f(x), x will be sampled in linear space ranging from 'a' to 'b'. Then f(x) is returned. Additionally, if x
    is defined 'cdf_y' then f(cdf_y) is returned.

    PARAMETERS:
    :param a: lower bound of truncation
    :param b: upper bound of truncation
    :param n_rv: number of random variables to be sampled, equal to the length of the returned array
    :param loc: location of the distribution
    :param scale: scale of the distribution
    :param cdf_y: array ranging with range (0, 1)
    :return sampled: sampled random variables

    USAGE:
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> loc, scale = gumbel_parameter_converter(600, 180)
    >>> rv = gumbel_r_trunc_ppf(0, 1800, 10, loc, scale)
    >>> print(np.round(rv, 0))
    [   0.  408.  462.  506.  548.  594.  646.  713.  819. 1800.]

    """

    # Generate a linear spaced array inline with lower and upper boundary of log normal cumulative probability density.
    sampled_cfd = np.linspace(
        stats.gumbel_r.cdf(x=a, loc=loc, scale=scale),
        stats.gumbel_r.cdf(x=b, loc=loc, scale=scale),
        n_rv,
    )

    # Following three lines are used to check the validity of the distribution
    # print("511 (0.80): {:.4f}".format(stats.gumbel_r._y_(x=510, loc=loc, scale=scale)))
    # print("584 (0.90): {:.4f}".format(stats.gumbel_r._y_(x=584, loc=loc, scale=scale)))
    # print("655 (0.95): {:.4f}".format(stats.gumbel_r._y_(x=655, loc=loc, scale=scale)))

    # Sample log normal distribution
    sampled = stats.gumbel_r.ppf(q=sampled_cfd, loc=loc, scale=scale)

    # Work out cumulative probability function from 'sampled', output in forms of x y.
    # Interpolate x and y are processed to be capable to cope with two extreme values. y[0] (cumulative probability,
    # initial boundary) is manually set to 0.
    x = np.linspace(a, b, int(n_rv) + 1, endpoint=False)
    x += (x[1] - x[0]) / 2
    x = x[0:-2]
    y = np.array([np.sum(sampled <= i) for i in x]) / len(sampled)

    # Interpolate
    f = interp1d(y, x, bounds_error=False, fill_value=(np.min(y), np.max(y)))

    if cdf_y is None:
        return sampled
    else:
        return f(cdf_y)


def latin_hypercube_sampling(
    num_samples, num_arguments=1, sample_lbound=0, sample_ubound=1
):
    """
    NAME: latin_hypercube_sampling
    AUTHOR: Yan Fu
    VERSION: 0.1
    DATE: 3 Aug 2018
    DESCRIPTION:
    Latin Hypercube Sampling, generates an nxm array where m equal to 'num_arguments' and n equal to 'num_samples'.
    Current version only adopts 'centered' sampling mode (each sampled cell value is centered).

    PARAMETERS:
    :param num_samples: Number of samples (i.e. rows)
    :param num_arguments: Number of arguments (i.e. columns)
    :param sample_lbound: Lower sampling boundary
    :param sample_ubound: Upper sampling boundary
    :return: An array with shape (num_samples, num_arguments)

    EXAMPLE:
    >>> result = latin_hypercube_sampling(num_samples=10, num_arguments=3, sample_lbound=0, sample_ubound=0.001)
    This example yields an array with shape of (100, 3), with each column filled 100 linear spaced numbers (shuffled)
    from 1 to 0.001 (i.e. shuffled [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]).
    An example output:
    result = [
        [0.85 0.95 0.55]
        [0.55 0.45 0.05]
        [0.25 0.25 0.15]
        [0.35 0.85 0.75]
        [0.45 0.65 0.85]
        [0.75 0.05 0.65]
        [0.05 0.35 0.45]
        [0.15 0.15 0.35]
        [0.65 0.55 0.25]
        [0.95 0.75 0.95]
    ]
    """

    if sample_lbound > sample_ubound:
        sample_ubound += sample_lbound
        sample_lbound = sample_ubound - sample_lbound
        sample_ubound = sample_ubound - sample_lbound

    # Generate sorted integers with correct shape
    mat_random_num = np.linspace(
        sample_lbound, sample_ubound, num_samples + 1, dtype=float
    )
    mat_random_num += (mat_random_num[1] - mat_random_num[0]) * 0.5
    mat_random_num = mat_random_num[0:-1]
    mat_random_num = np.reshape(mat_random_num, (len(mat_random_num), 1))
    mat_random_nums = mat_random_num * np.ones((1, num_arguments))

    # np.random.shuffle(mat_random_nums)

    for i in range(np.shape(mat_random_nums)[1]):
        np.random.shuffle(mat_random_nums[:, i])

    if num_arguments == 1:
        mat_random_nums = mat_random_nums.flatten()

    return mat_random_nums


def mc_inputs_generator_worker(arg):
    kwargs, q = arg
    result = mc_inputs_generator(**kwargs)
    q.put("index")
    return result


def mc_inputs_generator(
    n_simulations,
    room_depth,
    room_opening_fraction_lbound,
    room_opening_fraction_ubound,
    room_opening_fraction_std,
    room_opening_fraction_mean,
    room_opening_permanent_fraction,
    fire_qfd_std,
    fire_qfd_mean,
    fire_qfd_ubound,
    fire_qfd_lbound,
    fire_hrr_density_std,
    fire_hrr_density_mean,
    fire_hrr_density_ubound,
    fire_hrr_density_lbound,
    fire_com_eff_lbound,
    fire_com_eff_ubound,
    fire_spread_lbound,
    fire_spread_ubound,
    fire_nft_mean,
    beam_loc_ratio_lbound,
    beam_loc_ratio_ubound,
    **_
):

    # ==================================================================================================================
    # CHECKS
    # ==================================================================================================================

    # Fire duration has to be as long as travelling fire to travel through the entire floor
    # time_end = np.max([time_end, room_depth / fire_spread_lbound])

    # ==================================================================================================================
    # Distribution variables
    # ==================================================================================================================
    df_input_samples = None

    if n_simulations == 1:

        df_input_samples = df(
            dict(
                window_open_fraction=room_opening_fraction_mean,
                fire_load_density=fire_qfd_mean,
                fire_spread_speed=(fire_spread_ubound + fire_spread_lbound) / 2,
                beam_position=room_depth * 0.75,
                fire_nft_ubound=fire_nft_mean,
                fire_hrr_density=fire_hrr_density_mean,
                index=0,
            )
        )
        df_input_samples.set_index("index", inplace=True)

    elif n_simulations == 2:

        raise ValueError("Number of simulations need to be greater than 2.")

    elif n_simulations > 2:

        def generate_samples(
            dist_type: str,
            dist_scale: float,
            dist_loc: float,
            lbound: float,
            ubound: float,
            n: int,
        ):

            cfd_q = np.linspace(
                getattr(stats, dist_type).cdf(x=lbound, loc=dist_loc, scale=dist_scale),
                getattr(stats, dist_type).cdf(x=ubound, loc=dist_loc, scale=dist_scale),
                n,
            )
            samples = getattr(stats, dist_type).ppf(
                q=cfd_q, loc=dist_loc, scale=dist_scale
            )
            samples[samples == np.inf] = ubound
            samples[samples == -np.inf] = lbound
            np.random.shuffle(samples)

            return samples

        # Fuel load density
        # -----------------
        fire_combustion_efficiency = np.linspace(
            fire_com_eff_lbound, fire_com_eff_ubound, n_simulations
        )
        qfd_loc, qfd_scale = gumbel_parameter_converter(fire_qfd_mean, fire_qfd_std)
        fire_load_density_samples = (
            generate_samples(
                "gumbel_r",
                qfd_scale,
                qfd_loc,
                fire_qfd_ubound,
                fire_qfd_lbound,
                n_simulations,
            )
            * fire_combustion_efficiency
        )

        # Fire HRR density (travelling fire)
        # --------------------------------------------------------------------------------------------------------------
        fire_hrr_density_samples = generate_samples(
            "norm",
            fire_hrr_density_std,
            fire_hrr_density_mean,
            fire_hrr_density_ubound,
            fire_hrr_density_lbound,
            n_simulations,
        )

        # Opening fraction factor (glazing fall-out fraction)
        # --------------------------------------------------------------------------------------------------------------
        opening_fraction_mean_conv, opening_fraction_std_conv = lognorm_parameters_true_to_inv(
            room_opening_fraction_mean, room_opening_fraction_std
        )
        opening_sigma = opening_fraction_std_conv
        opening_loc = 0
        openg_scale = np.exp(opening_fraction_mean_conv)
        window_open_fraction_samples = 1 - lognorm_trunc_ppf(
            room_opening_fraction_lbound,
            room_opening_fraction_ubound,
            n_simulations,
            opening_sigma,
            opening_loc,
            openg_scale,
        )
        window_open_fraction_samples[
            window_open_fraction_samples == np.inf
        ] = room_opening_fraction_ubound
        window_open_fraction_samples[
            window_open_fraction_samples == -np.inf
        ] = room_opening_fraction_lbound
        window_open_fraction_samples = (
            window_open_fraction_samples * (1 - room_opening_permanent_fraction)
            + room_opening_permanent_fraction
        )
        np.random.shuffle(window_open_fraction_samples)

        # Beam location
        # --------------------------------------------------------------------------------------------------------------
        beam_position_samples = (
            np.linspace(beam_loc_ratio_lbound, beam_loc_ratio_ubound, n_simulations)
            * room_depth
        )
        np.random.shuffle(beam_position_samples)

        # Fire spread speed (travelling fire)
        # --------------------------------------------------------------------------------------------------------------
        fire_spread_speed_samples = np.linspace(
            fire_spread_lbound, fire_spread_ubound, n_simulations
        )
        np.random.shuffle(fire_spread_speed_samples)

        # Near field temperature (travelling fire)
        # --------------------------------------------------------------------------------------------------------------
        # todo: check lower and upper limit values
        fire_nft_std = (1.939 - (np.log(fire_nft_mean) * 0.266)) * fire_nft_mean
        fire_nft_ubound_samples = generate_samples(
            "norm", fire_nft_std, fire_nft_mean, 500, 1500, n_simulations
        )

        # Summary
        # --------------------------------------------------------------------------------------------------------------
        df_input_samples = df(
            dict(
                fire_load_density=fire_load_density_samples,
                fire_spread_speed=fire_spread_speed_samples,
                fire_nft_ubound=fire_nft_ubound_samples,
                fire_hrr_density=fire_hrr_density_samples,
                fire_combustion_effeciency=fire_combustion_efficiency,
                beam_position=beam_position_samples,
                window_open_fraction=window_open_fraction_samples,
                index=np.arange(0, n_simulations, 1),
            )
        )
        df_input_samples.set_index("index", inplace=True)

    return df_input_samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pass
