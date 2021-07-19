# distribution_auto_fitting
# Author: Yan Fu
# -*- coding: utf-8 -*-
# Date: 07/03/2019

import os
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import interpolate


# Create models from data
def fit(
    data,
    bins: int = 200,
    ax=None,
    distribution_list: Union[list, int] = None,
    suppress_print: bool = False,
    fmt_str: str = "{:20.20}{:60.60}",
):
    """
    Model data by finding best fit distribution to data
    :param data:
    :param bins:
    :param ax:
    :param distribution_list:
    :param suppress_print:
    :param fmt_str:
    :return list_fitted_distribution:
    :return list_fitted_distribution_params:
    :return list_fitted_distribution_sse:
    """

    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    # ax.plot(x, y)
    # ax.fill_between(x, y, color="black")

    def cprint(*args):
        if suppress_print:
            return 0
        else:
            print(*args)

    if ax:
        ax.hist(data, bins="auto", density=True, color="black")
        ax.set_xlim(
            min(*x) - (max(*x) - min(*x)) * 0.01,
            max(*x) + (max(*x) - min(*x)) * 0.01,
            auto=False,
        )
        ax.set_ylim(
            min(*y) - (max(*y) - min(*y)) * 0.01,
            max(*y) + (max(*y) - min(*y)) * 0.01,
            auto=False,
        )

    if distribution_list == 0 or distribution_list == None:
        # full list of available distribution functions
        distribution_list = [
            st.alpha,
            st.anglit,
            st.arcsine,
            st.beta,
            st.betaprime,
            st.bradford,
            st.burr,
            st.cauchy,
            st.chi,
            st.chi2,
            st.cosine,
            st.dgamma,
            st.dweibull,
            st.erlang,
            st.expon,
            st.exponnorm,
            st.exponweib,
            st.exponpow,
            st.f,
            st.fatiguelife,
            st.fisk,
            st.foldcauchy,
            st.foldnorm,
            st.genlogistic,
            st.genpareto,
            st.gennorm,
            st.genexpon,
            st.genextreme,
            st.gausshyper,
            st.gamma,
            st.gengamma,
            st.genhalflogistic,
            st.gilbrat,
            st.gompertz,
            st.gumbel_r,
            st.gumbel_l,
            st.halfcauchy,
            st.halflogistic,
            st.halfnorm,
            st.halfgennorm,
            st.hypsecant,
            st.invgamma,
            st.invgauss,
            st.invweibull,
            st.johnsonsb,
            st.johnsonsu,
            st.ksone,
            st.kstwobign,
            st.laplace,
            st.levy,
            # st.levy_l,  # removed due to extended computing time
            # st.levy_stable,  # removed due to extended computing time
            st.logistic,
            st.loggamma,
            st.loglaplace,
            st.lognorm,
            st.lomax,
            st.maxwell,
            st.mielke,
            st.nakagami,
            st.ncx2,
            # st.ncf,  # removed due to extended computing time
            st.nct,
            st.norm,
            st.pareto,
            st.pearson3,
            st.powerlaw,
            st.powerlognorm,
            st.powernorm,
            st.rdist,
            st.reciprocal,
            st.rayleigh,
            st.rice,
            st.recipinvgauss,
            st.semicircular,
            st.t,
            st.triang,
            st.truncexpon,
            st.truncnorm,
            st.tukeylambda,
            st.uniform,
            st.vonmises,
            st.vonmises_line,
            st.wald,
            st.weibull_min,
            st.weibull_max,
            st.wrapcauchy,
        ]
    elif distribution_list == 1:
        # https://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/
        distribution_list = [
            st.uniform,
            st.bernoulli,
            st.hypergeom,
            st.binom,
            st.geom,
            st.gumbel_r,
            st.gumbel_l,
            st.nbinom,
            st.poisson,
            st.expon,
            st.lognorm,
            st.weibull_max,
            st.weibull_min,
            st.dweibull,
            st.norm,
            st.chi,
            st.gamma,
            st.beta,
        ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data

    cprint(
        fmt_str.format(
            "Distribution", "Loss (Residual sum of squares) and distribution parameters"
        )
    )
    list_fitted_distribution = []
    list_fitted_distribution_sse = []
    list_fitted_distribution_params = []

    for distribution in distribution_list:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        ax.plot(x, pdf, label=distribution.name)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

                params_ = ["{:10.2f}".format(i) for i in params]

                cprint(
                    fmt_str.format(
                        distribution.name,
                        "{:10.5E} - [{}]".format(sse, ",".join(params_)),
                    )
                )

                list_fitted_distribution.append(distribution)
                list_fitted_distribution_sse.append(sse)
                list_fitted_distribution_params.append(params)

        except Exception:
            pass

    return (
        list_fitted_distribution,
        list_fitted_distribution_params,
        list_fitted_distribution_sse,
    )


def cdf_to_samples(
    cdf_x: Union[list, np.ndarray],
    cdf_y: Union[list, np.ndarray],
    n_samples: int = 10000,
):
    """Produce a number of samples for a given cumulative density function (CDF) in x and y. The samples will be
    randomly selected.

    :param cdf_x: the x value of the CDF
    :param cdf_y: the y value of the CDF
    :param n_samples: number of samples desired
    :return samples: produced samples
    """

    cdf_func = interpolate.interp1d(cdf_y, cdf_x, kind="linear")
    cdf_pts = np.random.uniform(high=np.max(cdf_y), low=np.min(cdf_y), size=n_samples)

    samples = cdf_func(cdf_pts).flatten()

    return samples


def pdf_to_samples(
    pdf_x: Union[list, np.ndarray],
    pdf_y: Union[list, np.ndarray],
    n_samples: int = 10000,
):
    """Produce a number of samples for a given probability density function (PDF) in x and y. The samples will be
    randomly selected.

    :param pdf_x: the x value of the PDF
    :param pdf_y: the y value of the PDF
    :param n_samples: number of samples desired
    :return samples: produced samples
    """

    cdf_x = pdf_x
    cdf_y = np.cumsum(pdf_y)

    samples = cdf_to_samples(cdf_x, cdf_y, n_samples)

    return samples


def auto_fit(
        data_type: int,
        distribution_list: Union[int, list],
        data: Union[str, pd.DataFrame],
        savefig: bool = False,
):
    """
    :param data_type:
    :param distribution_list:   a csv file path string or a pandas.DataFrame object; or
                                0   fit to all available distributions
                                1   fit to common distribution types
    :param data:    a list of distribution function names as defined in scipy.stats; or integers:
                    0   single column containing sampled values.
                    1   two columns with 1st column containing values and second column probability.
                    2   two columns with 1st column containing values and second column cumulative
                                    probability.
    :return:
    """

    if isinstance(data, str):
        data = os.path.realpath(data)
        dir_work = os.path.dirname(data)
        data_csv = pd.read_csv(data, header=None, dtype=float)
    elif isinstance(data, pd.DataFrame):
        data_csv = data
        dir_work = os.getcwd()
    else:
        raise ValueError("Unknown data type.")

    # Load data
    if data_type == 0:
        samples = data_csv.values.flatten()
    elif data_type == 1:
        data_csv = pd.read_csv(data, header=None, dtype=float)
        samples = pdf_to_samples(
            pdf_x=data_csv[0].values.flatten(),
            pdf_y=data_csv[1].values.flatten(),
            n_samples=10000,
        )
    elif data_type == 2:
        data_csv = pd.read_csv(data, header=None, dtype=float)
        #     print(data_csv[1])
        samples = cdf_to_samples(
            cdf_x=data_csv[0].values.flatten(),
            cdf_y=data_csv[1].values.flatten(),
            n_samples=10000,
        )
    else:
        samples = np.nan

    # FITTING DISTRIBUTIONS TO DATA

    fig_fitting, ax_fitting = plt.subplots(figsize=(3.94 * 2.5, 2.76 * 2.5))
    list_dist, list_params, list_sse = fit(
        samples,
        ax=ax_fitting,
        distribution_list=distribution_list,
        fmt_str="{:30.30}{}",
    )

    # FINDING THE BEST FIT

    list_dist = np.asarray(list_dist)[np.argsort(list_sse)]
    list_params = np.asarray(list_params)[np.argsort(list_sse)]
    list_sse = np.asarray(list_sse)[np.argsort(list_sse)]

    print(
        "\n{:30.30}{}".format(
            "Distribution (sorted)",
            "Loss (Residual sum of squares) and distribution parameters",
        )
    )
    for i, v in enumerate(list_dist):
        dist_name = v.name
        sse = list_sse[i]
        dist_params = ", ".join(["{:10.2f}".format(j) for j in list_params[i]])
        print(f"{dist_name:30.30}{sse:10.5E} - [{dist_params}]")

    dist_best = list_dist[0]
    params_best = list_params[0]

    # PRODUCE FIGURES

    ax_fitting.set_xlabel("Sample value")
    ax_fitting.set_ylabel("PDF")
    ax_fitting.legend().set_visible(True)
    fig_fitting.savefig(os.path.join(dir_work, "distfit_fitting.png"))

    cdf_x_sampled = np.sort(samples)
    cdf_y_sampled = np.linspace(0, 1, len(cdf_x_sampled))
    cdf_y_fitted = np.linspace(0, 1, 1000)
    cdf_x_fitted = dist_best.ppf(cdf_y_fitted, *params_best)

    if savefig is True:
        fig_results, ax_results = plt.subplots(figsize=(3.94 * 2.5, 2.76 * 2.5))

        ax_results.hist(samples, bins="auto", density=True, color="black")
        ax_results.plot(cdf_x_sampled, cdf_y_sampled, label="Sampled", color="black")
        ax_results.plot(cdf_x_fitted, cdf_y_fitted, label="Fitted", color="red")
        ax_results.set_xlabel("Sample values")
        ax_results.set_ylabel("CDF")
        ax_results.legend().set_visible(True)
        fig_results.savefig(os.path.join(dir_work, "distfit_results.png"))


def _test_auto_fit():
    from scipy import stats
    auto_fit(data_type=0, distribution_list=1, data=pd.DataFrame.from_dict(dict(x=stats.norm(0, 1).rvs(1000))))


if __name__ == '__main__':
    _test_auto_fit()
