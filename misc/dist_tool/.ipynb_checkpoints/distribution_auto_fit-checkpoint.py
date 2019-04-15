# distribution_auto_fitting
# Author: Yan Fu
# Date 07/03/2019

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, StringVar


# Create models from data
def best_fit_distribution(data, bins=200, ax=None, distribution_list=None):
    """Model data by finding best fit distribution to data"""

    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    # ax.plot(x, y)
    ax.fill_between(x, y, color="black")
    ax.set_xlim(min(*x), max(*x), auto=False)
    ax.set_ylim(min(*y), max(*y), auto=False)

    if distribution_list == 0 or distribution_list == None:
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
            st.frechet_r,
            st.frechet_l,
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
            st.levy_l,
            st.levy_stable,
            st.logistic,
            st.loggamma,
            st.loglaplace,
            st.lognorm,
            st.lomax,
            st.maxwell,
            st.mielke,
            st.nakagami,
            st.ncx2,
            st.ncf,
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
            st.wrapcauchy
        ]
    elif distribution_list == 1:
        distribution_list = [
            st.alpha,
            st.beta,
            st.chi,
            st.chi2,
            st.cosine,
            st.dweibull,
            st.expon,
            st.exponnorm,
            st.exponweib,
            st.exponpow,
            st.gamma,
            st.gumbel_r,
            st.gumbel_l,
            st.logistic,
            st.lognorm,
            st.maxwell,
            st.norm,
            st.powerlaw,
            st.rdist,
            st.truncexpon,
            st.truncnorm,
            st.uniform,
        ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data

    print('{:30}{:30}'.format('Distribution', 'Loss (Residual sum of squares)'))

    for distribution in distribution_list:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

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
                        ax.plot(x, pdf, alpha=0.6, label=distribution.name)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
                
                params_ = ["{:.5f}".format(i) for i in params]
                print('{:30}{:.5E} - [{}]'.format(distribution.name, sse, ", ".join(params_)))

        except Exception:
            pass

    return best_distribution, best_params


def run():

    # Select data csv file
    root = Tk()
    root.withdraw()
    folder_path = StringVar()
    path_input_file_csv = filedialog.askopenfile(title='Select data csv file', filetypes=[('csv', ['.csv'])])
    folder_path.set(path_input_file_csv)
    root.update()

    # Read data
    samples = pd.read_csv(path_input_file_csv)
    samples = samples.values.flatten()

    # Create figures
    fig_fitting, ax_fitting = plt.subplots(figsize=(3.94*2, 2.76*2))
    fig_final, ax_final = plt.subplots(figsize=(3.94*2, 2.76*2))

    ax_fitting.set_xlabel('Sample values')
    ax_fitting.set_ylabel('PDF')

    ax_final.set_xlabel('Sample values')
    ax_final.set_ylabel('CDF')

    # Fit data
    dist, params = best_fit_distribution(samples, ax=ax_fitting, distribution_list=1)
    print("Best: {}, Params: {}".format(dist.name, str(params)))
    
    # Make plots to compare fitted and actual
    cdf_x_sampled = np.sort(samples)
    cdf_y_sampled = np.linspace(0, 1, len(cdf_x_sampled))
    cdf_y_fitted = np.linspace(0,1,1000)
    cdf_x_fitted = dist.ppf(cdf_y_fitted, *params)

    ax_final.plot(cdf_x_fitted, cdf_y_fitted, label='Fitted')
    ax_final.plot(cdf_x_sampled, cdf_y_sampled, label='Sampled')

    # ax_fitting.legend().set_visible(True)
    ax_fitting.legned().set_visible(True)
    ax_final.legend().set_visible(True)

    # Save figures
    fig_fitting.savefig("fitting.png")
    fig_final.savefig("final.png")


if __name__ == '__main__':
    
    run()
