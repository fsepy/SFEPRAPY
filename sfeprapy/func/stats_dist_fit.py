# distribution_auto_fitting
# Author: Yan Fu
# Date 07/03/2019

import os
import warnings
from tkinter import filedialog, Tk, StringVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import interpolate


# Create models from data
def fit(data, bins=200, ax=None, distribution_list=None):
    """Model data by finding best fit distribution to data"""

    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    # ax.plot(x, y)
    # ax.fill_between(x, y, color="black")
    ax.hist(data, bins='auto', density=True, color='black')
    ax.set_xlim(min(*x) - (max(*x) - min(*x)) * 0.01, max(*x) + (max(*x) - min(*x)) * 0.01, auto=False)
    ax.set_ylim(min(*y) - (max(*y) - min(*y)) * 0.01, max(*y) + (max(*y) - min(*y)) * 0.01, auto=False)

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

    print('{:20.20}{:60.60}'.format('Distribution', 'Loss (Residual sum of squares) and distribution parameters'))
    list_fitted_distribution = []
    list_fitted_distribution_sse = []
    list_fitted_distribution_params = []

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
                        ax.plot(x, pdf, label=distribution.name)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

                params_ = ["{:10.2f}".format(i) for i in params]
                print('{:20.20}{:10.5E} - [{}]'.format(distribution.name, sse, ", ".join(params_)))

                list_fitted_distribution.append(distribution)
                list_fitted_distribution_sse.append(sse)
                list_fitted_distribution_params.append(params)

        except Exception:
            pass

    return list_fitted_distribution, list_fitted_distribution_params, list_fitted_distribution_sse


def cdf_to_samples(cdf_x, cdf_y, n_samples=10000):
    cdf_func = interpolate.interp1d(cdf_y, cdf_x, kind='linear')
    cdf_pts = np.random.uniform(size=n_samples)
    return cdf_func(cdf_pts).flatten()


def pdf_to_samples(pdf_x, pdf_y, n_samples=10000):
    cdf_x = pdf_x
    cdf_y = np.cumsum(pdf_y)

    return cdf_to_samples(cdf_x, cdf_y, n_samples)


def auto_fit():
    # INPUTS

    # Note 1: *.csv must contain no headers and numbers only.

    # Define data type
    # 0 - single column containing sampled values
    # 1 - two columns with 1st column containing values and second column probability
    # 2 - two columns with 1st column containing values and second column cumulative probability
    data_type = input(
        "Define Data Type\n0 - single column containing sampled values\n1 - two columns with 1st column containing values and second column probability\n2 - two columns with 1st column containing values and second column cumulative probability\nEnter your value and press Enter: ")
    data_type = int(data_type)

    # Distribution type or list
    # list - will only fit to the provided list of distributions (scipy.stats.*)
    # 0 - fit to all available distributions
    # 1 - fit to common distribution types
    distribution_list = input(
        "Distribution Type\n0 - fit to all available distributions\n1 - fit to common distribution types\nEnter your value and press Enter: ")
    distribution_list = int(distribution_list)

    # Select data csv file
    root = Tk()
    root.withdraw()
    folder_path = StringVar()
    path_input_file_csv = filedialog.askopenfile(title='Select data csv file', filetypes=[('csv', ['.csv'])])
    folder_path.set(path_input_file_csv)
    root.update()

    dir_work = os.path.dirname(path_input_file_csv.name)
    print(dir_work)

    # Load data
    if data_type == 0:
        data_csv = pd.read_csv(path_input_file_csv, header=None, dtype=float)
        samples = data_csv.values.flatten()
    elif data_type == 1:
        data_csv = pd.read_csv(path_input_file_csv, header=None, dtype=float)
        samples = pdf_to_samples(pdf_x=data_csv[0].values.flatten(), pdf_y=data_csv[1].values.flatten(),
                                 n_samples=10000)
    elif data_type == 2:
        data_csv = pd.read_csv(path_input_file_csv, header=None, dtype=float)
        #     print(data_csv[1])
        samples = cdf_to_samples(cdf_x=data_csv[0].values.flatten(), cdf_y=data_csv[1].values.flatten(),
                                 n_samples=10000)
    else:
        samples = np.nan

    # FITTING DISTRIBUTIONS TO DATA

    fig_fitting, ax_fitting = plt.subplots(figsize=(3.94 * 2.5, 2.76 * 2.5))
    list_dist, list_params, list_sse = fit(samples, ax=ax_fitting, distribution_list=distribution_list)

    # FINDING THE BEST FIT

    list_dist = np.asarray(list_dist)[np.argsort(list_sse)]
    list_params = np.asarray(list_params)[np.argsort(list_sse)]
    list_sse = np.asarray(list_sse)[np.argsort(list_sse)]
    print('{:30.30}{:60.60}'.format('Distribution (sorted)',
                                    'Loss (Residual sum of squares) and distribution parameters'))
    for i, v in enumerate(list_dist):
        print('{:30.30}{:10.5E} - [{}]'.format(v.name, list_sse[i],
                                               ", ".join(["{:10.2f}".format(i) for j in list_params[i]])))

    dist_best = list_dist[0]
    params_best = list_params[0]

    # PRODUCE FIGURES

    print("Figures saved at", dir_work)

    ax_fitting.set_xlabel("Sample value")
    ax_fitting.set_ylabel("PDF")
    ax_fitting.legend().set_visible(True)
    fig_fitting.savefig(os.path.join(dir_work, "fitting.png"))

    cdf_x_sampled = np.sort(samples)
    cdf_y_sampled = np.linspace(0, 1, len(cdf_x_sampled))
    cdf_y_fitted = np.linspace(0, 1, 1000)
    cdf_x_fitted = dist_best.ppf(cdf_y_fitted, *params_best)

    fig_results, ax_results = plt.subplots(figsize=(3.94 * 2.5, 2.76 * 2.5))
    ax_results.hist(samples, bins='auto', density=True, color='black')
    ax_results.plot(cdf_x_sampled, cdf_y_sampled, label='Sampled', color='black')
    ax_results.plot(cdf_x_fitted, cdf_y_fitted, label='Fitted', color='red')
    ax_results.set_xlabel('Sample values')
    ax_results.set_ylabel('CDF')
    ax_results.legend().set_visible(True)
    fig_results.savefig(os.path.join(dir_work, "results.png"))

    input("Press any key to finish.")


if __name__ == '__main__':
    data_cdf_example = [[226.3874615,0],
        [229.4193217,0.001362959],
        [230.4470709,0.002016362],
        [231.4748201,0.002084425],
        [232.5539568,0.00381152],
        [233.4789311,0.002217148],
        [234.2497431,0.005579453],
        [235.3802672,0.008965581],
        [236.4594039,0.014003934],
        [237.2302158,0.019021869],
        [238.0524152,0.025698835],
        [238.9260021,0.032379205],
        [239.8509764,0.044029866],
        [240.8273381,0.057339559],
        [241.5981501,0.073946897],
        [242.5745118,0.08725659],
        [243.036999,0.10384351],
        [243.8591984,0.125421139],
        [244.4758479,0.143673897],
        [245.04111,0.163578881],
        [245.709147,0.183490672],
        [246.3257965,0.20174343],
        [246.7368962,0.226605092],
        [247.3021583,0.251476964],
        [247.7646454,0.276342029],
        [248.2271326,0.304518353],
        [248.6896197,0.329383419],
        [249.2034943,0.355907516],
        [249.5632066,0.380765775],
        [250.1284687,0.403982018],
        [250.6937307,0.428853889],
        [251.1048304,0.458682439],
        [251.5673176,0.483547504],
        [252.0811922,0.506760344],
        [252.5950668,0.531628812],
        [253.1603289,0.561467571],
        [253.5714286,0.583017975],
        [254.1880781,0.611204508],
        [254.5991778,0.627788025],
        [255.2158273,0.659285816],
        [255.3186023,0.677504543],
        [255.626927,0.689114366],
        [256.1921891,0.718953125],
        [256.5519013,0.735533238],
        [257.3741007,0.768700272],
        [257.7338129,0.778657868],
        [258.3504625,0.813466918],
        [258.6587873,0.833354887],
        [259.4809866,0.858243774],
        [260.2004111,0.87319208],
        [260.8684481,0.899726387],
        [261.2795478,0.909687387],
        [262.3586845,0.93459329],
        [262.8725591,0.944561097],
        [263.7461459,0.956208354],
        [264.5169579,0.967848805],
        [265.4933196,0.971224723],
        [266.1613566,0.977891481],
        [267.3946557,0.991218189],
        [268.5251799,0.994604317],
        [270.1181912,0.999676701],
        [275.2569373,1]]

    
    auto_fit()
