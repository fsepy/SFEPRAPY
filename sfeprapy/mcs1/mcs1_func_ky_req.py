# __author__ = "Danny Hopkin"
# __email__= "Danny.Hopkin@OFRconsultants.com"
# __date__= "02-05-19"

import numpy as np
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import norm


## inverse CDF function ##


def Finv_Normal(r, m, s):
    return norm.ppf(r, m, s)


def Finv_Lognormal(r, m, s):
    sln, mln = p_Lognormal(m, s)

    return lognorm.ppf(r, sln, 0, np.exp(mln))


## parameter calculation ##


def p_Lognormal(m, s):
    cov = s / m

    sln = np.sqrt(np.log(1 + cov ** 2))
    mln = np.log(m) - 1 / 2 * sln ** 2

    return sln, mln


def p_Gamma(m, s):
    # --------------------------------------------------------------------------------------------------------
    # X ~ Gamma distribution
    # Input:m,s
    # Intermediate calc: k,lambda cfr. W&S p.5.8
    # Output: scale, loc cfr. scipy
    # --------------------------------------------------------------------------------------------------------
    # parameters Gamma W&S
    l = m / s ** 2
    k = m * l
    # parameters Gamma scipy
    a = k
    scale = 1 / l
    return a, scale


def ky_req_calc(n_samples):
    ########### Testing #####################

    # Starting section utilisation

    u = 1.0  # Section utilisation between 0 and 1.0 based on ULS ambient design

    # Live load parameters

    NomQ = 3  # Nominal instantaneous imposed load (Q) [kN/sq.m]
    Vimp = 0.95  # Instantaneous imposed load COV
    Mimp = 0.2 * NomQ  # Mean instantaneous imposed load [kN/sq.m]
    Simp = Vimp * Mimp  # St. Dev of instantaneous imposed load [kN/sq.m]
    GammaQ = 1.5  # ULS ambient partial factor on Q
    a, scale = p_Gamma(Mimp, Simp)  # Get distribution parameters

    # Permanent load parameters

    NomP = 3  # Nominal permanent load (G) [kN/sq.m]
    Vper = 0.1  # Permanent load (G) COV
    Mper = 1.0 * NomP  # Mean permanent load (G) [kN/sq.m]
    GammaG = 1.35  # ULS ambient partial factor on G
    Sper = Vper * Mper  # St. Dev of permanent load [kN/sq.m]

    # Model uncertainty parameters

    Mk = 1.0  # Mean total load model uncertainty [-]
    Vk = 0.1  # COV for total model uncertainty [-]
    Sk = Vk * Mk  # Standard dev of total model uncertainty [-]

    # Generate random variables

    samples = n_samples  # Number of samples for combined load distribution

    xT = np.linspace(0, 1, samples)  # probability array for plotting
    xQ = np.random.rand(samples)  # random number array for Q
    xG = np.random.rand(samples)  # random number array for G
    xK = np.random.rand(samples)  # random number array for KE

    Qp = gamma.ppf(xQ, a, loc=0, scale=scale)  # Calculate imposed load dist
    Gp = norm.ppf(xG, Mper, Sper)  # Calculate permanent load dist
    Kp = Finv_Lognormal(xK, Mk, Sk)  # Calculate model uncertainty factor
    Tp = Kp * (Gp + Qp)  # Total load model
    Tpscale = (u * Tp) / (
        (GammaG * NomP) + (GammaQ * NomQ)
    )  # Normalisation relative to ambient ULS
    # Tpscale = np.sort(Tpscale)  # Sort for CDF
    # weights = np.ones_like(Tpscale) / float(len(Tpscale))  # Calculate weights for histogram

    return Tpscale

    # Plot results

    # plt.figure(1)
    # plt.subplot(211)
    # plt.xlabel('ky,req [-]')
    # plt.ylabel('CDF [-]')
    # plt.xlim([0, 1.0])
    # plt.ylim([0, 1.0])
    # plt.plot(Tpscale, xT, "y-", label=(r'$\alpha=29, \beta=3$'))
    # plt.subplot(212)
    # plt.xlabel('ky,req [-]')
    # plt.ylabel('PDF [-]')
    # plt.xlim([0, 1.0])
    # plt.hist(Tpscale, bins=200, weights=weights)
    # plt.show()

    # mcs_out = np.array(Tpscale)
    # np.savetxt('test.csv', mcs_out)


if __name__ == "__main__":
    res = ky_req_calc(5000)
    np.savetxt("kyr.csv", res)
