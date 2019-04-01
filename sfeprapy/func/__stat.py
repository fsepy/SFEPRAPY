# __author__ = "RVC"
# __email__= "ruben.vancoile@gmail.com"
# __date__= "2017-11-12"

from copy import deepcopy
from scipy.stats import uniform
from scipy.stats import gumbel_r
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import t

import numpy as np


## inverse CDF function ##
##########################

def Finv_Uniform(r, a, b):
    return uniform.ppf(r, a, b - a)


def Finv_Gumbel(r, m, s):
    scale, loc = p_Gumbel(m, s)

    return gumbel_r.ppf(r, loc, scale)


def Finv_Normal(r, m, s):
    return norm.ppf(r, m, s)


def Finv_Lognormal(r, m, s):
    sln, mln = p_Lognormal(m, s)

    return lognorm.ppf(r, sln, 0, np.exp(mln))


def Finv_t(r, df):
    return t.ppf(r, df)


## CDF function ##
##################

def F_Normal(x, m, s):
    return norm.cdf(x, m, s)


def F_Lognormal(x, m, s):
    sln, mln = p_Lognormal(m, s)

    return lognorm.cdf(x, sln, 0, np.exp(mln))


def F_Gumbel(r, m, s):
    scale, loc = p_Gumbel(m, s)

    return gumbel_r.cdf(r, loc, scale)


## parameter calculation ##
###########################

def p_Lognormal(m, s):
    cov = s / m;

    sln = np.sqrt(np.log(1 + cov ** 2))
    mln = np.log(m) - 1 / 2 * sln ** 2

    return sln, mln


if __name__ == '__main__':
    print(p_Lognormal(0.2, 0.2))


def p_Gumbel(m, s):
    # parameters Gumbel W&S
    alpha = 1.282 / s
    u = m - 0.5772 / alpha

    # parameters Gumbel scipy
    scale = 1 / alpha
    loc = u

    return scale, loc


## 'moment' calculation ##
##########################

def m_Lognormal(mln, sln):
    cov = np.sqrt(np.exp(sln ** 2) - 1)
    m = np.exp(mln + 1 / 2 * sln ** 2)

    return m, m * cov
