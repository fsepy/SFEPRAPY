# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Union


def gumbel_r(mean: float, sd: float, **_):

    # parameters Gumbel W&S
    alpha = 1.282 / sd
    u = mean - 0.5772 / alpha

    # parameters Gumbel scipy
    scale = 1 / alpha
    loc = u

    return dict(loc=loc, scale=scale)


def lognorm(mean: float, sd: float, **_):

    cov = sd / mean

    sigma_ln = np.sqrt(np.log(1 + cov ** 2))
    miu_ln = np.log(mean) - 1 / 2 * sigma_ln ** 2

    s = sigma_ln
    loc = 0
    scale = np.exp(miu_ln)

    return dict(s=s, loc=loc, scale=scale)


def norm(mean: float, sd: float, **_):

    loc = mean
    scale = sd

    return dict(loc=loc, scale=scale)


def uniform(ubound: float, lbound: float, **_):

    if lbound > ubound:
        lbound += ubound
        ubound = lbound - ubound
        lbound -= ubound

    loc = lbound
    scale = ubound - lbound

    return dict(loc=loc, scale=scale)


def random_variable_generator(dict_in: dict, num_samples: int):

    # assign distribution type
    dist = dict_in['dist']

    # assign distribution boundary (for samples)
    ubound = dict_in['ubound']
    lbound = dict_in['lbound']

    # convert human distribution parameters to scipy distribution parameters
    dist_kw = dict()
    if dist is 'gumbel_r':
        dist_kw = gumbel_r(**dict_in)
    elif dist is 'uniform':
        dist_kw = uniform(**dict_in)
    elif dist is 'norm':
        dist_kw = norm(**dict_in)
    elif dist is 'lognorm_mod':
        dist_kw = lognorm(**dict_in)
    else:
        raise ValueError('Unknown distribution type {}.'.format(dist))

    # sample CDF points (y-axis value)
    if dist is 'lognorm_mod':
        cfd_q = np.linspace(
            getattr(stats)
        )
        samples = 0
    else:
        cfd_q = np.linspace(
            getattr(stats, dist).cdf(x=lbound, **dist_kw),
            getattr(stats, dist).cdf(x=ubound, **dist_kw),
            num_samples)
        samples = getattr(stats, dist).ppf(q=cfd_q, **dist_kw)

    samples[samples == np.inf] = ubound
    samples[samples == -np.inf] = lbound

    np.random.shuffle(samples)

    return samples


def dict_unflatten(dict_in: dict):

    dict_out = dict()

    for k in list(dict_in.keys()):
        if ':' in k:
            k1, k2 = k.split(':')

            if k1 in dict_out:
                dict_out[k1][k2] = dict_in[k]
            else:
                dict_out[k1] = dict(k2 = dict_in[k])


def main(x: dict, num_samples: int):

    dict_out = dict()

    for k, v in x.items():

        if isinstance(v, float) or isinstance(v, int) or isinstance(v, np.float):
            dict_out[k] = np.full((num_samples,), v, dtype=float)

        elif isinstance(v, str):
            dict_out[k] = np.full((num_samples,), v, dtype=np.dtype('U{:d}'.format(len(v))))

        elif isinstance(v, dict):
            if 'dist' in v:
                dict_out[k] = random_variable_generator(v, num_samples)
            else:
                raise ValueError('Unknown input data type for {}.'.format(k))
        else:
            raise TypeError('Unknown input data type for {}.'.format(k))

    dict_out['index'] = np.arange(0, num_samples, 1)

    df_out = pd.DataFrame.from_dict(dict_out, orient='columns')

    return df_out


def _test_random_variable_generator():

    dict_in = dict(
        v1=np.pi,
        v2='hello world.',
        v3=dict(
            dist='uniform',
            ubound=10,
            lbound=-1
        ),
        v4=dict(
            dist='norm',
            ubound=5+1,
            lbound=5-1,
            mean=5,
            sd=1
        ),
        v5=dict(
            dist='gumbel_r',
            ubound=2500,
            lbound=50,
            mean=420,
            sd=126
        ),
        v6=dict(
            dist='lognorm_mod',
            ubound=1,
            lbound=0,
            mean=0.5,
            sd=1,
        )
    )

    df_out = main(dict_in, 10000)

    print(df_out)


if __name__ == '__main__':
    _test_random_variable_generator()
