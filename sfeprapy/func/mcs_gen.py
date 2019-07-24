# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as stats


def gumbel_r_(mean: float, sd: float, **_):

    # parameters Gumbel W&S
    alpha = 1.282 / sd
    u = mean - 0.5772 / alpha

    # parameters Gumbel scipy
    scale = 1 / alpha
    loc = u

    return dict(loc=loc, scale=scale)


def lognorm_(mean: float, sd: float, **_):

    cov = sd / mean

    sigma_ln = np.sqrt(np.log(1 + cov ** 2))
    miu_ln = np.log(mean) - 1 / 2 * sigma_ln ** 2

    s = sigma_ln
    loc = 0
    scale = np.exp(miu_ln)

    return dict(s=s, loc=loc, scale=scale)


def norm_(mean: float, sd: float, **_):

    loc = mean
    scale = sd

    return dict(loc=loc, scale=scale)


def uniform_(ubound: float, lbound: float, **_):

    if lbound > ubound:
        lbound += ubound
        ubound = lbound - ubound
        lbound -= ubound

    loc = lbound
    scale = ubound - lbound

    return dict(loc=loc, scale=scale)


def random_variable_generator(dict_in: dict, num_samples: int):

    # assign distribution type
    dist_0 = dict_in['dist']
    dist = dict_in['dist']

    # assign distribution boundary (for samples)
    ubound = dict_in['ubound']
    lbound = dict_in['lbound']

    # sample CDF points (y-axis value)
    def generate_cfd_q(dist_, dist_kw_, lbound_, ubound_):
        cfd_q_ = np.linspace(
            getattr(stats, dist_).cdf(x=lbound_, **dist_kw_),
            getattr(stats, dist_).cdf(x=ubound_, **dist_kw_),
            num_samples)
        samples_ = getattr(stats, dist_).ppf(q=cfd_q_, **dist_kw_)
        return samples_

    # convert human distribution parameters to scipy distribution parameters
    if dist_0 == 'gumbel_r_':
        dist_kw = gumbel_r_(**dict_in)
        dist = 'gumbel_r'
        samples = generate_cfd_q(dist_=dist, dist_kw_=dist_kw, lbound_=lbound, ubound_=ubound)
    elif dist_0 == 'uniform_':
        dist_kw = uniform_(**dict_in)
        dist = 'uniform'
        samples = generate_cfd_q(dist_=dist, dist_kw_=dist_kw, lbound_=lbound, ubound_=ubound)

    elif dist_0 == 'norm_':
        dist_kw = norm_(**dict_in)
        dist = 'norm'
        samples = generate_cfd_q(dist_=dist, dist_kw_=dist_kw, lbound_=lbound, ubound_=ubound)

    elif dist_0 == 'lognorm_':
        dist_kw = lognorm_(**dict_in)
        dist = 'lognorm'
        samples = generate_cfd_q(dist_=dist, dist_kw_=dist_kw, lbound_=lbound, ubound_=ubound)

    elif dist_0 == 'lognorm_mod_':
        dist_kw = lognorm_(**dict_in)
        dist = 'lognorm'
        samples = generate_cfd_q(dist_=dist, dist_kw_=dist_kw, lbound_=lbound, ubound_=ubound)
        samples = 1 - samples

    else:
        raise ValueError('Unknown distribution type {}.'.format(dist))

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


def dict_flatten(dict_in: dict):

    dict_out = dict()

    for k in list(dict_in.keys()):
        if isinstance(dict_in[k], dict):
            for kk, vv in dict_in[k].items():
                dict_out[f'{k}:{kk}'] = vv
        else:
            dict_out[k] = dict_in[k]
            
    return dict_in


def main(x: dict, num_samples: int):

    dict_out = dict()

    for k, v in x.items():

        if isinstance(v, float) or isinstance(v, int) or isinstance(v, np.float):
            dict_out[k] = np.full((num_samples,), v, dtype=float)

        elif isinstance(v, str):
            dict_out[k] = np.full((num_samples,), v, dtype=np.dtype('U{:d}'.format(len(v))))
        
        elif isinstance(v, np.ndarray) or isinstance(v, list):
            dict_out[k] = list(np.full((num_samples, len(v)), v, dtype=float))

        elif isinstance(v, dict):
            if 'dist' in v:
                try:
                    dict_out[k] = random_variable_generator(v, num_samples)
                except KeyError:
                    raise('Missing parameters in input variable {}.'.format(k))
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
        v3=np.array([0, 1, 2]),
        v4=dict(
            dist='uniform_',
            ubound=10,
            lbound=-1
        ),
        v5=dict(
            dist='norm_',
            ubound=5+1,
            lbound=5-1,
            mean=5,
            sd=1
        ),
        v6=dict(
            dist='gumbel_r_',
            ubound=2500,
            lbound=50,
            mean=420,
            sd=126
        ),
        v7=dict(
            dist='lognorm_',
            ubound=1,
            lbound=0,
            mean=0.5,
            sd=1,
        ),
        v8=dict(
            dist='lognorm_mod_',
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
