# Monte Carlo Simulation Multi-Process Implementation
# Yan Fu, October 2017

from io import StringIO
from typing import Union, Any, Type

import numpy as np

import sfeprapy.dists as dists
from sfeprapy.func.xlsx import dict_to_xlsx


class TrueToScipy:
    """Converts 'normal' distribution parameters, e.g. normal, standard deviation etc., to Scipy recognisable
    parameters, e.g. loc, scale etc.
    """

    @staticmethod
    def gumbel_r_(mean: float, sd: float, **_):
        # parameters Gumbel W&S
        alpha = 1.282 / sd
        u = mean - 0.5772 / alpha

        # parameters Gumbel scipy
        scale = 1 / alpha
        loc = u

        return dict(loc=loc, scale=scale)

    @staticmethod
    def lognorm_(mean: float, sd: float, **_):
        cov = sd / mean

        sigma_ln = np.sqrt(np.log(1 + cov ** 2))
        miu_ln = np.log(mean) - 1 / 2 * sigma_ln ** 2

        s = sigma_ln
        loc = 0
        scale = np.exp(miu_ln)

        return dict(s=s, loc=loc, scale=scale)

    @staticmethod
    def lognorm_mod_(mean: float, sd: float, **_):
        return TrueToScipy.lognorm_(mean, sd, **_)

    @staticmethod
    def norm_(mean: float, sd: float, **_):
        loc = mean
        scale = sd

        return dict(loc=loc, scale=scale)

    @staticmethod
    def uniform_(ubound: float, lbound: float, **_):
        if lbound > ubound:
            lbound += ubound
            ubound = lbound - ubound
            lbound -= ubound

        loc = lbound
        scale = ubound - lbound

        return dict(loc=loc, scale=scale)


class InputParser:
    """Converts """

    def __init__(self, dist_params: dict, n: int):
        assert isinstance(dist_params, dict)
        assert isinstance(n, int)

        self.__n = n
        self.__in_raw = dist_params
        self.__in = InputParser.unflatten_dict(dist_params)

    def to_dict(self):
        n = self.__n
        dist_params = self.__in
        dict_out = dict()

        for k, v in dist_params.items():
            if isinstance(v, float) or isinstance(v, int) or isinstance(v, float):
                dict_out[k] = np.full((n,), v, dtype=float)
            elif isinstance(v, str):
                dict_out[k] = np.full(
                    (n,), v, dtype=np.dtype("U{:d}".format(len(v)))
                )
            elif isinstance(v, np.ndarray) or isinstance(v, list):
                dict_out[k] = list(np.full((n, len(v)), v, dtype=float))
            elif isinstance(v, dict):
                if "dist" in v:
                    try:
                        dict_out[k] = InputParser._sampling(v, n)
                    except KeyError:
                        raise KeyError(f"Missing parameters in input variable {k}.")
                elif "ramp" in v:
                    s_ = StringIO(v["ramp"])
                    d_ = np.loadtxt(s_, delimiter=',')
                    t_ = d_[:, 0]
                    v_ = d_[:, 1]
                    if all(v_ == v_[0]):
                        f_interp = v_[0]
                    else:
                        def f_interp(x):
                            return np.interp(x, t_, v_)
                    dict_out[k] = np.full((n,), f_interp)
                else:
                    raise ValueError(f"Unknown input data type for {k}. {v}.")
            elif v is None:
                dict_out[k] = np.full((n,), np.nan, dtype=float)
            else:
                raise TypeError(f"Unknown input data type for {k}.")

        dict_out["index"] = np.arange(0, n, 1)
        return dict_out

    def to_xlsx(self, fp: str):
        dict_to_xlsx({i: InputParser.flatten_dict(v) for i, v in self.to_dict().items()}, fp)

    @staticmethod
    def unflatten_dict(dict_in: dict) -> dict:
        """Invert flatten_dict.

        :param dict_in:
        :return dict_out:
        """
        dict_out = dict()

        for k, v in dict_in.items():
            InputParser.__unflatten_dict(k, v, dict_out)

        return dict_out

    @staticmethod
    def __unflatten_dict(k: str, v: Any, dict_out: dict):
        if ":" in k:
            k1, *k2 = k.split(':')
            if k1 not in dict_out:
                dict_out[k1] = dict()
            InputParser.__unflatten_dict(':'.join(k2), v, dict_out[k1])
        else:
            dict_out[k] = v

    @staticmethod
    def flatten_dict(dict_in: dict) -> dict:
        dict_out = dict()
        InputParser.__flatten_dict(dict_in, dict_out)
        return dict_out

    @staticmethod
    def __flatten_dict(dict_in: dict, dict_out: dict, history: str = None):
        """Converts two levels dict to single level dict. Example input and output see _test_dict_flatten.
        >>> dict_in = {
        >>>             'a': 1,
        >>>             'b': {'b1': 21, 'b2': 22},
        >>>             'c': {'c1': 31, 'c2': 32, 'c3': 33}
        >>>         }
        >>> output = {
        >>>             'a': 1,
        >>>             'b:b1': 21,
        >>>             'b:b2': 22,
        >>>             'c:c1': 31,
        >>>             'c:c2': 32,
        >>>             'c:c3': 33,
        >>>         }
        >>> assert InputParser.flatten_dict(dict_in) == output  # True

        :param dict_in:     Any two levels (or less) dict.
        :return dict_out:   Single level dict.
        """
        for k, v in dict_in.items():
            if isinstance(v, dict):
                InputParser.__flatten_dict(v, dict_out=dict_out, history=k if history is None else f'{history}:{k}')
            else:
                dict_out[f'{k}' if history is None else f'{history}:{k}'] = v

    @staticmethod
    def _sampling(dist_params: dict, num_samples: int, randomise: bool = True) -> Union[float, np.ndarray]:
        """A reimplementation of _sampling_scipy but without scipy"""
        dist_name = ''.join(dist_params.pop('dist').replace('_', ' ').strip().title().split())
        if dist_name == 'Norm':
            dist_name = 'Normal'
        elif dist_name == 'GumbelR':
            dist_name = 'Gumbel'
        elif dist_name == 'Uniform':
            if (
                    'lbound' in dist_params and 'ubound' in dist_params and
                    'mean' not in dist_params and 'sd' not in dist_params
            ):
                a = dist_params.pop('lbound')
                b = dist_params.pop('ubound')
                mean = (a + b) / 2
                sd = (b - a) / (2 * np.sqrt(3))
                dist_params['mean'] = mean
                dist_params['sd'] = sd
        elif dist_name == 'LognormMod':
            dist_name = 'LognormalMod'
        elif dist_name == 'Lognorm':
            dist_name = 'Lognormal'
        elif dist_name == 'Constant':
            if 'ubound' in dist_params and 'lbound' in dist_params:
                dist_params['value'] = (dist_params.pop('lbound') + dist_params.pop('ubound')) / 2.

        dist_cls: Type[dists.DistFunc] = getattr(dists, dist_name)
        dist_obj: dists.DistFunc = dist_cls(**dist_params)
        lim_1 = None if 'lbound' not in dist_params else dist_params['lbound']
        lim_2 = None if 'ubound' not in dist_params else dist_params['ubound']
        return dist_obj.sampling(n=num_samples, lim_1=lim_1, lim_2=lim_2, shuffle=randomise)

    @staticmethod
    def _sampling_scipy(dist_params: dict, num_samples: int, randomise: bool = True) -> Union[float, np.ndarray]:
        """Evacuate sampled values based on a defined distribution. This is build upon `scipy.stats` library.

        :param dist_params: Distribution inputs, required keys are distribution dependent, should be aligned with inputs
                            required in the scipy.stats. Additional compulsory keys are:
                                `dist`: str, distribution type.
        :param num_samples: Number of samples to be generated.
        :param randomise:   Whether to randomise the sampled values.
        :return samples:    Sampled values based upon `dist` in the range [`lbound`, `ubound`] with `num_samples` number
                            of values.
        """
        import scipy.stats as stats

        if dist_params['dist'] == 'discrete_':
            v_ = dist_params['values']
            if isinstance(v_, str):
                assert ',' in v_, f'`discrete_ distribution `values` parameter is not a list separated by comma.'
                v_ = [float(i.strip()) for i in v_.split(',')]

            w_ = dist_params['weights']
            if isinstance(w_, str):
                assert ',' in w_, f'`discrete_`:`weights` is not a list of numbers separated by comma.'
                w_ = [float(i.strip()) for i in w_.split(',')]

            assert len(v_) == len(w_), f'Length of values ({len(v_)}) and weights ({len(v_)}) do not match.'
            assert sum(w_) == 1., f'Sum of all weights should be unity, got {sum(w_)}.'

            w_ = [int(round(i * num_samples)) for i in w_]
            if (sum_sampled := sum(w_)) < num_samples:
                for i in np.random.choice(np.arange(len(w_)), size=sum_sampled - num_samples):
                    w_[i] += 1
            elif sum_sampled > num_samples:
                for i in np.random.choice(np.arange(len(w_)), size=sum_sampled - num_samples):
                    w_[i] -= 1
            w_ = np.cumsum(w_)
            assert w_[-1] == num_samples, f'Total weight length does not match `num_samples`.'
            samples = np.empty((num_samples,), dtype=float)
            for i, v__ in enumerate((v_)):
                if i == 0:
                    samples[0:w_[i]] = v__
                else:
                    samples[w_[i - 1]:w_[i]] = v__

            if randomise:
                np.random.shuffle(samples)

            return samples

        if dist_params['dist'] == 'constant_':
            return np.full((num_samples,), (dist_params['lbound'] + dist_params['ubound']) / 2, dtype=float)

        # sample CDF points (y-axis value)
        def generate_cfd_q(dist, dist_params_scipy, lbound, ubound, num_samples_=None):
            num_samples_ = num_samples if num_samples_ is None else num_samples_
            cfd_q_ = np.linspace(
                getattr(stats, dist).cdf(x=lbound, **dist_params_scipy),
                getattr(stats, dist).cdf(x=ubound, **dist_params_scipy),
                num_samples_,
            )
            samples_ = getattr(stats, dist).ppf(q=cfd_q_, **dist_params_scipy)
            return samples_

        # convert true distribution parameters to scipy distribution parameters
        try:
            if dist_params['dist'] == 'lognorm_mod_':
                dist_params_scipy = getattr(TrueToScipy, 'lognorm_')(
                    **dist_params
                )
                samples = generate_cfd_q(
                    dist='lognorm', dist_params_scipy=dist_params_scipy, lbound=dist_params['lbound'],
                    ubound=dist_params['ubound']
                )
                samples = 1 - samples
            elif dist_params['dist'] == 'br187_fuel_load_density_':
                dist_params_list = list()
                dist_params_list.append(
                    dict(dist='gumbel_r_', lbound=dist_params['lbound'], ubound=dist_params['ubound'], mean=780,
                         sd=234))
                dist_params_list.append(
                    dict(dist='gumbel_r_', lbound=dist_params['lbound'], ubound=dist_params['ubound'], mean=420,
                         sd=126))
                samples_ = list()
                for dist_params in dist_params_list:
                    dist_params_scipy = getattr(TrueToScipy, dist_params['dist'])(**dist_params)
                    samples__ = generate_cfd_q(
                        dist=dist_params['dist'].rstrip('_'), dist_params_scipy=dist_params_scipy,
                        lbound=dist_params['lbound'], ubound=dist_params['ubound']
                    )
                    samples_.append(samples__)
                samples = np.random.choice(np.append(*samples_), num_samples, replace=False)
            elif dist_params['dist'] == 'br187_hrr_density_':
                dist_params_list = list()
                dist_params_list.append(dict(dist='uniform_', lbound=0.32, ubound=0.57))
                dist_params_list.append(dict(dist='uniform_', lbound=0.15, ubound=0.65))
                samples_ = list()
                for dist_params in dist_params_list:
                    dist_params_scipy = getattr(TrueToScipy, dist_params['dist'])(**dist_params)
                    samples__ = generate_cfd_q(
                        dist=dist_params['dist'].rstrip('_'), dist_params_scipy=dist_params_scipy,
                        lbound=dist_params['lbound'], ubound=dist_params['ubound']
                    )
                    samples_.append(samples__)
                samples = np.random.choice(np.append(*samples_), num_samples, replace=False)
            else:
                dist_params_scipy = getattr(TrueToScipy, dist_params['dist'])(**dist_params)
                samples = generate_cfd_q(
                    dist=dist_params['dist'].rstrip('_'), dist_params_scipy=dist_params_scipy,
                    lbound=dist_params['lbound'], ubound=dist_params['ubound']
                )

        except Exception as e:
            try:
                samples = generate_cfd_q(
                    dist=dist_params['dist'], dist_params_scipy=dist_params, lbound=dist_params['lbound'],
                    ubound=dist_params['ubound']
                )
            except AttributeError:
                raise ValueError(f"Unknown distribution type {dist_params['dist']}, {e}")

        samples[samples == np.inf] = dist_params['ubound']
        samples[samples == -np.inf] = dist_params['lbound']

        if "permanent" in dist_params:
            samples += dist_params["permanent"]

        if randomise:
            np.random.shuffle(samples)

        return samples
