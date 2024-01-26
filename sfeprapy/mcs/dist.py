from abc import ABC, abstractmethod
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


class DistSampler:
    """Converts """

    def __init__(self, dist_params: dict, n: int):
        assert isinstance(dist_params, dict)
        assert isinstance(n, int)

        self.__n = n
        self.__in_raw = dist_params
        self.__in = DistSampler.unflatten_dict(dist_params)

    def to_dict(self, suppress_constant: bool = False):
        n = self.__n
        dist_params = self.__in
        dict_out = dict()

        def make_const_arr(v, dtype=float):
            if suppress_constant:
                return v
            else:
                return np.full((n,), v, dtype=dtype)

        for k, v in dist_params.items():
            if isinstance(v, float) or isinstance(v, int) or isinstance(v, float):
                # dict_out[k] = np.full((n,), v, dtype=float)
                dict_out[k] = make_const_arr(v)
            elif isinstance(v, str):
                # dict_out[k] = np.full((n,), v, dtype=np.dtype("U{:d}".format(len(v))))
                dict_out[k] = make_const_arr(v, np.dtype("U{:d}".format(len(v))))
            elif isinstance(v, np.ndarray) or isinstance(v, list):
                dict_out[k] = list(np.full((n, len(v)), v, dtype=float))
            elif isinstance(v, dict):
                if "dist" in v:
                    try:
                        dict_out[k] = DistSampler._sampling(v, n)
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
                    raise NotImplementedError('RAMP is currently not available')
                else:
                    raise ValueError(f"Unknown input data type for {k}. {v}.")
            elif v is None:
                # dict_out[k] = np.full((n,), np.nan, dtype=float)
                dict_out[k] = make_const_arr(np.nan)
            else:
                raise TypeError(f"Unknown input data type for {k}.")

        dict_out["index"] = np.arange(0, n, 1)
        return dict_out

    def to_xlsx(self, fp: str):
        dict_to_xlsx({i: DistSampler.flatten_dict(v) for i, v in self.to_dict().items()}, fp)

    @staticmethod
    def unflatten_dict(dict_in: dict) -> dict:
        """Invert flatten_dict.

        :param dict_in:
        :return dict_out:
        """
        dict_out = dict()

        for k, v in dict_in.items():
            DistSampler.__unflatten_dict(k, v, dict_out)

        return dict_out

    @staticmethod
    def __unflatten_dict(k: str, v: Any, dict_out: dict):
        if ":" in k:
            k1, *k2 = k.split(':')
            if k1 not in dict_out:
                dict_out[k1] = dict()
            DistSampler.__unflatten_dict(':'.join(k2), v, dict_out[k1])
        else:
            dict_out[k] = v

    @staticmethod
    def flatten_dict(dict_in: dict) -> dict:
        dict_out = dict()
        DistSampler.__flatten_dict(dict_in, dict_out)
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
        >>> assert DistSampler.flatten_dict(dict_in) == output  # True

        :param dict_in:     Any two levels (or less) dict.
        :return dict_out:   Single level dict.
        """
        for k, v in dict_in.items():
            if isinstance(v, dict):
                DistSampler.__flatten_dict(v, dict_out=dict_out, history=k if history is None else f'{history}:{k}')
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
        try:
            import scipy.stats as stats
        except ImportError:
            raise ImportError('scipy required')

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


class DistFunc(ABC):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x: Union[int, float, np.ndarray]):
        return self._pdf(x, self.mu, self.sigma)

    def cdf(self, x: Union[int, float, np.ndarray]):
        return self._cdf(x, self.mu, self.sigma)

    def ppf(self, p: Union[int, float, np.ndarray]):
        return self._ppf(p, self.mu, self.sigma)

    @staticmethod
    @abstractmethod
    def _pdf(x: Union[int, float, np.ndarray], mu: float, sigma: float) -> Union[int, float, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _cdf(x: Union[int, float, np.ndarray], mu: float, sigma: float) -> Union[int, float, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _ppf(p: Union[int, float, np.ndarray], mu: float, sigma: float) -> Union[int, float, np.ndarray]:
        raise NotImplementedError


def erf(x):
    coefficients = np.array([1, -1 / 3, 1 / 10, -1 / 42, 1 / 216])
    powers = np.arange(1, 10, 2)
    terms = coefficients * (x.reshape(-1, 1) ** powers)
    return (2 / np.sqrt(np.pi)) * terms.sum(axis=-1)


def erfinv(x):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.where(x < 0, -1, 1)
    x = np.abs(x)

    t = 1.0 / (1.0 + p * x)
    y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    return sign * y * (1 - x * x + y * y * (a1 + y * (a2 + y * (a3 + y * (a4 + a5 * y)))))


class Gumbel(DistFunc):
    EULER = 0.57721566490153286060  # Euler-Mascheroni constant
    PI = 3.141592653589793

    @staticmethod
    def _pdf(x, mean, stddev):
        mu, sigma = Gumbel._convert_params(mean, stddev)
        z = (x - mu) / sigma
        return (1 / sigma) * np.exp(-(z + np.exp(-z)))

    @staticmethod
    def _cdf(x, mean, stddev):
        mu, sigma = Gumbel._convert_params(mean, stddev)
        z = (x - mu) / sigma
        return np.exp(-np.exp(-z))

    @staticmethod
    def _ppf(q, mean, stddev):
        mu, sigma = Gumbel._convert_params(mean, stddev)
        return mu - sigma * np.log(-np.log(q))

    @staticmethod
    def _convert_params(mean, stddev):
        sigma = stddev * np.sqrt(6) / Gumbel.PI
        mu = mean - Gumbel.EULER * sigma
        return mu, sigma

    @staticmethod
    def test():
        def assert_func(r_, a_):
            assert abs(r_ - a_) < 1e-1, ValueError(f'{r_} != {a_}')

        assert_func(Gumbel(420, 126).ppf(.2), 316.54)
        assert_func(Gumbel(420, 126).ppf(.4), 371.88)
        assert_func(Gumbel(420, 126).ppf(.6), 429.28)
        assert_func(Gumbel(420, 126).ppf(.8), 510.65)


class Normal(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def _cdf(x, mu, sigma):
        return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))

    @staticmethod
    def _ppf(p, mu, sigma):
        return mu + sigma * np.sqrt(2) * erfinv(2 * p - 1)


class Lognormal(DistFunc):
    SQRT_2PI = np.sqrt(2 * np.pi)

    @staticmethod
    def _pdf(x, mu, sigma):
        return (1 / (x * sigma * Lognormal.SQRT_2PI)) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def _cdf(x, mu, sigma):
        return 0.5 + 0.5 * np.erf((np.log(x) - mu) / (sigma * np.sqrt(2)))

    @staticmethod
    def _ppf(q, mu, sigma):
        return np.exp(mu + sigma * np.sqrt(2) * np.erfinv(2 * q - 1))


class Anglit(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        return np.where((x >= mu - np.pi * sigma / 4) & (x <= mu + np.pi * sigma / 4),
                        1 / (np.pi * sigma) * (1 + np.cos((x - mu) / sigma)), 0)

    @staticmethod
    def _cdf(x, mu, sigma):
        return np.where((x >= mu - np.pi * sigma / 4) & (x <= mu + np.pi * sigma / 4),
                        2 / np.pi * (x - mu) / sigma + np.sin((x - mu) / sigma) / (np.pi * sigma) + 0.5, 0)

    @staticmethod
    def _ppf(p, mu, sigma):
        return np.where((p >= 0) & (p <= 1), mu + sigma * np.arcsin(2 * np.pi * p - 1), 0)


class Arcsine(DistFunc):
    @staticmethod
    def _pdf(x, mean, std_dev):
        return np.where((x > mean - std_dev) & (x < mean + std_dev),
                        1 / (np.pi * std_dev * np.sqrt((x - mean) * (1 - x + mean))), 0)

    @staticmethod
    def _cdf(x, mean, std_dev):
        return np.where(
            (x >= mean - std_dev) & (x <= mean + std_dev),
            2 / np.pi * np.arcsin(np.sqrt((x - mean) / (2 * std_dev) + 0.5)),
            0
        )

    @staticmethod
    def _ppf(p, mean, std_dev):
        return np.where((p >= 0) & (p <= 1), mean + std_dev * np.sin(np.pi * p / 2) ** 2, 0)


class Cauchy(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        return 1 / (np.pi * sigma * (1 + ((x - mu) / sigma) ** 2))

    @staticmethod
    def _cdf(x, mu, sigma):
        return 1 / np.pi * np.arctan((x - mu) / sigma) + 0.5

    @staticmethod
    def _ppf(p, mu, sigma):
        return mu + sigma * np.tan(np.pi * (p - 0.5))


class Cosine(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        return np.where((x >= mu - np.pi * sigma / 2) & (x <= mu + np.pi * sigma / 2),
                        1 / (np.pi * sigma) * (np.cos((x - mu) / sigma) + 1) / 2, 0)

    @staticmethod
    def _cdf(x, mu, sigma):
        return np.where((x >= mu - np.pi * sigma / 2) & (x <= mu + np.pi * sigma / 2),
                        (x - mu) / (np.pi * sigma) + np.sin((x - mu) / sigma) / (np.pi * sigma) + 0.5, 0)

    @staticmethod
    def _ppf(p, mu, sigma):
        return np.where((p >= 0) & (p <= 1), mu + sigma * np.arccos(1 - 2 * p), 0)


class HyperbolicSecantDistribution(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        return 1 / (np.pi * sigma * np.cosh((x - mu) / sigma))

    @staticmethod
    def _cdf(x, mu, sigma):
        return 0.5 + 0.5 * np.tanh((x - mu) / (2 * sigma))

    @staticmethod
    def _ppf(p, mu, sigma):
        return mu + 2 * sigma * np.arctanh(2 * p - 1)


class HalfCauchy(DistFunc):

    @staticmethod
    def _pdf(x, mu, sigma):
        """
        Returns the value of the probability density function for half-Cauchy distribution.
        """
        return 1 / (np.pi * sigma * (1 + ((x - mu) / sigma) ** 2))

    @staticmethod
    def _cdf(x, mean, std_dev):
        """
        Returns the value of the cumulative distribution function for half-Cauchy distribution.
        """
        return 2 / np.pi * np.arctan((x - mean) / std_dev)

    @staticmethod
    def _ppf(q, mean, std_dev):
        """
        Returns the value of the percent point function (also called inverse cumulative function) for half-Cauchy distribution.
        """
        return mean + std_dev * np.tan(np.pi * (q - 0.5))


class HalfNormal(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Half Normal distribution
        x = (x - mu) / sigma
        return np.exp(-0.5 * (x ** 2)) / (np.sqrt(2 * np.pi) * sigma)

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Half Normal distribution
        x = (x - mu) / sigma
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Half Normal distribution
        x = np.sqrt(2) * sigma * np.erfinv(2 * p - 1)
        return mu + x


class HalfLogistic(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Half Logistic distribution
        z = (x - mu) / sigma
        return np.exp(-z) / (sigma * (1 + np.exp(-z)) ** 2)

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Half Logistic distribution
        z = (x - mu) / sigma
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Half Logistic distribution
        z = np.log(p / (1 - p))
        return mu + sigma * z


class Laplace(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Laplace distribution
        z = np.abs((x - mu) / sigma)
        return np.exp(-z) / (2 * sigma)

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Laplace distribution
        z = (x - mu) / sigma
        return np.where(z < 0, 0.5 * np.exp(z), 1 - 0.5 * np.exp(-z))

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Laplace distribution
        q = p - 0.5
        return mu - np.sign(q) * sigma * np.log(1 - 2 * np.abs(q))


class Levy(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Levy distribution
        z = (x - mu) / sigma
        return np.sqrt(2 / (np.pi * sigma ** 2)) * np.exp(-1 / (2 * sigma ** 2 * z)) / np.abs(z) ** (3 / 2)

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Levy distribution
        z = (x - mu) / sigma
        return 1 - np.exp(-1 / (2 * sigma ** 2 * z))

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Levy distribution
        return mu + sigma / np.sqrt(-2 * np.log(1 - p))


class Logistic(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Logistic distribution
        z = (x - mu) / sigma
        exp_term = np.exp(-z)
        return exp_term / (sigma * (1 + exp_term) ** 2)

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Logistic distribution
        z = (x - mu) / sigma
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Logistic distribution
        return mu + sigma * np.log(p / (1 - p))


class Maxwell(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Maxwell distribution
        z = (x - mu) / sigma
        exp_term = np.exp(-0.5 * (z ** 2))
        return (np.sqrt(2 / np.pi) * z ** 2 * exp_term) / sigma ** 3

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Maxwell distribution
        z = (x - mu) / sigma
        return 1 - np.exp(-0.5 * (z ** 2))

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Maxwell distribution
        return mu + sigma * np.sqrt(-2 * np.log(1 - p))


class Rayleigh(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Rayleigh distribution
        z = (x - mu) / sigma
        return (z / sigma ** 2) * np.exp(-0.5 * (z ** 2))

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Rayleigh distribution
        z = (x - mu) / sigma
        return 1 - np.exp(-0.5 * (z ** 2))

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Rayleigh distribution
        return mu + sigma * np.sqrt(-2 * np.log(1 - p))


class Semicircular(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Semicircular distribution
        z = (x - mu) / sigma
        pdf = np.zeros_like(z)
        pdf[np.abs(z) <= 1] = (2 / np.pi) * np.sqrt(1 - z[np.abs(z) <= 1] ** 2)
        return pdf / sigma

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Semicircular distribution
        z = (x - mu) / sigma
        cdf = np.zeros_like(z)
        cdf[np.abs(z) <= 1] = (1 / np.pi) * (
                np.arcsin(z[np.abs(z) <= 1]) + (z[np.abs(z) <= 1] * np.sqrt(1 - z[np.abs(z) <= 1] ** 2)))
        return cdf

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Semicircular distribution
        ppf = np.zeros_like(p)
        ppf[(p >= 0) & (p <= 0.5)] = mu + sigma * np.sin(2 * np.pi * p[(p >= 0) & (p <= 0.5)])
        ppf[p > 0.5] = mu + sigma * np.sin(2 * np.pi * (p[p > 0.5] - 1))
        return ppf


class Uniform(DistFunc):
    @staticmethod
    def _pdf(x, a, b):
        # Calculate the probability density function (PDF) of the Uniform distribution
        pdf = np.zeros_like(x)
        pdf[(x >= a) & (x <= b)] = 1 / (b - a)
        return pdf

    @staticmethod
    def _cdf(x, a, b):
        # Calculate the cumulative distribution function (CDF) of the Uniform distribution
        cdf = np.zeros_like(x)
        cdf[x >= b] = 1
        cdf[(x >= a) & (x < b)] = (x[(x >= a) & (x < b)] - a) / (b - a)
        return cdf

    @staticmethod
    def _ppf(p, a, b):
        # Calculate the percent point function (PPF) of the Uniform distribution
        ppf = np.zeros_like(p)
        ppf[p >= 1] = b
        ppf[p < 1] = a + (b - a) * p[p < 1]
        return ppf


class Wald(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Calculate the probability density function (PDF) of the Wald distribution
        pdf = np.sqrt(2 * np.pi) * (x ** 3 / (mu ** 3 * sigma ** 5)) * np.exp(-0.5 * ((x - mu) / (mu * sigma)) ** 2)
        return pdf

    @staticmethod
    def _cdf(x, mu, sigma):
        # Calculate the cumulative distribution function (CDF) of the Wald distribution
        cdf = 0.5 * (1 + np.erf((np.sqrt(x / mu) - 1) / (np.sqrt(2) * sigma)))
        return cdf

    @staticmethod
    def _ppf(p, mu, sigma):
        # Calculate the percent point function (PPF) of the Wald distribution
        ppf = mu * (1 + (mu / sigma) ** 2 * erfinv(2 * p - 1)) ** 2 / (1 - (mu / sigma) ** 2 * erfinv(2 * p - 1))
        return ppf
