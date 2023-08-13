from abc import ABC, abstractmethod
from inspect import getfullargspec
from typing import Union

import numpy as np

from sfeprapy.func.erf import erf, erfinv

__all__ = ('Normal', 'Gumbel', 'Lognormal', 'Arcsine', 'Cauchy', 'HyperbolicSecant', 'HalfCauchy', 'Logistic',
           'Uniform', 'DistFunc', 'Constant', 'LognormalMod', 'Discrete')


class DistFunc(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if 'lbound' in kwargs and 'lim_1' not in kwargs:
            kwargs['lim_1'] = kwargs.pop('lbound')
        if 'ubound' in kwargs and 'lim_2' not in kwargs:
            kwargs['lim_2'] = kwargs.pop('ubound')

    def pdf(self, x: Union[int, float, np.ndarray]):
        args = list()
        for i, name in enumerate(getfullargspec(self._pdf).args[1:]):
            if name in self.kwargs:
                args.append(self.kwargs[name])
            else:
                args.append(self.args[i])
        return self._pdf(x, *args)

    def cdf(self, x: Union[int, float, np.ndarray]):
        args = list()
        for i, name in enumerate(getfullargspec(self._cdf).args[1:]):
            if name in self.kwargs:
                args.append(self.kwargs[name])
            else:
                args.append(self.args[i])
        return self._cdf(x, *args)

    def ppf(self, p: Union[int, float, np.ndarray]):
        args = list()
        for i, name in enumerate(getfullargspec(self._ppf).args[1:]):
            if name in self.kwargs:
                args.append(self.kwargs[name])
            else:
                args.append(self.args[i])
        return self._ppf(p, *args)

    def sampling(self, n: int, lim_1: float = None, lim_2: float = None, shuffle: bool = True):
        padding = 1. / n

        if lim_1 is None:
            lim_1 = padding
        else:
            lim_1 = self.cdf(lim_1)

        if lim_2 is None:
            lim_2 = 1 - padding
        else:
            lim_2 = self.cdf(lim_2)

        samples = self.ppf(np.linspace(lim_1, lim_2, n))
        if shuffle:
            np.random.shuffle(samples)
        return samples

    @staticmethod
    @abstractmethod
    def _pdf(x: Union[int, float, np.ndarray], *args, **kwargs) -> Union[int, float, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _cdf(x: Union[int, float, np.ndarray], *args, **kwargs) -> Union[int, float, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _ppf(p: Union[int, float, np.ndarray], *args, **kwargs) -> Union[int, float, np.ndarray]:
        raise NotImplementedError


def assert_func(r_, a_, tol=1e-1):
    assert abs(r_ - a_) < tol, ValueError(f'{r_} != {a_}')


def test_erf():
    try:
        from scipy.special import erf as erf_
    except ImportError:
        raise ImportError('SciPy is required for testing')

    i = np.linspace(-2, 2, 11)[1:-1]
    a = erf_(i)
    b = erf(i)
    for _ in range(len(i)):
        print(f'{a[_]:7.5f}  {b[_]:7.5f}')


def test_erfinv():
    try:
        from scipy.special import erfinv as erfinv_
    except ImportError:
        raise ImportError('SciPy is required for testing')

    i = np.linspace(-1, 1, 11)[1:-1]
    a = erfinv_(i)
    b = erfinv(i)
    for _ in range(len(i)):
        print(f'{a[_]:7.5f}  {b[_]:7.5f}')


class Constant(DistFunc):
    def __init__(self, value: Union[int, float], *_, **__):
        self.value = value
        super().__init__(value)

    def sampling(self, n: int, lim_1: float = None, lim_2: float = None, shuffle: bool = True):
        return np.full((n,), self.value, )

    @staticmethod
    def _pdf():
        raise ValueError('PDF not available to Constant DistFunc')

    @staticmethod
    def _cdf():
        raise ValueError('CDF not available to Constant DistFunc')

    @staticmethod
    def _ppf():
        raise ValueError('PPF not available to Constant DistFunc')


class Discrete(DistFunc):
    def __init__(self, values, weights):
        if isinstance(values, str):
            assert ',' in values, f'`discrete_ distribution `values` parameter is not a list separated by comma.'
            values = [float(i.strip()) for i in values.split(',')]

        if isinstance(weights, str):
            assert ',' in weights, f'`discrete_`:`weights` is not a list of numbers separated by comma.'
            weights = [float(i.strip()) for i in weights.split(',')]

        assert len(values) == len(
            weights), f'Length of values ({len(values)}) and weights ({len(values)}) do not match.'
        assert sum(weights) == 1., f'Sum of all weights should be unity, got {sum(weights)}.'

        super().__init__(values, weights)

    def sampling(self, n: int, lim_1: float = None, lim_2: float = None, shuffle: bool = True):
        samples = self._sampling(n, *self.args, **self.kwargs)
        if shuffle:
            np.random.shuffle(samples)
        return samples

    @staticmethod
    def _sampling(n, values, weights):
        weights = [int(round(i * n)) for i in weights]
        if (sum_sampled := sum(weights)) < n:
            for i in np.random.choice(np.arange(len(weights)), size=sum_sampled - n):
                weights[i] += 1
        elif sum_sampled > n:
            for i in np.random.choice(np.arange(len(weights)), size=sum_sampled - n):
                weights[i] -= 1
        weights = np.cumsum(weights)
        assert weights[-1] == n, f'Total weight length does not match `num_samples`.'
        samples = np.empty((n,), dtype=float)
        for i, v__ in enumerate((values)):
            if i == 0:
                samples[0:weights[i]] = v__
            else:
                samples[weights[i - 1]:weights[i]] = v__
        return samples

    @staticmethod
    def _pdf(x, v, w):
        v = np.asarray(v)
        w = np.asarray(w)

        if np.sum(w) != 1.0:
            w = w / np.sum(w)

        pdf_dict = dict(zip(v, w))

        # Check if x exists in v and return corresponding w
        return pdf_dict.get(x, 0)

    @staticmethod
    def _cdf(x, v, w):
        v = np.asarray(v)
        w = np.asarray(w)

        if np.sum(w) != 1.0:
            w = w / np.sum(w)

        cdf_dict = dict(zip(v, np.cumsum(w)))

        # If x is not in v, return 1 for values larger than max(v) and 0 for values smaller than min(v)
        if x not in cdf_dict:
            if x < min(v):
                return 0
            elif x > max(v):
                return 1

        return cdf_dict[x]

    @staticmethod
    def _ppf(x, v, w):
        v = np.asarray(v)
        w = np.asarray(w)

        if np.sum(w) != 1.0:
            w = w / np.sum(w)

        cdf = np.cumsum(w)

        # If x is not between 0 and 1, return None
        if x < 0 or x > 1:
            return None

        return v[np.argwhere(cdf >= x)[0][0]]


class Gumbel(DistFunc):
    @staticmethod
    def _pdf(x, mean, sd):
        mean, sd = Gumbel._convert_params(mean, sd)
        z = (x - mean) / sd
        return (1 / sd) * np.exp(-(z + np.exp(-z)))

    @staticmethod
    def _cdf(x, mean, sd):
        mean, sd = Gumbel._convert_params(mean, sd)
        z = (x - mean) / sd
        return np.exp(-np.exp(-z))

    @staticmethod
    def _ppf(q, mean, sd):
        mean, sd = Gumbel._convert_params(mean, sd)
        return mean - sd * np.log(-np.log(q))

    @staticmethod
    def _convert_params(mean, sd):
        sd = sd * np.sqrt(6) / np.pi
        mean = mean - 0.57721566490153286060 * sd
        return mean, sd

    @staticmethod
    def test():
        d = Gumbel(420, 126)
        assert_func(d.pdf(316.541), 0.00327648, tol=1e-3)
        assert_func(d.pdf(365.069), 0.00374402, tol=1e-3)
        assert_func(d.pdf(413.596), 0.00335019, tol=1e-3)
        assert_func(d.pdf(462.123), 0.00258223, tol=1e-3)
        assert_func(d.pdf(510.650), 0.00181710, tol=1e-3)
        assert_func(d.cdf(316.541), 0.20000, tol=1e-3)
        assert_func(d.cdf(365.069), 0.37453, tol=1e-3)
        assert_func(d.cdf(413.596), 0.54921, tol=1e-3)
        assert_func(d.cdf(462.123), 0.69372, tol=1e-3)
        assert_func(d.cdf(510.650), 0.80000, tol=1e-3)
        assert_func(d.ppf(0.20000), 316.541)
        assert_func(d.ppf(0.37453), 365.069)
        assert_func(d.ppf(0.54921), 413.596)
        assert_func(d.ppf(0.69372), 462.123)
        assert_func(d.ppf(0.80000), 510.650)


if __name__ == '__main__':
    Gumbel.test()


class Normal(DistFunc):
    @staticmethod
    def _pdf(x, mean, sd):
        return (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * sd ** 2))

    @staticmethod
    def _cdf(x, mean, sd):
        return 0.5 * (1 + erf((x - mean) / (np.sqrt(2) * sd)))

    @staticmethod
    def _ppf(p, mean, sd):
        return mean + sd * np.sqrt(2) * erfinv(2 * p - 1)

    @staticmethod
    def test():
        d = Normal(420, 126)
        assert_func(d.pdf(313.956), 0.00222192, tol=1e-3)
        assert_func(d.pdf(366.978), 0.00289792, tol=1e-3)
        assert_func(d.pdf(420.000), 0.00316621, tol=1e-3)
        assert_func(d.pdf(473.022), 0.00289792, tol=1e-3)
        assert_func(d.pdf(526.044), 0.00222192, tol=1e-3)
        assert_func(d.cdf(313.956), 0.20000, tol=1e-3)
        assert_func(d.cdf(366.978), 0.33695, tol=1e-3)
        assert_func(d.cdf(420.000), 0.50000, tol=1e-3)
        assert_func(d.cdf(473.022), 0.66305, tol=1e-3)
        assert_func(d.cdf(526.044), 0.80000, tol=1e-3)
        assert_func(d.ppf(0.20000), 313.956)
        assert_func(d.ppf(0.33695), 366.978)
        assert_func(d.ppf(0.50000), 420.000)
        assert_func(d.ppf(0.66305), 473.022)
        assert_func(d.ppf(0.80000), 526.044)


if __name__ == '__main__':
    Normal.test()


class Lognormal(DistFunc):
    SQRT_2PI = np.sqrt(2 * np.pi)

    @staticmethod
    def _pdf(x, mean, sd):
        # Convert M and S to mean and sd of the underlying normal distribution
        mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + sd ** 2))
        sigma = np.sqrt(np.log(1 + sd ** 2 / mean ** 2))

        # Probability Density Function
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def _cdf(x, mean, sd):
        # Convert M and S to mean and sd of the underlying normal distribution
        mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + sd ** 2))
        sigma = np.sqrt(np.log(1 + sd ** 2 / mean ** 2))

        # Cumulative Distribution Function
        return 0.5 + 0.5 * erf((np.log(x) - mu) / (sigma * np.sqrt(2)))

    @staticmethod
    def _ppf(q, mean, sd):
        # Convert M and S to mean and sd of the underlying normal distribution
        mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + sd ** 2))
        sigma = np.sqrt(np.log(1 + sd ** 2 / mean ** 2))

        # Percent Point Function (Quantile Function)
        return np.exp(mu + sigma * np.sqrt(2) * erfinv(2 * q - 1))

    @staticmethod
    def test():
        d = Lognormal(420, 126)
        assert_func(d.pdf(314.222), 0.00303505, tol=1e-3)
        assert_func(d.pdf(364.425), 0.00352359, tol=1e-3)
        assert_func(d.pdf(414.628), 0.00326027, tol=1e-3)
        assert_func(d.pdf(464.831), 0.00259000, tol=1e-3)
        assert_func(d.pdf(515.034), 0.00185168, tol=1e-3)
        assert_func(d.cdf(314.222), 0.20000, tol=1e-3)
        assert_func(d.cdf(364.425), 0.36817, tol=1e-3)
        assert_func(d.cdf(414.628), 0.54099, tol=1e-3)
        assert_func(d.cdf(464.831), 0.68873, tol=1e-3)
        assert_func(d.cdf(515.034), 0.80000, tol=1e-3)
        assert_func(d.ppf(0.20000), 314.222)
        assert_func(d.ppf(0.36817), 364.425)
        assert_func(d.ppf(0.54099), 414.628)
        assert_func(d.ppf(0.68873), 464.831)
        assert_func(d.ppf(0.80000), 515.034)


if __name__ == '__main__':
    Lognormal.test()


# br187_fuel_load_density_
# br187_hrr_density_
class Br187FuelLoadDensity(DistFunc):
    @staticmethod
    def _pdf(*_, **__):
        pass

    @staticmethod
    def _cdf(*_, **__):
        pass

    @staticmethod
    def _ppf(*_, **__):
        pass

    def sampling(self, n: int, lim_1: float = None, lim_2: float = None, shuffle: bool = True):
        samples_1 = Gumbel(mean=780, sd=234).sampling(n, lim_1=lim_1, lim_2=lim_2, shuffle=shuffle)
        samples_2 = Gumbel(mean=420, sd=420).sampling(n, lim_1=lim_1, lim_2=lim_2, shuffle=shuffle)
        samples = np.random.choice(np.append((samples_1, samples_2)), n, replace=False)
        return samples


class Br187HrrDensity(DistFunc):
    @staticmethod
    def _pdf(*_, **__):
        pass

    @staticmethod
    def _cdf(*_, **__):
        pass

    @staticmethod
    def _ppf(*_, **__):
        pass

    def sampling(self, n: int, lim_1: float = None, lim_2: float = None, shuffle: bool = True):
        a, b = 0.32, 0.57
        mean, sd = (a + b) / 2, (b - a) / (2 * np.sqrt(3))
        samples_1 = Uniform(mean=mean, sd=sd).sampling(n, lim_1=lim_1, lim_2=lim_2, shuffle=shuffle)
        a, b = 0.15, 0.65
        mean, sd = (a + b) / 2, (b - a) / (2 * np.sqrt(3))
        samples_2 = Uniform(mean=mean, sd=sd).sampling(n, lim_1=lim_1, lim_2=lim_2, shuffle=shuffle)
        samples = np.random.choice(np.append((samples_1, samples_2)), n, replace=False)
        return samples


class LognormalMod(Lognormal):
    def sampling(self, n: int, lim_1: float = None, lim_2: float = None, shuffle: bool = True):
        return 1 - super().sampling(n=n, lim_1=lim_1, lim_2=lim_2, shuffle=shuffle)


class Arcsine(DistFunc):
    @staticmethod
    def _pdf(x, mean, sd):
        # Calculate a and b from the mean and standard deviation
        a = mean - np.sqrt(2) * sd
        b = mean + np.sqrt(2) * sd

        # Ensure that x is within the range [a, b]
        # if x < a or x > b:
        #     raise ValueError("x must be within the range [a, b]")

        # Compute the PDF of the arcsine distribution
        pdf_value = 1 / (np.pi * np.sqrt((x - a) * (b - x)))

        return pdf_value

    @staticmethod
    def _cdf(x, mean, sd):
        # Calculate a and b from the mean and standard deviation
        a = mean - np.sqrt(2) * sd
        b = mean + np.sqrt(2) * sd

        # Ensure that x is within the range [a, b]
        # if x < a or x > b:
        #     raise ValueError("x must be within the range [a, b]")

        # Standardize x
        x_std = (x - a) / (b - a)

        # Compute the CDF of the standardized arcsine distribution
        cdf_value = (2 / np.pi) * np.arcsin(np.sqrt(x_std))

        # CDF of the generalized arcsine distribution
        return cdf_value

    @staticmethod
    def _ppf(p, mean, sd):
        # Calculate a and b from the mean and standard deviation
        a = mean - np.sqrt(2) * sd
        b = mean + np.sqrt(2) * sd

        # Compute the PPF of the arcsine distribution
        ppf_value = a + (b - a) * (np.sin(np.pi * p / 2)) ** 2

        return ppf_value

    @staticmethod
    def test():
        d = Arcsine(420, 126)
        assert_func(d.pdf(275.84), 0.00303911, tol=1e-3)
        assert_func(d.pdf(347.92), 0.00195328, tol=1e-3)
        assert_func(d.pdf(420.00), 0.00178634, tol=1e-3)
        assert_func(d.pdf(492.08), 0.00195328, tol=1e-3)
        assert_func(d.pdf(564.16), 0.00303911, tol=1e-3)
        assert_func(d.cdf(250), 0.09689, tol=1e-3)
        assert_func(d.cdf(300), 0.26482, tol=1e-3)
        assert_func(d.cdf(400), 0.46420, tol=1e-3)
        assert_func(d.cdf(500), 0.64820, tol=1e-3)
        assert_func(d.cdf(550), 0.76027, tol=1e-3)
        assert_func(d.ppf(0.09689), 250)
        assert_func(d.ppf(0.26482), 300)
        assert_func(d.ppf(0.46420), 400)
        assert_func(d.ppf(0.64820), 500)
        assert_func(d.ppf(0.76027), 550)


if __name__ == '__main__':
    Arcsine.test()


class Cauchy(DistFunc):
    @staticmethod
    def _pdf(x, mean, sd):
        return 1 / (np.pi * sd * (1 + ((x - mean) / sd) ** 2))

    @staticmethod
    def _cdf(x, mean, sd):
        return 1 / np.pi * np.arctan((x - mean) / sd) + 0.5

    @staticmethod
    def _ppf(p, mean, sd):
        return mean + sd * np.tan(np.pi * (p - 0.5))

    @staticmethod
    def test():
        d = Cauchy(420, 126)
        assert_func(d.pdf(246.576), 0.00087280, tol=1e-3)
        assert_func(d.pdf(333.288), 0.00171434, tol=1e-3)
        assert_func(d.pdf(420.000), 0.00252627, tol=1e-3)
        assert_func(d.pdf(506.712), 0.00171434, tol=1e-3)
        assert_func(d.pdf(593.424), 0.00087280, tol=1e-3)
        assert_func(d.cdf(246.576), 0.20000, tol=1e-3)
        assert_func(d.cdf(333.288), 0.30814, tol=1e-3)
        assert_func(d.cdf(420.000), 0.50000, tol=1e-3)
        assert_func(d.cdf(506.712), 0.69186, tol=1e-3)
        assert_func(d.cdf(593.424), 0.80000, tol=1e-3)
        assert_func(d.ppf(0.20000), 246.576)
        assert_func(d.ppf(0.30814), 333.288)
        assert_func(d.ppf(0.50000), 420.000)
        assert_func(d.ppf(0.69186), 506.712)
        assert_func(d.ppf(0.80000), 593.424)


if __name__ == '__main__':
    Cauchy.test()


class HyperbolicSecant(DistFunc):
    @staticmethod
    def _pdf(x, sd, mean):
        return (1 / (2 * mean)) * (1 / np.cosh(np.pi / 2 * ((x - sd) / mean)))

    @staticmethod
    def _cdf(x, sd, mean):
        return (2 / np.pi) * np.arctan(np.exp(np.pi / 2 * ((x - sd) / mean)))

    @staticmethod
    def _ppf(p, sd, mean):
        # Check for valid input
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1)")

        # Define the function we want to find the root of
        def f(x):
            return HyperbolicSecant._cdf(x, sd, mean) - p

        # Initial boundaries for bisection method
        lower, upper = sd - 10 * mean, sd + 10 * mean

        # Bisection method
        while upper - lower > 1e-6:  # 1e-6 is the desired accuracy
            midpoint = (upper + lower) / 2
            if f(midpoint) > 0:  # If the function at the midpoint is > 0, the root must be in the left interval
                upper = midpoint
            else:  # Otherwise, the root must be in the right interval
                lower = midpoint

        return (upper + lower) / 2

    @staticmethod
    def test():
        d = HyperbolicSecant(420, 126)

        assert_func(d.pdf(329.825), 0.00233248, tol=1e-3)
        assert_func(d.pdf(374.913), 0.00341451, tol=1e-3)
        assert_func(d.pdf(420.000), 0.00396825, tol=1e-3)
        assert_func(d.pdf(465.087), 0.00341451, tol=1e-3)
        assert_func(d.pdf(510.175), 0.00233248, tol=1e-3)
        assert_func(d.cdf(329.825), 0.20000, tol=1e-3)
        assert_func(d.cdf(374.913), 0.32982, tol=1e-3)
        assert_func(d.cdf(420.000), 0.50000, tol=1e-3)
        assert_func(d.cdf(465.087), 0.67018, tol=1e-3)
        assert_func(d.cdf(510.175), 0.80000, tol=1e-3)
        assert_func(d.ppf(0.20000), 329.825)
        assert_func(d.ppf(0.32982), 374.913)
        assert_func(d.ppf(0.50000), 420.000)
        assert_func(d.ppf(0.67018), 465.087)
        assert_func(d.ppf(0.80000), 510.175)


if __name__ == '__main__':
    HyperbolicSecant.test()


class HalfCauchy(DistFunc):

    @staticmethod
    def _pdf(x, mean, sd):
        if isinstance(x, (int, float)):
            if x < mean:
                return 0
            else:
                return (2 / (np.pi * sd)) / (1 + ((x - mean) / sd) ** 2)
        elif isinstance(x, np.ndarray):
            y = np.where(x < mean, 0, (2 / (np.pi * sd)) / (1 + ((x - mean) / sd) ** 2))
            return y

    @staticmethod
    def _cdf(x, mean, sd):
        if isinstance(x, (int, float)):
            if x < mean:
                return 0
            else:
                return 2 / np.pi * np.arctan((x - mean) / sd)
        elif isinstance(x, np.ndarray):
            y = np.where(x < mean, 0, 2 / np.pi * np.arctan((x - mean) / sd))
            return y

    @staticmethod
    def _ppf(q, mean, sd):
        """
        Returns the value of the percent point function (also called inverse cumulative function) for half-Cauchy distribution.
        """
        return mean + sd * np.tan(np.pi / 2 * q)

    @staticmethod
    def test():
        d = HalfCauchy(420, 126)
        assert_func(d.pdf(460.940), 0.004570060, tol=1e-3)
        assert_func(d.pdf(547.652), 0.002493360, tol=1e-3)
        assert_func(d.pdf(634.364), 0.001297380, tol=1e-3)
        assert_func(d.pdf(721.076), 0.000753023, tol=1e-3)
        assert_func(d.pdf(807.788), 0.000482474, tol=1e-3)
        assert_func(d.cdf(460.940), 0.20000, tol=1e-3)
        assert_func(d.cdf(547.652), 0.50415, tol=1e-3)
        assert_func(d.cdf(634.364), 0.66171, tol=1e-3)
        assert_func(d.cdf(721.076), 0.74767, tol=1e-3)
        assert_func(d.cdf(807.788), 0.80000, tol=1e-3)
        assert_func(d.ppf(0.20000), 460.940)
        assert_func(d.ppf(0.50415), 547.652)
        assert_func(d.ppf(0.66171), 634.364)
        assert_func(d.ppf(0.74767), 721.076)
        assert_func(d.ppf(0.80000), 807.788)


if __name__ == '__main__':
    HalfCauchy.test()


class Logistic(DistFunc):
    @staticmethod
    def _pdf(x, mean, sd):
        # Convert sd to s
        s = sd * np.sqrt(3.) / np.pi

        # Probability Density Function
        return np.exp(-(x - mean) / s) / (s * (1 + np.exp(-(x - mean) / s)) ** 2)

    @staticmethod
    def _cdf(x, mean, sd):
        # Convert sd to s
        s = sd * np.sqrt(3.) / np.pi

        # Cumulative Distribution Function
        return 1 / (1 + np.exp(-(x - mean) / s))

    @staticmethod
    def _ppf(q, mean, sd):
        # Convert sd to s
        s = sd * np.sqrt(3.) / np.pi

        # Percent-Point Function (Quantile Function)
        return mean - s * np.log((1 / q) - 1)

    @staticmethod
    def test():
        d = Logistic(420, 126)
        assert_func(d.pdf(323.698), 0.00230324, tol=1e-3)
        assert_func(d.pdf(371.849), 0.00319894, tol=1e-3)
        assert_func(d.pdf(420.000), 0.00359881, tol=1e-3)
        assert_func(d.pdf(468.151), 0.00319894, tol=1e-3)
        assert_func(d.pdf(516.302), 0.00230324, tol=1e-3)
        assert_func(d.cdf(323.698), 0.20000, tol=1e-3)
        assert_func(d.cdf(371.849), 0.33333, tol=1e-3)
        assert_func(d.cdf(420.000), 0.50000, tol=1e-3)
        assert_func(d.cdf(468.151), 0.66667, tol=1e-3)
        assert_func(d.cdf(516.302), 0.80000, tol=1e-3)
        assert_func(d.ppf(0.20000), 323.698)
        assert_func(d.ppf(0.33333), 371.849)
        assert_func(d.ppf(0.50000), 420.000)
        assert_func(d.ppf(0.66667), 468.151)
        assert_func(d.ppf(0.80000), 516.302)


if __name__ == '__main__':
    Logistic.test()


class Uniform(DistFunc):
    @staticmethod
    def _pdf(x, mean, sd):
        """Probability density function"""
        a = mean - np.sqrt(3) * sd
        b = mean + np.sqrt(3) * sd
        return np.where((x >= a) & (x <= b), 1 / (b - a), 0)

    @staticmethod
    def _cdf(x, mean, sd):
        """Cumulative distribution function"""
        a = mean - np.sqrt(3) * sd
        b = mean + np.sqrt(3) * sd
        return np.where(x < a, 0, np.where(x > b, 1, (x - a) / (b - a)))

    @staticmethod
    def _ppf(p, mean, sd):
        """Percent-point function (Inverse of cdf)"""
        # Ensure p is in [0, 1]
        p = np.clip(p, 0, 1)
        a = mean - np.sqrt(3) * sd
        b = mean + np.sqrt(3) * sd
        return a + p * (b - a)

    @staticmethod
    def test():
        d = Uniform(420, 126)
        assert_func(d.pdf(289.057), 0.00229107, tol=1e-3)
        assert_func(d.pdf(354.528), 0.00229107, tol=1e-3)
        assert_func(d.pdf(420.000), 0.00229107, tol=1e-3)
        assert_func(d.pdf(485.472), 0.00229107, tol=1e-3)
        assert_func(d.pdf(550.943), 0.00229107, tol=1e-3)
        assert_func(d.cdf(289.057), 0.20000, tol=1e-3)
        assert_func(d.cdf(354.528), 0.35000, tol=1e-3)
        assert_func(d.cdf(420.000), 0.50000, tol=1e-3)
        assert_func(d.cdf(485.472), 0.65000, tol=1e-3)
        assert_func(d.cdf(550.943), 0.80000, tol=1e-3)
        assert_func(d.ppf(0.20000), 289.057)
        assert_func(d.ppf(0.35000), 354.528)
        assert_func(d.ppf(0.50000), 420.000)
        assert_func(d.ppf(0.65000), 485.472)
        assert_func(d.ppf(0.80000), 550.943)


if __name__ == '__main__':
    Uniform.test()
