from abc import ABC, abstractmethod
from typing import Union

import numpy as np


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
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = np.where(x < 0, -1, 1)
    x = np.abs(x)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t

    return sign * (1 - y * np.exp(-x * x))


def test_erf():
    from scipy.special import erf as erf_

    i = np.linspace(-2, 2, 11)[1:-1]
    a = erf_(i)
    b = erf(i)
    for _ in range(len(i)):
        print(f'{a[_]:7.5f}  {b[_]:7.5f}')


def erfinv(y):
    # Initial guess for the Newton-Raphson method
    x = np.where(y < 0, -1.0, np.where(y == 0, 0.0, 1.0))

    # Iterate until the result is accurate enough
    n = 0
    while np.max(np.abs(y - erf(x))) > 1e-5 and n < 10:
        x = x - (erf(x) - y) / (2 / np.sqrt(np.pi) * np.exp(-x ** 2))
        n += 1
    return x


def test_erfinv():
    from scipy.special import erfinv as erfinv_
    i = np.linspace(-1, 1, 11)[1:-1]
    a = erfinv_(i)
    b = erfinv(i)
    for _ in range(len(i)):
        print(f'{a[_]:7.5f}  {b[_]:7.5f}')


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


if __name__ == '__main__':
    Gumbel.test()


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

    @staticmethod
    def test():
        def assert_func(r_, a_):
            assert abs(r_ - a_) < 1e-1, ValueError(f'{r_} != {a_}')

        assert_func(Normal(420, 126).ppf(.2), 313.96)
        assert_func(Normal(420, 126).ppf(.4), 388.08)
        assert_func(Normal(420, 126).ppf(.5), 420.00)
        assert_func(Normal(420, 126).ppf(.6), 451.92)
        assert_func(Normal(420, 126).ppf(.8), 526.04)


if __name__ == '__main__':
    Normal.test()


class Lognormal(DistFunc):
    SQRT_2PI = np.sqrt(2 * np.pi)

    @staticmethod
    def _pdf(x, M, S):
        # Convert M and S to mu and sigma of the underlying normal distribution
        mu = np.log(M ** 2 / np.sqrt(M ** 2 + S ** 2))
        sigma = np.sqrt(np.log(1 + S ** 2 / M ** 2))

        # Probability Density Function
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def _cdf(x, M, S):
        # Convert M and S to mu and sigma of the underlying normal distribution
        mu = np.log(M ** 2 / np.sqrt(M ** 2 + S ** 2))
        sigma = np.sqrt(np.log(1 + S ** 2 / M ** 2))

        # Cumulative Distribution Function
        return 0.5 + 0.5 * erf((np.log(x) - mu) / (sigma * np.sqrt(2)))

    @staticmethod
    def _ppf(q, M, S):
        # Convert M and S to mu and sigma of the underlying normal distribution
        mu = np.log(M ** 2 / np.sqrt(M ** 2 + S ** 2))
        sigma = np.sqrt(np.log(1 + S ** 2 / M ** 2))

        # Percent Point Function (Quantile Function)
        return np.exp(mu + sigma * np.sqrt(2) * erfinv(2 * q - 1))

    @staticmethod
    def test():
        def assert_func(r_, a_):
            assert abs(r_ - a_) < 1e-1, ValueError(f'{r_} != {a_}')

        assert_func(Lognormal(420, 126).ppf(.2), 314.22)
        assert_func(Lognormal(420, 126).ppf(.4), 373.45)
        assert_func(Lognormal(420, 126).ppf(.5), 402.29)
        assert_func(Lognormal(420, 126).ppf(.6), 433.35)
        assert_func(Lognormal(420, 126).ppf(.8), 515.03)


if __name__ == '__main__':
    Lognormal.test()


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
        # Convert sigma to s
        s = sigma * np.sqrt(3.) / np.pi

        # Probability Density Function
        return np.exp(-(x - mu) / s) / (s * (1 + np.exp(-(x - mu) / s)) ** 2)

    @staticmethod
    def _cdf(x, mu, sigma):
        # Convert sigma to s
        s = sigma * np.sqrt(3.) / np.pi

        # Cumulative Distribution Function
        return 1 / (1 + np.exp(-(x - mu) / s))

    @staticmethod
    def _ppf(q, mu, sigma):
        # Convert sigma to s
        s = sigma * np.sqrt(3.) / np.pi

        # Percent-Point Function (Quantile Function)
        return mu - s * np.log((1 / q) - 1)

    @staticmethod
    def test():
        def assert_func(r_, a_):
            assert abs(r_ - a_) < 1e-1, ValueError(f'{r_} != {a_}')

        assert_func(Logistic(420, 126).ppf(.2), 323.70)
        assert_func(Logistic(420, 126).ppf(.4), 391.83)
        assert_func(Logistic(420, 126).ppf(.5), 420.00)
        assert_func(Logistic(420, 126).ppf(.6), 448.17)
        assert_func(Logistic(420, 126).ppf(.8), 516.30)


if __name__ == '__main__':
    Logistic.test()


class Maxwell(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Rescale and shift x
        x = (x - mu) / sigma

        # Probability Density Function
        return np.sqrt(2/np.pi) * (x**2) * np.exp(-x**2 / 2)

    @staticmethod
    def _cdf(x, mu, sigma):
        # Rescale and shift x
        x = (x - mu) / sigma

        # Cumulative Distribution Function
        return erf(x / np.sqrt(2)) - np.sqrt(2 / np.pi) * x * np.exp(-x ** 2 / 2)

    @staticmethod
    def _ppf(q, mu, sigma, tol=1e-6, max_iter=1000):
        # Function to solve
        f = lambda x: Maxwell._cdf(x, mu, sigma) - q

        # Bisection method
        left, right = 0, 10
        for _ in range(max_iter):
            mid = (left + right) / 2
            if f(mid) > 0:  # mid is too high
                right = mid
            else:  # mid is too low
                left = mid
            if abs(f(mid)) < tol:
                return mid

        raise ValueError("Did not converge")

    @staticmethod
    def test():
        print(Maxwell(420, 126).cdf(np.array([200,400,500,600,800])))

        def assert_func(r_, a_, tol=1e-1):
            assert abs(r_ - a_) < tol, ValueError(f'{r_} != {a_}')

        # assert_func(Maxwell(420, 126).cdf(200), 0.01868, tol=1e-4)
        # assert_func(Maxwell(420, 126).cdf(400), 0.47134, tol=1e-4)
        # assert_func(Maxwell(420, 126).cdf(420), 0.53305, tol=1e-4)
        # assert_func(Maxwell(420, 126).cdf(600), 0.91200, tol=1e-4)
        # assert_func(Maxwell(420, 126).cdf(800), 0.99568, tol=1e-4)

        # assert_func(Maxwell(420, 126).ppf(.2), 309.02)
        # assert_func(Maxwell(420, 126).ppf(.4), 377.23)
        # assert_func(Maxwell(420, 126).ppf(.5), 409.22)
        # assert_func(Maxwell(420, 126).ppf(.6), 442.58)
        # assert_func(Maxwell(420, 126).ppf(.8), 524.53)


if __name__ == '__main__':
    Maxwell.test()


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
