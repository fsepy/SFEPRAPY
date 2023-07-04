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


def assert_func(r_, a_, tol=1e-1):
    assert abs(r_ - a_) < tol, ValueError(f'{r_} != {a_}')


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
        assert_func(Gumbel(420, 126).cdf(200), 0.00514, tol=1e-3)
        assert_func(Gumbel(420, 126).cdf(400), 0.50247, tol=1e-3)
        assert_func(Gumbel(420, 126).cdf(500), 0.77982, tol=1e-3)
        assert_func(Gumbel(420, 126).cdf(600), 0.91405, tol=1e-3)
        assert_func(Gumbel(420, 126).cdf(800), 0.98833, tol=1e-3)

        assert_func(Gumbel(420, 126).ppf(.2), 316.54)
        assert_func(Gumbel(420, 126).ppf(.4), 371.88)
        assert_func(Gumbel(420, 126).ppf(.5), 399.30)
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
        assert_func(Normal(420, 126).cdf(200), 0.04040, tol=1e-3)
        assert_func(Normal(420, 126).cdf(400), 0.43694, tol=1e-3)
        assert_func(Normal(420, 126).cdf(500), 0.73726, tol=1e-3)
        assert_func(Normal(420, 126).cdf(600), 0.92344, tol=1e-3)
        assert_func(Normal(420, 126).cdf(800), 0.99872, tol=1e-3)

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
        assert_func(Lognormal(420, 126).ppf(.2), 314.22)
        assert_func(Lognormal(420, 126).ppf(.4), 373.45)
        assert_func(Lognormal(420, 126).ppf(.5), 402.29)
        assert_func(Lognormal(420, 126).ppf(.6), 433.35)
        assert_func(Lognormal(420, 126).ppf(.8), 515.03)

        assert_func(Lognormal(420, 126).cdf(200), 0.00864, tol=1e-3)
        assert_func(Lognormal(420, 126).cdf(400), 0.49225, tol=1e-3)
        assert_func(Lognormal(420, 126).cdf(500), 0.77056, tol=1e-3)
        assert_func(Lognormal(420, 126).cdf(600), 0.91337, tol=1e-3)
        assert_func(Lognormal(420, 126).cdf(800), 0.99040, tol=1e-3)


if __name__ == '__main__':
    Lognormal.test()


class Arcsine(DistFunc):
    @staticmethod
    def _pdf(x, mean, std_dev):
        # Calculate a and b from the mean and standard deviation
        a = mean - np.sqrt(2) * std_dev
        b = mean + np.sqrt(2) * std_dev

        # Ensure that x is within the range [a, b]
        # if x < a or x > b:
        #     raise ValueError("x must be within the range [a, b]")

        # Compute the PDF of the arcsine distribution
        pdf_value = 1 / (np.pi * np.sqrt((x - a) * (b - x)))

        return pdf_value

    @staticmethod
    def _cdf(x, mean, std_dev):
        # Calculate a and b from the mean and standard deviation
        a = mean - np.sqrt(2) * std_dev
        b = mean + np.sqrt(2) * std_dev

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
    def _ppf(p, mean, std_dev):
        # Calculate a and b from the mean and standard deviation
        a = mean - np.sqrt(2) * std_dev
        b = mean + np.sqrt(2) * std_dev

        # Compute the PPF of the arcsine distribution
        ppf_value = a + (b - a) * (np.sin(np.pi * p / 2)) ** 2

        return ppf_value

    @staticmethod
    def test():
        d = Arcsine(420, 126)
        assert_func(d.pdf(275.841), 0.00303911, tol=1e-3)
        assert_func(d.pdf(347.92), 0.00195328, tol=1e-3)
        assert_func(d.pdf(420), 0.00178634, tol=1e-3)
        assert_func(d.pdf(492.08), 0.00195328, tol=1e-3)
        assert_func(d.pdf(564.159), 0.00303911, tol=1e-3)
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
    def _pdf(x, mu, sigma):
        return 1 / (np.pi * sigma * (1 + ((x - mu) / sigma) ** 2))

    @staticmethod
    def _cdf(x, mu, sigma):
        return 1 / np.pi * np.arctan((x - mu) / sigma) + 0.5

    @staticmethod
    def _ppf(p, mu, sigma):
        return mu + sigma * np.tan(np.pi * (p - 0.5))

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
    def _pdf(x, mu, sigma):
        return (1 / (2 * sigma)) * (1 / np.cosh(np.pi / 2 * ((x - mu) / sigma)))

    @staticmethod
    def _cdf(x, mu, sigma):
        return (2 / np.pi) * np.arctan(np.exp(np.pi / 2 * ((x - mu) / sigma)))

    @staticmethod
    def _ppf(p, mu, sigma):
        # Check for valid input
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1)")

        # Define the function we want to find the root of
        def f(x):
            return HyperbolicSecant._cdf(x, mu, sigma) - p

        # Initial boundaries for bisection method
        lower, upper = mu - 10 * sigma, mu + 10 * sigma

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
    def _pdf(x, mu, sigma):
        if x < mu:
            return 0
        else:
            return (2 / (np.pi * sigma)) / (1 + ((x - mu) / sigma) ** 2)

    @staticmethod
    def _cdf(x, mean, std_dev):
        if x < mean:
            return 0
        else:
            return 2 / np.pi * np.arctan((x - mean) / std_dev)

    @staticmethod
    def _ppf(q, mean, std_dev):
        """
        Returns the value of the percent point function (also called inverse cumulative function) for half-Cauchy distribution.
        """
        return mean + std_dev * np.tan(np.pi / 2 * q)

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


class Maxwell(DistFunc):
    @staticmethod
    def _pdf(x, mu, sigma):
        # Rescale and shift x
        x = (x - mu) / sigma

        # Probability Density Function
        return np.sqrt(2 / np.pi) * (x ** 2) * np.exp(-x ** 2 / 2)

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
        pass  # todo


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
