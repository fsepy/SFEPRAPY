from abc import ABC, abstractmethod
from typing import Union

import numpy as np

__all__ = 'Normal', 'Gumbel', 'Lognormal', 'Arcsine', 'Cauchy', 'HyperbolicSecant', 'HalfCauchy', 'Logistic', 'Uniform'

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
        sigma = stddev * np.sqrt(6) / np.pi
        mu = mean - 0.57721566490153286060 * sigma
        return mu, sigma

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


class Uniform(DistFunc):
    @staticmethod
    def _pdf(x, mean, std):
        """Probability density function"""
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std
        return np.where((x >= a) & (x <= b), 1 / (b - a), 0)

    @staticmethod
    def _cdf(x, mean, std):
        """Cumulative distribution function"""
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std
        return np.where(x < a, 0, np.where(x > b, 1, (x - a) / (b - a)))

    @staticmethod
    def _ppf(p, mean, std):
        """Percent-point function (Inverse of cdf)"""
        # Ensure p is in [0, 1]
        p = np.clip(p, 0, 1)
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std
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
