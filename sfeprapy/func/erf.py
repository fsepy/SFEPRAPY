# credit: https://github.com/dougthor42/PyErf
# modified by Ian 2023, implemented numpy vectorisation

import math
from math import inf

import numpy as np

ROOT_2PI = math.sqrt(2 * np.pi)
EXP_NEG2 = math.exp(-2)
MAXVAL = 1e50  #: Inputs above this value are considered infinity.


@np.vectorize
def erf(x):
    """
    Port of cephes ``ndtr.c`` ``erf`` function.

    See https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtr.c
    """
    T = [
        9.60497373987051638749E0,
        9.00260197203842689217E1,
        2.23200534594684319226E3,
        7.00332514112805075473E3,
        5.55923013010394962768E4,
    ]

    U = [
        3.35617141647503099647E1,
        5.21357949780152679795E2,
        4.59432382970980127987E3,
        2.26290000613890934246E4,
        4.92673942608635921086E4,
    ]

    # Shorcut special cases
    if x == 0:
        return 0
    if x >= MAXVAL:
        return 1
    if x <= -MAXVAL:
        return -1

    if abs(x) > 1:
        return 1 - erfc(x)

    z = x * x
    return x * _polevl(z, T, 4) / _p1evl(z, U, 5)


@np.vectorize
def erfc(a):
    """
    Port of cephes ``ndtr.c`` ``erfc`` function.

    See https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtr.c
    """
    # approximation for abs(a) < 8 and abs(a) >= 1
    P = [
        2.46196981473530512524E-10,
        5.64189564831068821977E-1,
        7.46321056442269912687E0,
        4.86371970985681366614E1,
        1.96520832956077098242E2,
        5.26445194995477358631E2,
        9.34528527171957607540E2,
        1.02755188689515710272E3,
        5.57535335369399327526E2,
    ]

    Q = [
        1.32281951154744992508E1,
        8.67072140885989742329E1,
        3.54937778887819891062E2,
        9.75708501743205489753E2,
        1.82390916687909736289E3,
        2.24633760818710981792E3,
        1.65666309194161350182E3,
        5.57535340817727675546E2,
    ]

    # approximation for abs(a) >= 8
    R = [
        5.64189583547755073984E-1,
        1.27536670759978104416E0,
        5.01905042251180477414E0,
        6.16021097993053585195E0,
        7.40974269950448939160E0,
        2.97886665372100240670E0,
    ]

    S = [
        2.26052863220117276590E0,
        9.39603524938001434673E0,
        1.20489539808096656605E1,
        1.70814450747565897222E1,
        9.60896809063285878198E0,
        3.36907645100081516050E0,
    ]

    # Shortcut special cases
    if a == 0:
        return 1
    if a >= MAXVAL:
        return 0
    if a <= -MAXVAL:
        return 2

    x = a
    if a < 0:
        x = -a

    # computationally cheaper to calculate erf for small values, I guess.
    if x < 1:
        return 1 - erf(a)

    z = -a * a

    z = math.exp(z)

    if x < 8:
        p = _polevl(x, P, 8)
        q = _p1evl(x, Q, 8)
    else:
        p = _polevl(x, R, 5)
        q = _p1evl(x, S, 6)

    y = (z * p) / q

    if a < 0:
        y = 2 - y

    return y


def _polevl(x, coefs, N):
    """
    Port of cephes ``polevl.c``: evaluate polynomial

    See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
    """
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        try:
            ans += coef * x ** power
        except OverflowError:
            pass
        power -= 1
    return ans


def _p1evl(x, coefs, N):
    """
    Port of cephes ``polevl.c``: evaluate polynomial, assuming coef[N] = 1

    See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
    """
    return _polevl(x, [1] + coefs, N)


def _ndtri(y):
    """
    Port of cephes ``ndtri.c``: inverse normal distribution function.

    See https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtri.c
    """
    # approximation for 0 <= abs(z - 0.5) <= 3/8
    P0 = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ]

    Q0 = [
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ]

    # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
    # i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
    P1 = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ]

    Q1 = [
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ]

    # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
    # i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
    P2 = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ]

    Q2 = [
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ]

    sign_flag = 1

    if y > (1 - EXP_NEG2):
        y = 1 - y
        sign_flag = 0

    # Shortcut case where we don't need high precision
    # between -0.135 and 0.135
    if y > EXP_NEG2:
        y -= 0.5
        y2 = y ** 2
        x = y + y * (y2 * _polevl(y2, P0, 4) / _p1evl(y2, Q0, 8))
        x = x * ROOT_2PI
        return x

    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x

    z = 1.0 / x
    if x < 8.0:  # y > exp(-32) = 1.2664165549e-14
        x1 = z * _polevl(z, P1, 8) / _p1evl(z, Q1, 8)
    else:
        x1 = z * _polevl(z, P2, 8) / _p1evl(z, Q2, 8)

    x = x0 - x1
    if sign_flag != 0:
        x = -x

    return x


@np.vectorize
def erfinv(z):
    """
    Calculate the inverse error function at point ``z``.

    This is a direct port of the SciPy ``erfinv`` function, originally
    written in C.

    Parameters
    ----------
    z : numeric

    Returns
    -------
    float

    References
    ----------
    + https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    + http://functions.wolfram.com/GammaBetaErf/InverseErf/

    Examples
    --------
    >>> round(erfinv(0.1), 12)
    0.088855990494
    >>> round(erfinv(0.5), 12)
    0.476936276204
    >>> round(erfinv(-0.5), 12)
    -0.476936276204
    >>> round(erfinv(0.95), 12)
    1.38590382435
    >>> round(erf(erfinv(0.3)), 3)
    0.3
    >>> round(erfinv(erf(0.5)), 3)
    0.5
    >>> erfinv(0)
    0
    >>> erfinv(1)
    inf
    >>> erfinv(-1)
    -inf
    """
    if abs(z) > 1:
        raise ValueError("`z` must be between -1 and 1 inclusive")

    # Shortcut special cases
    if z == 0:
        return 0
    if z == 1:
        return inf
    if z == -1:
        return -inf

    # otherwise calculate things.
    return _ndtri((z + 1) / 2.0) / math.sqrt(2)
