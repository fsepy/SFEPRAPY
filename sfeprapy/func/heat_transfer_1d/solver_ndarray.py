import numpy as np


def solver_exposed(
        dx: np.float64,
        U_solid: np.float64,
        U_gas: np.float64,
        rho: np.float64,
        c: np.float64,
        h: np.float64,
        e: np.float64,
        dt: np.float64
):
    u0 = U_gas - U_solid
    u0_4 = U_gas ** 4 - U_solid ** 4
    e0 = u0 * h
    e2 = u0_4 * e
    e3 = rho * c * dx
    dU_dt_0 = (e0 + e2) / e3
    U_solid = U_solid + dU_dt_0 * dt

    return U_solid


def solver_solid(
        dx: np.ndarray,
        U: np.ndarray,
        k: np.ndarray,
        rho: np.ndarray,
        c: np.ndarray,
        dt: float
):
    """
    Description
    1-d heat transfer, finite difference, non-propagation approach, entire U at next time step n+1 is based upon U at
    the current temperature U.

    Dimension Indexing
    m: space dimension in x-axis, x[m=0] origin, x[m=X] end.
    n: time dimension, t[n=0] is beginning, t[n=T] is end.

    :param dx: length dimension in x-axis direction of each nodes
    :param U: temperature at each nodes
    :param k: thermal conductivities at each nodes
    :param rho: density at each nodes
    :param c: thermal heat capacity at each nodes
    :param dt: change in time
    :return U: new temperature at each nodes
    """

    U_new = np.zeros_like(U)

    du1 = U[0] - U[1]
    e1 = - du1 * k[0]
    en = rho[0] * c[0] * dx[0]
    U_new[0] = U[0] + e1 / en * dt

    du0 = U[-2] - U[-1]
    e1 = du0 * k[-1]
    en = rho[-1] * c[-1] * dx[-1]
    U_new[-1] = U[-1] + e1 / en * dt

    for m in range(1, len(dx) - 1):
        du0 = U[m - 1] - U[m]
        du1 = U[m] - U[m + 1]
        e1 = du0 * k[m] - du1 * k[m]  # todo: thermal properties can be averaged between adjacent nodes, not on current
        en = rho[m] * c[m] * dx[m]
        du_dt = e1 / en
        U_new[m] = U[m] + du_dt * dt

    return U_new
