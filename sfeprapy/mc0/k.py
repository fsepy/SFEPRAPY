
# Equation 37: Mass flow rate of smoke from a vertical opening
def m_dot_smoke_func(
        C_e,
        p,
        h,
        w,
        C_d,
        rho_0,
        T_max,
        g,
        theta_max,
        T_0,
        kappa_m
):
    """
    :param C_e:
    :param p:
    :param h: Compartment opening.
    :param w:
    :param C_d: is a discharge coefficient for the opening, 0.6 for an opening with a deep down-stand or 1.0 for no down-stand.
    :param rho_0:
    :param T_max:
    :param g:
    :param theta_max:
    :param T_0:
    :param kappa_m: is a 'profile correction factor' to allow for departure from a step function for the vertical temperature gradient (kappa_m is 1.0 for a step function but can be taken as 1.3 for most hot gas layers where theta_max is less than 300 deg.C).
    :return:
    """
    aa = 2 / 3 * rho_0 / T_max
    bb = (2 * g * theta_max * T_0 * kappa_m) ** 0.5
    k = aa * bb

    a = C_e * p * h ** (1.5) * w
    b = w ** (2 / 3)
    c = 1 / (C_d)
    d = ((C_e * p) / k) ** (2 / 3)
    m_dot_smoke = a / (b + c * d) ** (1.5)

    return m_dot_smoke


# Equation 54. Theoretically-based design formula (utilizing an empirical entrainment coefficient from the same experiments) for the balcony spill plume with channelling screens.
def m_dot_smoke_func_54(
        eta, E, Q_dot_c, W_s, z_s, d_s
):
    # Equation 57. The virtual source.
    z_0 = 3 * d_s

    m_dot_smoke = eta * (2 * E) ** (2/3) * (Q_dot_c) ** (1/3) * (W_s) ** (2/3) * (z_s + z_0) * (1 + 2 * E * (z_s / W_s)) ** (2/3)
    return m_dot_smoke


# Equation 55. It is recommended in Air entrainment into balcony spill plumes [29], for the two-dimensional channelled balcony spill plume that E = 0.11 and eta = 0.42. Therefore, in the linear entrainment region (0 << z_s << 5 * W_s) for the two dimensional channelled balcony spill plume, equation 54 approximates to
def m_dot_smoke_func_55(
        Q_dot_c, W_s, z_s, d_s
):
    # Equation 57. The virtual source.
    z_0 = 3 * d_s

    m_dot_smoke = 0.15 * (Q_dot_c) ** (1/3) * (W_s) ** (2/3) * (z_s + z_0)
    return m_dot_smoke


# Eq uation 56. It is recommended in Air entrainment into balcony spill plumes [29], for the three dimensional channelled balcony spill plume that E = 0.13 and eta = 0.42. In the linear entrainment region (0 << z_s << 5 * W_s) for the three-dimensional channelled balcony spill plume, equation 54 approximates to.
def m_dot_smoke_func_56(
        Q_dot_c, W_s, z_s, d_s
):
    # Equation 57. The virtual source.
    z_0 = 3 * d_s

    m_dot_smoke = 0.17 * (Q_dot_c) ** (1/3) * (W_s) ** (2/3) * (z_s + z_0)
    return m_dot_smoke
