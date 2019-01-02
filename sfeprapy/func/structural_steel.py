# -*- coding: utf-8 -*-
"""Resistance to lateral-torsional buckling

This module is created to aid the calculation principles for structural steel in fire situation set out in EC 1,
PART 1-2. More detailed instruction can be found in F. Jean-Marc & R.P. Vila (2010) - Design of Steel Structures, which
the code in this module compliant with. Section 5.4 is particularly followed.
"""
import numpy as np
from sfeprapy.dat.steel_carbon import thermal


def _M_cr(C_1, C_2, I_z, k_z, k_w, I_w, I_t, L, E, G, z_g):
    """
    [Eq. 5.59]
    :param C_1: Constant, found in Table 5.4 or ECCS (2006)
    :param C_2: ECCS (2006) or in Galea Y (2002)
    :param I_z: Second moment of area about the minor axis
    :param k_z: 0.5 for full restraint, 0.7 for one end fixed and one pinned, 1.0 for both pinned
    :param k_w: 0.5 for full restraint, 0.7 for one end fixed and one pinned, 1.0 for both pinned
    :param I_w: Warping constant
    :param I_t: Torsion constant
    :param L:   Beam/column length
    :param E:   Steel modulus
    :param G:   Steel ?
    :param z_g: Factor governed by the load location on the section, see Fig. 5.24
    :return M_cr: The elastic critical moment for a uniform moment
    """
    a = C_1 * (np.pi**2 * E * I_z) / (k_z * L)**2
    b = (k_z / k_w)**2 * (I_w / I_z)
    c = (k_z * L)**2 * G * I_t / np.pi**2 / E / I_z
    d = (C_2 * z_g) ** 2
    e = C_2 * z_g

    M_cr = a * (b+c+d)**0.5 - e

    return M_cr


def _lambda_LT(W_pl_y, f_y, M_cr):
    """
    [Eq. 5.66]
    :param W_pl_y:      Plastic section modulus
    :param f_y:         Yield strength
    :param M_cr:        Critical bending moment, see [Eq. 5.59]
    :return lambda_LT:  Non-dimensional slenderness at normal temperature
    """

    lambda_LT = (W_pl_y * f_y / M_cr)**2

    return lambda_LT


def _lambda_LT_theta_com(lambda_LT, k_y_theta_com, k_E_theta_com):
    """
    [Eq. 5.65]
    :param lambda_LT: Non-dimensional slenderness at normal temperature
    :param k_y_theta_com: Reduction factor for Young's modulus at the maximum steel temp. in the com. flange
    :param k_E_theta_com: Reduction factor for Young's modulus at the maximum steel temp. in the com. flange
    :return lambda_LT_theta_com: Non-dimensional slenderness at elevated temperature
    """
    lambda_LT_theta_com = lambda_LT * (k_y_theta_com / k_E_theta_com)

    return lambda_LT_theta_com

def _alpha(f_y):
    """
    [Eq. 5.64]
    :param f_y: Steel yield strength
    :return alpha: The imperfection factor
    """

    alpha = 0.65 * (235/f_y) ** 0.5

    return alpha

def _phi_LT_theta_com(alpha, lambda_LT_theta_com):
    """
    [Eq. 5.63]
    :param alpha: The imperfection factor
    :param lambda_LT_theta_com: Non-dimensional slenderness at elevated temeperature
    :return phi_LT_theta_com: A variable used in [5.64]
    """

    phi_LT_theta_com = 0.5 * (1 + alpha * lambda_LT_theta_com + lambda_LT_theta_com**2)

    return phi_LT_theta_com


def _chi_LT_fi(phi_LT_theta_com, lambda_LT_theta_com):
    """
    [Eq. 5.62]
    :param phi_LT_theta_com: Refer to [5.63]
    :param lambda_LT_theta_com: Non-dimensional slenderness at elevated temperature
    :return chi_LT_fi: Reduction factor for lateral-torsion buckling in the fire design situation
    """

    chi_LT_fi = (phi_LT_theta_com + (phi_LT_theta_com**2-lambda_LT_theta_com**2))**-0.5

    return chi_LT_fi

def _M_b_fi_t_Rd(chi_LT_fi, W_y, k_y_theta_com, f_y, gamma_M_fi):
    """
    [Eq. 5.61]
    :param chi_LT_fi: Reduction factor for lateral-torsion buckling in the fire design situation
    :param W_y: Sectional modulus (plastic for class 1 steel)
    :param k_y_theta_com: Reduction factor for yield strength at elevated temperature
    :param f_y: Steel yield strength
    :param gamma_M_fi: Partial safety factor
    :return M_b_fi_t_Rd: The resistant lateral torsion bending moment
    """

    M_b_fi_t_Rd = chi_LT_fi * W_y * k_y_theta_com * f_y / gamma_M_fi

    return M_b_fi_t_Rd


if __name__ == "__main__":
    # REQUIRED PARAMETERS
    C_1 = 1.77
    C_2 = 0.
    I_z = 11400.e-8  # m4

    k_z = 1.  # conservative
    k_w = 1.  # conservative
    I_w = 19.3e-6  # m6
    I_t = 514.e-8  # m4
    L = 2.5  # m
    E = 210e9  # Pa
    G = 81e9  # Pa
    z_g = 0.
    theta = 273.15 + 20  # K
    theta = np.arange(0, 550, 50) + 273.15

    W_py_y = 7980e-6  # m3
    W_y = W_py_y
    f_y = 345e6  # Pa

    gamma_M_fi = 1

    # Calculate: The strength reduction factor for steel in fire condition
    k_y_theta_com = thermal("reduction factor for effective yield strength")(theta)
    k_E_theta_com = thermal("reduction factor for the slope of the linear elastic range")(theta)

    # Calculate: The critical bending moment resistance
    M_cr = _M_cr(C_1, C_2, I_z, k_z, k_w, I_w, I_t, L, E, G, z_g)

    # Calculate: The non-dimensional slenderness at normal temperature
    lambda_LT = _lambda_LT(W_py_y, f_y, M_cr)

    # Calculate: The non-dimensional slenderness at elevated temperature
    lambda_LT_theta_com =_lambda_LT_theta_com(lambda_LT, k_y_theta_com, k_E_theta_com)

    # Calculate: The imperfection factor for material
    alpha = _alpha(f_y)

    # Calculate: A variable used for later
    phi_LT_theta_com = _phi_LT_theta_com(alpha, lambda_LT_theta_com)

    # Calculate: The reduction factor for lateral torsion in fire condition
    chi_LT_fi = _chi_LT_fi(phi_LT_theta_com, lambda_LT_theta_com)

    # Calculate: The lateral torsion buckling moment resistance in fire condition
    M_b_fi_t_Rd = _M_b_fi_t_Rd(chi_LT_fi, W_y, k_y_theta_com, f_y, gamma_M_fi)

    # OUTPUTS
    print("The elastic resistance moment for lateral torsion is: {:.2f} MN m.".format(M_cr * 1e-6))
    print("The design lateral-torsional buckling resistance moment is: {:.2f} MN m.".format(M_b_fi_t_Rd * 1e-6))
