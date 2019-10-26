# Sub-functions for 1D heat transfer code
# __main__ will main_args a non-linear 1D heat transfer analysis
# Danny Hopkin
# OFR Consultants
# 15/05/2019

#  Conversion from DegC to DegK

import numpy as np
from matplotlib import pyplot as plt


def ISO834_ft(t):

    #  returns the ISO curve where t is [s]
    tmin = t / 60
    tiso = 345 * np.log10((8 * tmin) + 1) + 20
    return tiso


def Common_CtoK(T):

    # Returns Kelvin

    TK = T + 273
    return TK


#  Calculation of flux abosrbed by first element


def ONEDHT_QINC(TGAS, TSURF, EMIS, hc):

    # Get absorbed heat flux (in Watts)

    QRAD = (
        0.0000000567 * EMIS * ((Common_CtoK(TGAS) ** 4) - (Common_CtoK(TSURF) ** 4))
    )  # Radiation component
    QCONV = hc * (Common_CtoK(TGAS) - Common_CtoK(TSURF))  # Convective component
    QINC = QRAD + QCONV  # Sum components and return value
    return QINC


# Calculation of heat flux leaving unexposed element


def ONEDHT_QOUT(TSURF, TAMB, EMIS, hc):

    # Heat flux lost to ambient on unexposed face (in Watts)

    QRAD = (
        0.0000000567 * EMIS * ((Common_CtoK(TSURF) ** 4) - (Common_CtoK(TAMB) ** 4))
    )  # Radiation component
    QCONV = hc * (Common_CtoK(TSURF) - Common_CtoK(TAMB))  # Convective component
    QOUT = QRAD + QCONV  # Sum components and return value
    return QOUT


# Calculation of temperature of first element


def ONEDHT_ELEM1(Qinc, T1, T2, LAMDA1, LAMDA2, dx, dt, Cp, Rho):

    # Calculate temperature of first element given incoming flux Q1

    a1 = (2 * dt) / (Rho * Cp * dx)  # calculate diffusivity
    a2 = (LAMDA1 + LAMDA2) / 2  # calculate mean conductivity
    a3 = (T1 - T2) / dx  # Calculate temperature gradient

    dT1 = a1 * (Qinc - (a2 * a3))  #  Calculate change in temperature in time step

    T1new = max(T1 + dT1, 20)  # Calculate new temperature

    return T1new


# Calculation of temperature for intermediate elements


def ONEDHT_ELEMJ(TJN1, TJ, TJP1, LAMDAJN1, LAMDAJ, LAMDAJP1, dx, dt, Cp, Rho):

    # Calculate temperature of element j, using temps for j-1 and j+1
    b1 = dt / (Rho * Cp * (dx ** 2))  # calculate diffusivity
    b2 = (LAMDAJN1 + LAMDAJ) / 2  # calculate mean conductivity1
    b3 = TJN1 - TJ  # Calculate delta T1
    b4 = (LAMDAJ + LAMDAJP1) / 2  # calculate mean conductivity2
    b5 = TJ - TJP1  # Calculate delta T2
    dTJ = b1 * ((b2 * b3) - (b4 * b5))  # Calculate change in temperature in time step
    Tjnew = TJ + dTJ  # Calculate new temperature
    return Tjnew


# Calculation of temperature of final element


def ONEDHT_ELEMF(Qout, TFN1, TF, LAMDAFN1, LAMDAF, dx, dt, Cp, Rho):

    # Calculate temperature of the final element

    c1 = (2 * dt) / (Rho * Cp * dx)  # calculate diffusivity
    c2 = (LAMDAFN1 + LAMDAF) / 2  # calculate mean conductivity
    c3 = (TFN1 - TF) / dx  # Calculate temperature gradient

    dTF = c1 * ((c2 * c3) - Qout)  # calculate change in temperature in time step
    Tfnew = TF + dTF  # calculate new temperature
    return Tfnew


import warnings


def c_steel_T(temperature):

    if temperature < 20:
        warnings.warn("Temperature ({:.1f} °C) is below 20 °C".format(temperature))
        return 425 + 0.773 * 20 - 1.69e-3 * np.power(20, 2) + 2.22e-6 * np.power(20, 3)
    if 20 <= temperature < 600:
        return (
            425
            + 0.773 * temperature
            - 1.69e-3 * np.power(temperature, 2)
            + 2.22e-6 * np.power(temperature, 3)
        )
    elif 600 <= temperature < 735:
        return 666 + 13002 / (738 - temperature)
    elif 735 <= temperature < 900:
        return 545 + 17820 / (temperature - 731)
    elif 900 <= temperature <= 1200:
        return 650
    elif temperature > 1200:
        warnings.warn(
            "Temperature ({:.1f} °C) is greater than 1200 °C".format(temperature)
        )
        return 650
    else:
        warnings.warn("Temperature ({:.1f} °C) is outside bound.".format(temperature))
        return 0


def k_steel_T(temperature):
    if temperature < 20:
        warnings.warn("Temperature ({:.1f} °C) is below 20 °C".format(temperature))
        return 54 - 3.33e-2 * 20
    if temperature < 800:
        return 54 - 3.33e-2 * temperature
    elif temperature <= 1200:
        return 27.3
    elif temperature > 1200:
        warnings.warn(
            "Temperature ({:.1f} °C) is greater than 1200 °C".format(temperature)
        )
        return 27.3
    else:
        warnings.warn("Temperature ({:.1f} °C) is outside bound.".format(temperature))
        return 0


if __name__ == "__main__":

    import copy
    from scipy.interpolate import interp1d

    # Set geometrical parameters & simulation time frame

    nodes = 100  # set the number of nodes
    dx = 0.1 / nodes  # in m
    temp_arr = (
        np.ones(nodes) * 20
    )  # creates initial temperature nodal array with ambient condition
    endtime = 600  # number of timesteps
    dt = 0.025  # time step increment
    nth = 1000  # Temperature profile will be produced every nth time step
    endtimestep = int(endtime / dt)  # calculate total number of timesteps
    node_array = np.ones(nodes)  # Create depth array for temperature profile

    # Set heat transfer properties based upon initial temperature conditions

    lamda_temp_arr = [0, 1200]  #  Temperature array for conductivity [DegC]
    lamda_input_arr = [40, 40]  #  Corresponding conductivity input [W/m.K]
    # lamda_temp_arr = [19,500,1200]                              #  Temperature array for conductivity [DegC]
    # lamda_input_arr = [0.12,0.11, 1]                             #  Corresponding conductivity input [W/m.K]
    lamda_interp = interp1d(lamda_temp_arr, lamda_input_arr)  #  Interpolation function

    cp_temp_arr = [0, 1200]  #  Temperature array for specific heat [DegC]
    cp_input_arr = [450, 450]  #  Corresponding specific heat input [J/kg.K]
    # cp_temp_arr = [19, 99, 100, 120, 121, 1200]                 #  Temperature array for specific heat [DegC]
    # cp_input_arr =[950, 950, 2000, 2000, 950, 950]              #  Corresponding specific heat input [J/kg.K]
    cp_interp = interp1d(cp_temp_arr, cp_input_arr)  #  Interpolation function

    rho_temp_arr = [0, 1200]  #  Temperature array for density [DegC]
    rho_input_arr = [9850, 9850]
    # rho_temp_arr = [19,1200]                                    #  Temperature array for density [DegC]
    # rho_input_arr =[450,400]                                    #  Corresponding density input [kg/m3]
    rho_interp = interp1d(rho_temp_arr, rho_input_arr)  #  Interpolation function

    # Boundary conditions

    hchot = 25  #  hot side convection coefficient [W/m2.K]
    hccold = 9  # cold side convection coefficient [W/m2.K]
    effemis = 0.8  # net emissivity - applies to all faces [-]
    tamb = 20  # ambient temp on unexposed face [DegC]

    list_temp_arr = list()  # Create storage matrix

    #  Main heat transfer solver

    print("SOLVING")

    for tstep in range(1, endtimestep + 1):

        # Update thermal property array at each time time
        print(temp_arr)

        # lamdaNL = lamda_interp(temp_arr)
        # cpNL = cp_interp(temp_arr)
        # rhoNL = rho_interp(temp_arr)
        lamdaNL = np.array([k_steel_T(temp) for temp in temp_arr])
        cpNL = np.array([c_steel_T(temp) for temp in temp_arr])
        rhoNL = np.full_like(temp_arr, 7850)

        time = dt * tstep  #  current simulation time
        gas_temp = ISO834_ft(time)  #  gas temp in DegC
        gas_temp = 1000

        if tstep / 10 == int(tstep / 10):
            print("Time = ", time, " s ", "Gas temperature = ", gas_temp, " DegC")

        # First element calculations

        Qinc = ONEDHT_QINC(gas_temp, temp_arr[0], effemis, hchot)
        temp_arr[0] = ONEDHT_ELEM1(
            Qinc,
            temp_arr[0],
            temp_arr[1],
            lamdaNL[0],
            lamdaNL[1],
            dx,
            dt,
            cpNL[0],
            rhoNL[0],
        )

        # Intermediate element calculations

        for nodenum in range(1, nodes - 1):
            temp_arr[nodenum] = ONEDHT_ELEMJ(
                temp_arr[nodenum - 1],
                temp_arr[nodenum],
                temp_arr[nodenum + 1],
                lamdaNL[nodenum - 1],
                lamdaNL[nodenum],
                lamdaNL[nodenum + 1],
                dx,
                dt,
                cpNL[nodenum],
                rhoNL[nodenum],
            )

        # Final element calculations

        Qout = ONEDHT_QOUT(temp_arr[nodes - 1], tamb, effemis, hccold)
        temp_arr[nodes - 1] = ONEDHT_ELEMF(
            Qout,
            temp_arr[nodes - 2],
            temp_arr[nodes - 1],
            lamdaNL[nodes - 2],
            lamdaNL[nodes - 1],
            dx,
            dt,
            cpNL[nodes - 1],
            rhoNL[nodes - 1],
        )

        # Store results

        list_temp_arr.append(copy.copy(temp_arr))

    # Generate depth array [mm]

    for x in range(0, nodes):
        node_array[x] = x * dx * 1000

    # Plot temperature profiles at every nth time step

    import seaborn as sns

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, v in enumerate(list_temp_arr):
        if i / nth == int(i / nth):
            ax.plot(node_array, v)
            ax.grid(True)
            ax.set_xlabel("Depth [mm]")
            ax.set_ylabel("Temperature [$^\circ C$]")

    ax.plot(node_array, list_temp_arr[-1])

    plt.tight_layout()
    plt.show()
