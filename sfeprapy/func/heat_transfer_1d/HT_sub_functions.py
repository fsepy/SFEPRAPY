# Sub-functions for 1D heat transfer code
# __main__ will run a non-linear 1D heat transfer analysis
# Danny Hopkin
# OFR Consultants
# 15/05/2019

#  Conversion from DegC to DegK

import numpy as np
from matplotlib import pyplot as plt

def ISO834_ft(t):

    #  returns the ISO curve where t is [s]
    tmin = t/60
    tiso = 345*np.log10((8*tmin)+1)+20
    return tiso

def Common_CtoK(T):

    # Returns Kelvin

    TK = T + 273
    return TK

#  Calculation of flux abosrbed by first element

def ONEDHT_QINC(TGAS, TSURF, EMIS, hc):

    # Get absorbed heat flux (in Watts)

    QRAD = 0.0000000567 * EMIS * ((Common_CtoK(TGAS) ** 4) - (Common_CtoK(TSURF) ** 4))  # Radiation component
    QCONV = hc * (Common_CtoK(TGAS) - Common_CtoK(TSURF))  # Convective component
    QINC = QRAD + QCONV  # Sum components and return value
    return QINC

# Calculation of heat flux leaving unexposed element

def ONEDHT_QOUT(TSURF, TAMB, EMIS, hc):

    # Heat flux lost to ambient on unexposed face (in Watts)

    QRAD = 0.0000000567 * EMIS * ((Common_CtoK(TSURF) ** 4) - (Common_CtoK(TAMB) ** 4))  # Radiation component
    QCONV = hc * (Common_CtoK(TSURF) - Common_CtoK(TAMB))  # Convective component
    QOUT = QRAD + QCONV  # Sum components and return value
    return QOUT

# Calculation of temperature of first element

def ONEDHT_ELEM1(Qinc, T1, T2, LAMDA1, LAMDA2, dx, dt, Cp, Rho):

    # Calculate temperature of first element given incoming flux Q1

    a1 = (2 * dt) / (Rho * Cp * dx) # calculate diffusivity
    a2 = (LAMDA1 + LAMDA2) / 2 # calculate mean conductivity
    a3 = (T1 - T2) / dx # Calculate temperature gradient

    dT1 = a1 * (Qinc - (a2 * a3)) #  Calculate change in temperature in time step

    T1new = max(T1 + dT1, 20) #Calculate new temperature

    return T1new

# Calculation of temperature for intermediate elements

def ONEDHT_ELEMJ(TJN1, TJ, TJP1, LAMDAJN1, LAMDAJ, LAMDAJP1, dx, dt, Cp, Rho):

    # Calculate temperature of element j, using temps for j-1 and j+1
    b1 = dt / (Rho * Cp * (dx **2)) # calculate diffusivity
    b2 = (LAMDAJN1 + LAMDAJ)/2 # calculate mean conductivity1
    b3 = TJN1 - TJ # Calculate delta T1
    b4 = (LAMDAJ + LAMDAJP1)/2 # calculate mean conductivity2
    b5 = TJ - TJP1 # Calculate delta T2
    dTJ = b1 * ((b2 * b3) - (b4 * b5)) # Calculate change in temperature in time step
    Tjnew = TJ + dTJ # Calculate new temperature
    return Tjnew

# Calculation of temperature of final element

def ONEDHT_ELEMF(Qout, TFN1, TF, LAMDAFN1, LAMDAF, dx, dt, Cp, Rho):

    # Calculate temperature of the final element

    c1 = (2 * dt) / (Rho * Cp * dx) # calculate diffusivity
    c2 = (LAMDAFN1 + LAMDAF)/2 # calculate mean conductivity
    c3 = (TFN1 - TF) / dx # Calculate temperature gradient

    dTF = c1 * ((c2 * c3) - Qout) # calculate change in temperature in time step
    Tfnew = TF + dTF # calculate new temperature
    return Tfnew

if __name__ == '__main__':

    import copy
    from scipy.interpolate import interp1d

    # Set geometrical parameters & simulation time frame

    nodes = 11      # set the number of nodes
    dx = 0.0025     # in m
    temp_arr = np.ones(nodes)*20    # creates initial temperature nodal array with ambient condition
    endtime = 1800   # number of timesteps
    dt = 1          # time step increment - very important for numerical stability
    nth = 300       # Temperature profile will be produced every nth time step
    endtimestep = int(endtime/dt)   # calculate total number of timesteps
    node_array =np.ones(nodes)      # Create depth array for temperature profile
    track_iso = 300                 # Isotherm depth to be tracked

    # Set heat transfer properties based upon initial temperature conditions

    lamda_temp_arr = [19,1200]                              #  Temperature array for conductivity [DegC]
    lamda_input_arr =[0.09,0.09]                             #  Corresponding conductivity input [W/m.K]
    lamda_interp = interp1d(lamda_temp_arr,lamda_input_arr)     #  Interpolation function

    cp_temp_arr = [19, 1200]                 #  Temperature array for specific heat [DegC]
    cp_input_arr =[950, 950]              #  Corresponding specific heat input [J/kg.K]
    cp_interp = interp1d(cp_temp_arr,cp_input_arr)              #  Interpolation function

    rho_temp_arr = [19, 1200]                                    #  Temperature array for density [DegC]
    rho_input_arr =[1,1]                                    #  Corresponding density ratio input [-]
    rho_interp = interp1d(rho_temp_arr,rho_input_arr)           #  Interpolation function
    density = 500                                               #  Density in [kg/m3]

    # Boundary conditions

    hchot = 25                                                  # hot side convection coefficient [W/m2.K]
    hccold = 9                                                  # cold side convection coefficient [W/m2.K]
    effemish = 0.8                                              # net emissivity - cold faces [-]
    effemisc = 0.8                                              # net emissivity - cold faces [-]
    tamb = 20                                                   # ambient temp on unexposed face [DegC]

    list_temp_arr = list()  # Create storage matrix
    isotherm_arr=[]

    # Generate depth array [mm]

    for x in range(0,nodes):
        node_array[x]=x*dx*1000

    #  Main heat transfer solver

    print('SOLVING')

    for tstep in range(1,endtimestep+1):

        # Update thermal property array at each time time

        lamdaNL = lamda_interp(temp_arr)
        cpNL = cp_interp(temp_arr)
        rhoNL = rho_interp(temp_arr)*density

        time = dt*tstep                #  current simulation time
        gas_temp = ISO834_ft(time)     #  gas temp in DegC

        # First element calculations

        Qinc = ONEDHT_QINC(gas_temp,temp_arr[0],effemish,hchot)
        temp_arr[0] = ONEDHT_ELEM1(Qinc, temp_arr[0], temp_arr[1],lamdaNL[0],lamdaNL[1],dx,dt,cpNL[0],rhoNL[0])

        # Intermediate element calculations

        for nodenum in range(1,nodes-1):
            temp_arr[nodenum] = ONEDHT_ELEMJ(temp_arr[nodenum-1],temp_arr[nodenum],temp_arr[nodenum+1],lamdaNL[nodenum-1],lamdaNL[nodenum],lamdaNL[nodenum+1],dx,dt,cpNL[nodenum],rhoNL[nodenum])

        # Final element calculations

        Qout  = ONEDHT_QOUT(temp_arr[nodes-1],tamb,effemisc,hccold)
        temp_arr[nodes-1] = ONEDHT_ELEMF(Qout,temp_arr[nodes-2],temp_arr[nodes-1],lamdaNL[nodes-2],lamdaNL[nodes-1],dx,dt,cpNL[nodes-1],rhoNL[nodes-1])

        # Store results

        list_temp_arr.append(copy.copy(temp_arr))

        # Track isotherm

        isotherm_interp = interp1d(temp_arr, node_array)

        if max(temp_arr)<300:
            iso_pos = 0
        else:
            iso_pos = isotherm_interp(300)

        isotherm_arr.append(float(iso_pos))

        if tstep / 10 == int(tstep / 10):
            print('Time = ', time, ' s ', 'Gas temperature = ', gas_temp, ' DegC', 'Isotherm depth = ', iso_pos, 'mm')

    # Plot temperature profiles at every nth time step
    print(isotherm_arr)
    import seaborn as sns
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(4,3))
    for i, v in enumerate(list_temp_arr):
        if i/nth == int(i/nth):
            if i > 1:
                ax.plot(node_array, v, label=str(i*dt))
    ax.plot(node_array, temp_arr, label=str(endtime))
    ax.yaxis.set_ticks(np.arange(0, 900, 100))
    ax.grid(True)
    ax.set_xlabel('Depth [mm]')
    ax.set_ylabel('Temperature [$^\circ C$]')
    ax.legend(loc=0, fontsize=9).set_visible(True)
    plt.tight_layout()
    plt.show()