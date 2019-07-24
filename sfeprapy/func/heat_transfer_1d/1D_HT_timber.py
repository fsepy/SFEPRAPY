# Imports

from .HT_sub_functions import ONEDHT_QINC, ONEDHT_QOUT, ONEDHT_ELEM1, ONEDHT_ELEMJ, ONEDHT_ELEMF
import copy
from matplotlib import pyplot as plt
import numpy as np

def wall_HT_sub(gas_temp, temp_arr, nodes, dt, dx, hchot, hccold,effemish,effemisc,tamb):

    from scipy.interpolate import interp1d

    # Set heat transfer properties based upon initial temperature conditions

    lamda_temp_arr = [0, 300, 700, 1400]             #  Temperature array for conductivity [DegC]
    lamda_input_arr =[0.12,0.2, 0.4, 1.2]         #  Corresponding conductivity input [W/m.K]
    lamda_interp = interp1d(lamda_temp_arr,lamda_input_arr)     #  Interpolation function

    cp_temp_arr = [0, 1400]                 #  Temperature array for specific heat [DegC]
    cp_input_arr =[1600,1600]              #  Corresponding specific heat input [J/kg.K]
    cp_interp = interp1d(cp_temp_arr,cp_input_arr)              #  Interpolation function

    rho_temp_arr = [0, 1400]                                    #  Temperature array for density [DegC]
    rho_input_arr =[1,1]                                    #  Corresponding density ratio input [-]
    rho_interp = interp1d(rho_temp_arr,rho_input_arr)           #  Interpolation function
    density = 500                                              #  Density in [kg/m3]

    #  Main heat transfer solver

    lamdaNL = lamda_interp(temp_arr)
    cpNL = cp_interp(temp_arr)
    rhoNL = rho_interp(temp_arr) * density

    # First element calculations

    Qinc = ONEDHT_QINC(gas_temp, temp_arr[0], effemish, hchot)
    temp_arr[0] = ONEDHT_ELEM1(Qinc, temp_arr[0], temp_arr[1], lamdaNL[0], lamdaNL[1], dx, dt, cpNL[0], rhoNL[0])

    # Intermediate element calculations

    for nodenum in range(1, nodes - 1):
        temp_arr[nodenum] = ONEDHT_ELEMJ(temp_arr[nodenum - 1], temp_arr[nodenum], temp_arr[nodenum + 1],
                                         lamdaNL[nodenum - 1], lamdaNL[nodenum], lamdaNL[nodenum + 1], dx, dt,
                                         cpNL[nodenum], rhoNL[nodenum])

    # Final element calculations

    Qout = ONEDHT_QOUT(temp_arr[nodes - 1], tamb, effemisc, hccold)
    temp_arr[nodes - 1] = ONEDHT_ELEMF(Qout, temp_arr[nodes - 2], temp_arr[nodes - 1], lamdaNL[nodes - 2],
                                       lamdaNL[nodes - 1], dx, dt, cpNL[nodes - 1], rhoNL[nodes - 1])

    Temps=temp_arr
    return Temps

if __name__ == '__main__':

    nodes = 11                      # set the number of nodes
    dx = 0.0025                     # in m
    temp_arr = np.ones(nodes)*20    # creates initial temperature nodal array with ambient condition
    endtime = 1800                  # number of timesteps
    dt = 1                          # time step increment - very important for numerical stability
    nth = 300                       # Temperature profile will be produced every nth time step
    endtimestep = int(endtime/dt)   # calculate total number of timesteps
    node_array =np.ones(nodes)      # Create depth array for temperature profile
    track_iso = 300                 # Isotherm depth to be tracked

    wall_depth_arr = np.ones(nodes) # creates an array for interpolating an isotherm position & plotting

    # Generate depth array [mm]
    for x in range(0,nodes):
        wall_depth_arr[x]=x*dx*1000  # expressed in mm

    list_temp_arr=[]

    # Main heat transfer solver

    for tstep in range(1, endtimestep + 1):

        gas_temp = 500
        temp_arr = wall_HT_sub(gas_temp,temp_arr,nodes,dt,dx,35,4,0.7,0.7,20)   # HT function

        # Store results

        list_temp_arr.append(copy.copy(temp_arr))

        if tstep/nth == int(tstep/nth):
            if tstep > 1:
                plt.plot(wall_depth_arr, temp_arr, label=str(tstep*dt))

        print(temp_arr[3])

    plt.xlabel('Depth [mm]')
    plt.ylabel('Temperature [$^o$ C]')
    plt.grid(True)
    plt.legend()
    plt.show()