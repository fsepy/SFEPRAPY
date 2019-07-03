#  Danny Hopkin
#  OFR Consultants UK
#  Zone model sub-functions
#  16/05/18

import numpy as np

#  calculate max HRR

def hrrmax(h0,b0):

    # Returns output in kW

    a0 = b0*h0
    hrrmax = 1500 * a0 * np.sqrt(h0)
    return hrrmax

#  convective flow from opening

def convf_open(h0,b0, Tgas, Tamb, cp):

    # Outcome depends on units of cp

    a0 = h0*b0
    dT = Tgas - Tamb
    qc = 0.5*a0*np.sqrt(h0)*cp*dT
    return qc

# radiative flow from opening

def radf_open(h0, b0, Tgas, Tamb, h):

    #  Outcome in watts
    Tgas = Tgas+273
    Tamb = Tamb+273
    ef = 1- np.exp(-1.1*h)
    dT4 = (Tgas**4)-(Tamb**4)
    a0 = b0*h0
    boltz = 5.67e-8
    qr = a0*ef*boltz*dT4
    return qr

# Temperature rise of gas

def delta_Tgas(rhoamb,cp, b,d,h, dt, qgas):

    # Rise in K

    Vg = b*d*h
    dTgas = (qgas*dt)/(rhoamb*Vg*cp)
    return dTgas

# Energy stored in gas

def qgas_calc(HRR,qw,qc,qr):

    #  All parameters to be dimensionally consistent

    qgas = HRR-qw-qc-qr
    return qgas

# Heat transfer function to update wall temperature

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

    # Imports

    from scipy.interpolate import interp1d
    from HT_sub_functions import ONEDHT_QINC, ONEDHT_QOUT, ONEDHT_ELEM1, ONEDHT_ELEMJ, ONEDHT_ELEMF
    from complex_intersect import complex_intersect
    from matplotlib import pyplot as plt
    from crib_mass_loss_functions import fuel_contr_crib

    #   Define room parameters

    depth = 4.5       # [m]
    breadth = 3.5     # [m]
    height = 2.5    # [m]

    #   Define opening parameters

    op_height = 2   #  [m]
    op_width = 1    #  [m]

    #   max HRR

    maxhrr = hrrmax(op_height,op_width)*1000    # in [W]

    # CLT related parameters

    CLT_mass = 250    #  initialise crib mass
    mass_lost = 0
    crib_mass_remain = 0
    delta_mass = 0  #  initialise change in crib mass
    crib_MLR = 0    #  initialise crib MLR
    crib_mass_consumed = 0  # initialise mass consumed term
    GER = 2                 # Global equivalence ratio

    #   calculate the area of lining

    Aopen = op_height * op_width
    Afloor = depth * breadth
    Aceiling = Afloor
    Awalls = 2*(depth + breadth) * height
    Alining = Afloor + Aceiling + Awalls - Aopen

    #   Define ambient conditions

    Tambient = 18       #  [C]
    Twall = 20          #  [C]
    Tgas = Tambient     #  set initial gas temperature to ambient

    #   Design fire input

    hrr_temp_arr = [0,300, 600, 1200, 1800, 3600, 12000]                   # Time array [s]
    hrr_input_arr =[0,100, 5000, 4000, 1200, 500, 0]    # HRR array [kW]
    hrr_interp = interp1d(hrr_temp_arr,hrr_input_arr)                  # Interpolation function

    # Time parameters

    uptime = 10000   #  [s]
    dt = 1          #  [s]
    endtimestep = int(uptime / dt)
    nth = 250

    #  Set initial wall conditions & properties
    nodes = 101  # set the number of nodes
    dx = 0.002
    wall_depth = dx*(nodes-1)
    temp_arr = np.ones(nodes)*20    # creates initial temperature nodal array with ambient condition
    wall_depth_arr = np.ones(nodes) # creates an array for interpolating an isotherm position

    # Generate depth array [mm]
    for x in range(0,nodes):
        wall_depth_arr[x]=x*dx*1000  # expressed in mm

    # Wall properties and isotherm tracking
    hchot = 25
    hccold = 9
    emissh = 0.7
    emissc = 0.7
    iso = 300
    iso_pos = 0

    #  Initialise complex intersection for isotherm tracking

    f = np.ones(nodes)*iso  # Creates a straight line with same number of indices for isotherm tracking

    # Create storage arrays

    time_array = []
    crib_mlr_arr = []
    wall_surf_T_array =[]
    Tgas_array = []
    isotherm_arr = []
    isotherm_arr2 = []

    total_mass_lost = 0
    mlr_max_crib = 0

    for tstep in range(1, endtimestep + 1):

        #  calculate time and corresponding HRR from input
        time = tstep * dt

        hrr = hrr_interp(time)*1000     # Convert to watts

        if hrr > maxhrr:
            hrr = maxhrr

        # Gas properties
        gas_temp_arr = [0, 230, 1300]  # Temp array
        cp_input_arr = [1003, 1030, 1216]  # Specific heat array
        cp_interp = interp1d(gas_temp_arr, cp_input_arr)

        # calculate wall losses
        qwall = ONEDHT_QINC(Tgas,Twall,emissh,hchot)
        qwall_tot = qwall * Alining

        # calculate opening losses
        qcon = convf_open(op_height,op_width,Tgas,0,cp_interp(Tgas))
        qrad =radf_open(op_height,op_width,Tgas,0,height)

        # calculate energy stored in the gas
        qgas = qgas_calc(hrr,qwall_tot,qcon,qrad)

        #  calculate rise in gas temperature
        dTgas = delta_Tgas(1,cp_interp(Tgas),breadth,depth,height,dt,qgas)

        #  update gas temperature
        Tgas = min(Tgas + dTgas,1300)

        # HT to wall

        temp_arr = wall_HT_sub(Tgas,temp_arr,nodes,dt, dx,hchot,hccold,emissh,emissc,Tambient)
        Twall = temp_arr[0]

        print(qwall, Twall, Tgas)

        # Update ambient condition in wall
        for xx in range(0, nodes):
            if temp_arr[xx] < 20:
                temp_arr[xx] = 20

        # Track isotherm
        idx = complex_intersect(f, temp_arr)
        iso_old = iso_pos

        if max(temp_arr) < iso:
            iso_pos = 0
        elif min(temp_arr) > iso:
            iso_pos = wall_depth
        else:
            iso_pos = idx * dx * 1000
            iso_pos = max(iso_pos)

        if iso_old > iso_pos:
            iso_pos = iso_old
        isotherm_arr.append(float(iso_pos))

        iso_change = max(iso_pos - iso_old,0)

        if iso_change > 0:
            exposed_area = 4.5*2.5
            delta_mass = exposed_area * (iso_change/1000) * 450
            CLT_mass = CLT_mass + delta_mass

        #  Calculate crib mass loss for lining
        if CLT_mass > 0:
            crib_MLR = fuel_contr_crib(0.05,crib_mass_remain,CLT_mass)
            mass_lost = crib_MLR*dt
            total_mass_lost = total_mass_lost + mass_lost
            crib_mass_remain = max(CLT_mass - total_mass_lost,0)

        crib_mlr_arr.append(crib_MLR)

        time_array.append(time)
        Tgas_array.append(Tgas)
        wall_surf_T_array.append(temp_arr[0])

        if tstep/nth == int(tstep/nth):
            if tstep > 1:
                plt.subplot(221)
                plt.plot(wall_depth_arr, temp_arr, label=str(tstep*dt))

        print('Time = ', time, ' [s]', 'Isotherm depth =', iso_pos, ' [mm]', 'New CLT mass =', CLT_mass, ' [kg]')
        print('Mass lost', mass_lost, total_mass_lost, crib_mass_remain)

    plt.xlabel('Depth [mm]')
    plt.ylabel('Temperature [$^o$ C]')
    plt.grid(True)
    plt.legend()
    plt.subplot(222)
    plt.xlabel('Time [s]')
    plt.ylabel('Isotherm depth [mm]')
    plt.grid(True)
    plt.plot(time_array, isotherm_arr)
    plt.subplot(223)
    plt.xlabel('Time [s]')
    plt.ylabel('Gas temperature [$^o$ C]')
    plt.plot(time_array, Tgas_array)
    plt.grid(True)
    plt.subplot(224)
    plt.xlabel('Time [s]')
    plt.ylabel('CLT MLR [kg/s]')
    plt.plot(time_array, crib_mlr_arr)
    plt.grid(True)
    plt.show()