"""DESCRIPTION
This file contains template input files for apps.
"""

app_time_equiv = """
# MC INPUT FILE


# ------------------------------------------------------------------------------------------------------------------
#   Settings
# ------------------------------------------------------------------------------------------------------------------

simulations = 500						# Number of simulations
building_height = (64.8/(1-0.80))**0.5	# Height of building [m]
n_proc = 4								# Number of threads, 0 to use all processors
select_fires_teq = 0.8					# Select fires based on time equivalence fractile
select_fires_teq_tol = 0.01				# Tolerance of 'select_fires_teq'

# ------------------------------------------------------------------------------------------------------------------
#   Compartment dimensions all in [m]
# ------------------------------------------------------------------------------------------------------------------

room_breadth = 15.8						# Room breadth [m]
room_depth = 31.6						# Room depth [m]
room_height = 3.0						# Room height [m]
window_width = 72						# Window width [m]
window_height = 2.8						# Window height [m]

# ------------------------------------------------------------------------------------------------------------------
#   Deterministic fire inputs
# ------------------------------------------------------------------------------------------------------------------

time_start = 0                          # Start time of simulation [s]
time_limiting = 0.333                   # Limiting time for fuel controlled case in EN 1991-1-2 parametric fire [hr]
time_step = 30                          # Time step used for fire and heat transfer [s]
fire_duration = 18000                   # Maximum time in time array [s]
fire_hrr_density = 0.25                 # HRR density [MW/sq.m]
room_wall_thermal_inertia = 720         # Compartment thermal inertia [J/m2s1/2K]

# ------------------------------------------------------------------------------------------------------------------
#   Section properties for heat transfer evaluation
# ------------------------------------------------------------------------------------------------------------------

beam_cross_section_area = 0.017         # Cross section area [sq.m]
beam_rho = 7850							# Density [kg/m3]
beam_temperature_goal = 620+273		    # Beam (steel) failure temperature for goal seek [K]. -1 to opt-out seeking.
protection_protected_perimeter = 2.14   # Heated perimeter [m]
protection_thickness = 0.0125           # Thickness of protection [m]. -1 to opt-out maximum steel temperature calc.
protection_k = 0.2                      # Protection conductivity [W/m.K]
protection_rho = 800                    # Density of protection [kg/cb.m]
protection_c = 1700                     # Specific heat of protection [J/kg.K]

# ------------------------------------------------------------------------------------------------------------------
#  Distributed variables
# ------------------------------------------------------------------------------------------------------------------

qfd_std = 126     		                # Fire load density - Gumbel distribution - standard dev [MJ/sq.m]
qfd_mean = 420       	                # Fire load density - Gumbel distribution - mean [MJ/sq.m]
qfd_ubound = 9999  		                # Fire load density - Gumbel distribution - upper limit [MJ/sq.m]
qfd_lbound = 0  			            # Fire load density - Gumbel distribution - lower limit [MJ/sq.m]
glaz_min = 0.1  	                    # Min glazing fall-out fraction - Linear dist
glaz_max = 0.999	                    # Max glazing fall-out fraction - Linear dist
beam_min = 0.6            	            # Min beam location relative to compartment length for TFM [-] - Linear dist
beam_max = 0.9          	            # Max beam location relative to compartment length for TFM [-] - Linear dist
com_eff_min = 0.75   	                # Min combustion efficiency - Linear dist
com_eff_max = 0.999  	                # Max combustion efficiency - Linear dist
spread_min = 0.0035           	        # Min spread rate for TFM [m/s] - Linear dist
spread_max = 0.0193           	        # Max spread rate for TFM [m/s] - Linear dist
avg_nft = 1050  	                    # TFM near field temperature - Norm distribution - mean [C]"""

app_deterministic = """
{
    'time_step': 1,
    'time_start': 0,
    'time_limiting': 0.333,
    'window_height': 2.8,
    'window_width': 72,
    'window_open_fraction': 0.8,
    'room_breadth': 15.8,
    'room_depth': 31.6,
    'room_height': 2.8,
    'room_wall_thermal_inertia': 720,
    'fire_load_density': 420,
    'fire_hrr_density': 0.25,
    'fire_spread_speed': 0.0114,
    'fire_duration': 18000,
    'beam_position': 0.1,
    'beam_rho': 7850,
    'beam_cross_section_area': 0.017,
    'beam_temperature_goal': -1,
    'protection_k': 0.2,
    'protection_rho': 800,
    'protection_c': 1700,
    'protection_thickness': 0.0125,
    'protection_protected_perimeter': 2.14
}"""
