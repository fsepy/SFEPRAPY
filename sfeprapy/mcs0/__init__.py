EXAMPLE_CONFIG_DICT = dict(
    n_threads=2,
)

EXAMPLE_INPUT_DICT = {
    "Standard Case 1": dict(
        case_name='Standard Case 1',
        n_simulations=100,
        probability_weight=0.5,
        fire_time_step=30,
        fire_time_duration=18000,

        fire_hrr_density=dict(
            dist='uniform_',
            lbound=0.25 - 0.001,
            ubound=0.25 + 0.001),
        fire_load_density=dict(
            dist='gumbel_r_',
            lbound=10,
            ubound=1500,
            mean=420,
            sd=126),
        fire_spread_speed=dict(
            dist='uniform_',
            lbound=0.0035,
            ubound=0.0190),
        fire_nft_limit=dict(
            dist='norm_',
            lbound=623.15,
            ubound=2023.15,
            mean=1323.15,
            sd=93),
        fire_combustion_efficiency=dict(
            dist='uniform_',
            lbound=0.8,
            ubound=1.0),
        window_open_fraction=dict(
            dist='lognorm_mod_',
            ubound=0.9999,
            lbound=0.0001,
            mean=0.2,
            sd=0.2),

        beam_cross_section_area=0.017,
        beam_position_horizontal=-1,
        beam_position_vertical=3.2,
        beam_rho=7850,
        fire_mode=3,
        fire_gamma_fi_q=1,
        fire_t_alpha=300,
        fire_tlim=0.333,
        protection_c=1700,
        protection_k=0.2,
        protection_protected_perimeter=2.14,
        protection_rho=800,
        room_breadth=16,
        room_depth=31.25,
        room_height=3.3,
        room_wall_thermal_inertia=720,
        solver_temperature_goal=893.15,
        solver_max_iter=20,
        solver_thickness_lbound=0.0001,
        solver_thickness_ubound=0.0500,
        solver_tol=1.,
        window_height=2.8,
        window_width=72,
        window_open_fraction_permanent=0
    ),
    "Standard Case 2": dict(
        case_name='Standard Case 2',
        n_simulations=100,
        probability_weight=0.5,
        fire_time_step=30,
        fire_time_duration=18000,

        fire_hrr_density=dict(
            dist='uniform_',
            lbound=0.25 - 0.001,
            ubound=0.25 + 0.001),
        fire_load_density=dict(
            dist='gumbel_r_',
            lbound=10,
            ubound=1500,
            mean=420,
            sd=126),
        fire_spread_speed=dict(
            dist='uniform_',
            lbound=0.0035,
            ubound=0.0190),
        fire_nft_limit=dict(
            dist='norm_',
            lbound=623.15,
            ubound=2023.15,
            mean=1323.15,
            sd=93),
        fire_combustion_efficiency=dict(
            dist='uniform_',
            lbound=0.8,
            ubound=1.0),
        window_open_fraction=dict(
            dist='lognorm_mod_',
            ubound=0.9999,
            lbound=0.0001,
            mean=0.2,
            sd=0.2),

        beam_cross_section_area=0.017,
        beam_position_horizontal=-1,
        beam_position_vertical=3.2,
        beam_rho=7850,
        fire_mode=3.0,
        fire_gamma_fi_q=1,
        fire_t_alpha=300,
        fire_tlim=0.333,
        protection_c=1700,
        protection_k=0.2,
        protection_protected_perimeter=2.14,
        protection_rho=800,
        room_breadth=16,
        room_depth=31.25,
        room_height=3.3,
        room_wall_thermal_inertia=720,
        solver_temperature_goal=620 + 273.15,
        solver_max_iter=20,
        solver_thickness_lbound=0.0001,
        solver_thickness_ubound=0.0500,
        solver_tol=1.,
        window_height=2.8,
        window_width=72,
        window_open_fraction_permanent=0
    )
}

EXAMPLE_INPUT_CSV = """PARAMETERS, Standard Case 1, Standard Case 2
case_name,Standard Case 1,Standard Case 2
n_simulations,1000,1000
probability_weight,0.5,0.5
fire_time_step,30,30
fire_time_duration,18000,18000
fire_hrr_density:dist,uniform_,uniform_
fire_hrr_density:lbound,0.26,0.26
fire_hrr_density:ubound,0.24,0.24
fire_load_density:dist,gumbel_r_,gumbel_r_
fire_load_density:lbound,10,10
fire_load_density:ubound,1500,1500
fire_load_density:mean,420,420
fire_load_density:sd,126,126
fire_spread_speed:dist,uniform_,uniform_
fire_spread_speed:lbound,0.0035,0.0035
fire_spread_speed:ubound,0.019,0.019
fire_nft_limit:dist,norm_,norm_
fire_nft_limit:lbound,623.15,623.15
fire_nft_limit:ubound,2023.15,2023.15
fire_nft_limit:mean,1323.15,1323.15
fire_nft_limit:sd,93,93
fire_combustion_efficiency:dist,uniform_,uniform_
fire_combustion_efficiency:lbound,0.8,0.8
fire_combustion_efficiency:ubound,1,1
window_open_fraction:dist,lognorm_mod_,lognorm_mod_
window_open_fraction:ubound,0.9999,0.9999
window_open_fraction:lbound,0.0001,0.0001
window_open_fraction:mean,0.2,0.2
window_open_fraction:sd,0.2,0.2
beam_position_horizontal,-1,-1
beam_cross_section_area,0.017,0.017
beam_position_vertical,3.2,3.2
beam_rho,7850,7850
fire_mode,3,3
fire_gamma_fi_q,1,1
fire_t_alpha,300,300
fire_tlim,0.333,0.333
protection_c,1700,1700
protection_k,0.2,0.2
protection_protected_perimeter,2.14,2.14
protection_rho,800,800
room_breadth,16,16
room_depth,31.25,31.25
room_height,3.3,3.3
room_wall_thermal_inertia,720,720
solver_temperature_goal,893.15,893.15
solver_max_iter,20,20
solver_thickness_lbound,0.0001,0.0001
solver_thickness_ubound,0.05,0.05
solver_tol,1,1
window_height,2.8,2.8
window_width,72,72
window_open_fraction_permanent,0,0.2
"""
