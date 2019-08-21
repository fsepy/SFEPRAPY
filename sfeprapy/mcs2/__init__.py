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
        window_open_fraction_permanent=0,

        timber_charring_rate=0.7,
        timber_hc=13200,
        timber_density=400,
        timber_exposed_area=10,
    )
}
