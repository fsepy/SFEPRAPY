# -*- coding: utf-8 -*-


if __name__ == '__main__':
    import time
    import warnings
    import numpy as np
    from tqdm import tqdm
    from sfeprapy.func.mcs_gen import main as mcs_gen
    from sfeprapy.mc0.mc0_func_main_2 import main as mcs_calc
    from sfeprapy.func.fire_iso834 import fire as isofire
    
    warnings.filterwarnings('ignore')

    fire_time = np.arange(0, 3*60*60+30, 30, dtype=float)
    fire_temperature_iso834 = isofire(fire_time, 293.15)

    dict_in = dict(
        probability_weight=0.2,
        n_simulations=10000,
        time_step=30,
        time_duration=18000,

        fire_hrr_density=dict(
            dist='uniform_',
            lbound=0.25,
            ubound=1.00),
        fire_load_density=dict(
            dist='gumbel_r_',
            lbound=50,
            ubound=2000,
            mean=420,
            sd=126),
        fire_spread_speed=dict(
            dist='uniform_',
            lbound=0.0035,
            ubound=0.019),
        fire_nft_limit=dict(
            dist='norm_',
            lbound=293.15,
            ubound=1673.15,
            mean=1323.15,
            sd=1.135),
        fire_combustion_efficiency=dict(
            dist='uniform_',
            lbound=0.8,
            ubound=1.0),
        room_opening_fraction=dict(
            dist='lognorm_mod_',
            ubound=0.0009,
            lbound=0.0001,
            mean=0.5,
            sd=1),

        beam_cross_section_area=0.017,
        beam_position_vertical=2.5,
        beam_position_horizontal=18,
        beam_rho=7850,
        fire_time=fire_time,
        fire_mode=3,
        fire_gamma_fi_q=1,
        fire_t_alpha=300,
        fire_tlim=0.333,
        fire_temperature_iso834=fire_temperature_iso834,
        fire_time_iso834=fire_time,
        protection_c=1700,
        protection_k=0.2,
        protection_protected_perimeter=2.14,
        protection_rho=7850,
        room_breadth=16,
        room_depth=31.25,
        room_height=3,
        room_wall_thermal_inertia=720,
        solver_temperature_goal=550 + 273.15,
        solver_max_iter=20,
        solver_thickness_lbound=0.0001,
        solver_thickness_ubound=0.0500,
        solver_tol=1.,
        window_height=2.5,
        window_open_fraction=0.8,
        window_width=72

    )

    df_out = mcs_gen(dict_in, 1000)

    list_out = df_out.to_dict(orient='records')

    for i in tqdm(list_out):
            mcs_calc(**i)
