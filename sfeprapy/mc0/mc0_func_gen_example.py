# -*- coding: utf-8 -*-

from io import StringIO
import pandas as pd


def benchmark_0():
    csv_ = """PARAMETERS,Example Case
    is_live,1
    fire_mode,3
    probability_weight,0.2
    n_simulations,10000
    time_step,30
    time_duration,18000
    fire_hrr_density_lbound,0.249
    fire_hrr_density_ubound,0.251
    fire_hrr_density_mean,0.25
    fire_hrr_density_std,100
    fire_qfd_std,126
    fire_qfd_mean,420
    fire_qfd_ubound,1500
    fire_qfd_lbound,10
    fire_spread_lbound,0.0035
    fire_spread_ubound,0.019
    fire_nft_mean,1050
    fire_com_eff_lbound,0.75
    fire_com_eff_ubound,0.999
    fire_tlim,0.333
    fire_t_alpha,300
    fire_gamma_fi_q,1
    room_breadth,16
    room_depth,31.25
    room_height,3
    room_window_width,72
    room_window_height,2.8
    room_opening_fraction_std,0.2
    room_opening_fraction_mean,0.2
    room_opening_fraction_ubound,0.999
    room_opening_fraction_lbound,0.001
    room_opening_permanent_fraction,0
    room_wall_thermal_inertia,720
    beam_cross_section_area,0.017
    beam_rho,7850
    beam_temperature_goal,893
    beam_protection_protected_perimeter,2.14
    beam_protection_thickness,0
    beam_protection_k,0.2
    beam_protection_rho,800
    beam_protection_c,1700
    beam_loc_z,3
    beam_loc_ratio_lbound,0.666
    beam_loc_ratio_ubound,0.999"""

    df = pd.read_csv(StringIO(csv_), sep=",")

    print(df)


def mcs_gen_test_muted():
    from sfeprapy.func.mcs_gen import main
    import numpy as np

    dict_in = dict(
        fire_mode=np.pi,
        probability_weight=0.2,
        n_simulations=10000,
        time_step=30,
        time_duration=18000,
        fire_hrr_density=dict(
            dist='uniform_',
            lbound=0.25,
            ubound=1.00),
        fire_qfd=dict(
            dist='gumbel_r_',
            lbound=50,
            ubound=2000,
            mean=420,
            sd=126),
        fire_spread=dict(
            dist='uniform_',
            lbound=0.0035,
            uboudn=0.019),
        fire_nft_limit=dict(
            dist='norm_',
            lbound=293.15,
            ubound=1673.15,
            mean=1323.15,
            sd=0),
        fire_combustion_efficiency=dict(
            dist='uniform_',
        ),
        fire_tlim=0.333,
        fire_t_alpha=300,
        fire_gamma_fi_q=1,
        room_breadth=16,
        room_depth=31.25,
        room_height=3,
        room_window_width=72,
        room_window_height=2.8,
        room_opening_fraction=dict(
            dict='lognorm_mod_',
        ),
        room_opening_permanent_fraction=0,
        room_wall_thermal_interia=720,
        beam_cross_section_area=702,
        #         v2='hello world.',
        #         v3=dict(
        #             dist='uniform_',
        #             ubound=10,
        #             lbound=-1
        #         ),
        #         v4=dict(
        #             dist='norm_',
        #             ubound=5+1,
        #             lbound=5-1,
        #             mean=5,
        #             sd=1
        #         ),
        #         v5=dict(
        #             dist='gumbel_r_',
        #             ubound=2500,
        #             lbound=50,
        #             mean=420,
        #             sd=126
        #         ),
        #         v6=dict(
        #             dist='lognorm_',
        #             ubound=1,
        #             lbound=0,
        #             mean=0.5,
        #             sd=1,
        #         ),
        #         v7=dict(
        #             dist='lognorm_mod_',
        #             ubound=1,
        #             lbound=0,
        #             mean=0.5,
        #             sd=1,
        #         )
    )

    df_out = main(dict_in, 10000)

    print(df_out.to_string(max_rows=10, max_cols=10, index=False))

mcs_gen_test_muted()
