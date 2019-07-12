# -*- coding: utf-8 -*-
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from sfeprapy.func.mcs_gen import main as mcs_gen
from sfeprapy.mc0.mc0_func_main_2 import main as mcs_calc
from sfeprapy.mc0.mc0_func_main_2 import main_mp_wrapper

warnings.filterwarnings('ignore')

def main(dict_mc0_in):
    n_threads = 1

    df_mcs_in = mcs_gen(dict_mc0_in, dict_mc0_in['n_simulations'])
    list_mcs_in = df_mcs_in.to_dict(orient='records')
    list_df_mcs_out = list()

    if n_threads == 1:
        for i in tqdm(list_mcs_in):
            mcs_calc(**i)
    else:
        import multiprocessing as mp
        m, p = mp.Manager(), mp.Pool(n_threads, maxtasksperchild=1000)
        q = m.Queue()
        jobs = p.map_async(main_mp_wrapper, [(dict_, q) for dict_ in list_mcs_in])
        n_simulations = len(list_mcs_in)
        with tqdm(total=n_simulations, ncols=60) as pbar:
            while True:
                if jobs.ready():
                    if n_simulations > pbar.n:
                        pbar.update(n_simulations - pbar.n)
                    break
                else:
                    if q.qsize() - pbar.n > 0:
                        pbar.update(q.qsize() - pbar.n)
                    time.sleep(1)
            p.close()
            p.join()
            mcs_out = jobs.get()
    df_mcs_out = pd.DataFrame(mcs_out)
    df_mcs_out.drop('fire_temperature')
    df_mcs_out.sort_values('solver_time_equivalence_solved', inplace=True)  # sort base on time equivalence

    return df_mcs_out


def main_test():
    from sfeprapy.func.fire_iso834 import fire as isofire

    fire_time = np.arange(0, 3 * 60 * 60 + 30, 30, dtype=float)
    fire_temperature_iso834 = isofire(fire_time, 293.15)

    dict_mc0_in = dict(
        n_simulations=1000,
        probability_weight=0.2,
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

    n_threads = 1

    df_mcs_in = mcs_gen(dict_mc0_in, dict_mc0_in['n_simulations'])
    list_mcs_in = df_mcs_in.to_dict(orient='records')
    list_df_mcs_out = list()

    if n_threads == 1:
        for i in tqdm(list_mcs_in):
            mcs_calc(**i)
    else:
        import multiprocessing as mp
        m, p = mp.Manager(), mp.Pool(n_threads, maxtasksperchild=1000)
        q = m.Queue()
        jobs = p.map_async(main_mp_wrapper, [(dict_, q) for dict_ in list_mcs_in])
        n_simulations = len(list_mcs_in)
        with tqdm(total=n_simulations, ncols=60) as pbar:
            while True:
                if jobs.ready():
                    if n_simulations > pbar.n:
                        pbar.update(n_simulations - pbar.n)
                    break
                else:
                    if q.qsize() - pbar.n > 0:
                        pbar.update(q.qsize() - pbar.n)
                    time.sleep(1)
            p.close()
            p.join()
        mcs_out = jobs.get()
        df_mcs_out = pd.DataFrame(mcs_out)
        df_mcs_out.drop('fire_temperature')
        df_mcs_out.sort_values('solver_time_equivalence_solved', inplace=True)  # sort base on time equivalence
        list_df_mcs_out.append(df_mcs_out)
