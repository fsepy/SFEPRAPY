# -*- coding: utf-8 -*-
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from sfeprapy.func.mcs_gen import main as mcs_gen
from sfeprapy.mc0.mc0_func_main_2 import teq_main, teq_main_wrapper, summerise_mcs_results

warnings.filterwarnings('ignore')


def main_args(dict_mc0_in, n_threads: int = 1) -> pd.DataFrame:

    df_mcs_in = mcs_gen(dict_mc0_in, dict_mc0_in['n_simulations'])
    list_mcs_in = df_mcs_in.to_dict(orient='records')

    if n_threads == 1:
        mcs_out = list()
        for i in tqdm(list_mcs_in, ncols=60):
            mcs_out.append(teq_main(**i))
    else:
        import multiprocessing as mp
        m, p = mp.Manager(), mp.Pool(n_threads, maxtasksperchild=1000)
        q = m.Queue()
        jobs = p.map_async(teq_main_wrapper, [(dict_, q) for dict_ in list_mcs_in])
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
    try:
        df_mcs_out.drop('fire_temperature')
    except KeyError:
        pass
    df_mcs_out.sort_values('solver_time_equivalence_solved', inplace=True)  # sort base on time equivalence

    print(summerise_mcs_results(df_mcs_out))

    return df_mcs_out


def test():
    import time

    def test_standard_case():
        import copy
        from sfeprapy.mc0 import EXAMPLE_INPUT_DICT
        from scipy.interpolate import interp1d
        input_ = copy.copy(EXAMPLE_INPUT_DICT)
        teq = main_args(input_, 4)['solver_time_equivalence_solved'] / 60.
        hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.5))
        x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
        assert abs(interp1d(y, x)(0.8) - 60) < 2

    def test_single_thread():
        import copy
        from sfeprapy.mc0 import EXAMPLE_INPUT_DICT
        input_ = copy.copy(EXAMPLE_INPUT_DICT)
        input_['n_simulations'] = 100
        input_['n_threads'] = 1
        main_args(input_, 1)

    def test_multiple_threads():
        import copy
        from sfeprapy.mc0 import EXAMPLE_INPUT_DICT
        input_ = copy.copy(EXAMPLE_INPUT_DICT)
        input_['n_simulations'] = 100
        input_['n_threads'] = 1
        main_args(input_, 2)

    print("Testing standard benchmark case...")
    test_standard_case()
    time.sleep(0.5)
    print("Successful.")

    print("Testing standard benchmark case...")
    test_single_thread()
    time.sleep(0.5)
    print("Successful.")

    print("Testing standard benchmark case...")
    test_multiple_threads()
    time.sleep(0.5)
    print("Successful.")


if __name__ == '__main__':
    test()
