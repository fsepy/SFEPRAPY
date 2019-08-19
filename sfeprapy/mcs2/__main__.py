#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import warnings
    import numpy as np
    from scipy.interpolate import interp1d
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs2.mcs2_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen

    warnings.filterwarnings('ignore')

    mcs = MCS()
    mcs.define_problem()
    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)
    mcs_out = mcs.run_mcs()

    bin_width = 0.1
    dict_teq = dict()
    v = np.asarray(mcs_out['solver_time_equivalence_solved'].values, dtype=float) / 60.

    xlim = (0, np.max(v) + bin_width)
    edges = np.arange(*xlim, bin_width)

    x = (edges[1:] + edges[:-1]) / 2
    y_pdf = {k: np.histogram(v, edges)[0] / len(v) for k, v in dict_teq.items()}
    y_cdf = {k: np.cumsum(v) for k, v in y_pdf.items()}

    func_ = interp1d(y_cdf, x)[]
