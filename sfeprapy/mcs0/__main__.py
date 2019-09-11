#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import os
    import sys
    import warnings
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen

    warnings.filterwarnings('ignore')

    mcs_problem_definition = None
    n_threads = None
    if len(sys.argv) > 1:
        mcs_problem_definition = os.path.realpath(sys.argv[1])
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if 'mp' in arg:
                n_threads = int(str(arg).replace('mp', ''))

    mcs = MCS()
    mcs.define_problem(mcs_problem_definition)
    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)

    if n_threads:
        mcs.config = dict(n_threads=n_threads) if mcs.config else mcs.config.update(dict(n_threads=n_threads))

    mcs.run_mcs()
