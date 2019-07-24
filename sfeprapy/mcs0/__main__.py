#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import warnings
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen

    warnings.filterwarnings('ignore')

    mcs = MCS()
    mcs.define_problem()
    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)
    mcs.run_mcs()
