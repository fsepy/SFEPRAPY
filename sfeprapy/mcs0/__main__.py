#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

from sfeprapy.func.mcs_obj import MCS

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    from sfeprapy.mcs0.mcs0_calc import teq_main as calc
    from sfeprapy.mcs0.mcs0_calc import teq_main_wrapper as calc_mp
    from sfeprapy.func.mcs_gen import main as gen

    mcs = MCS()
    mcs.define_problem()
    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(calc, calc_mp)
    mcs.run_mcs()
