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

    mcs = MCS()

    if len(sys.argv) > 1:
        mcs.define_problem(os.path.realpath(sys.argv[1]))
        for arg in sys.argv[2:]:
            print(arg)
            if 'sim' in arg:
                print(arg)
                n_simulations = int(str(arg).replace('sim', ''))
                if mcs.config is None:
                    print(mcs.config)

                    mcs.config = dict(n_simulations=n_simulations)
                else:
                    mcs.config['n_simulations'] = int(str(arg).replace('sim', ''))
                break
    else:
        mcs.define_problem()

    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)
    mcs.run_mcs()
