# -*- coding: utf-8 -*-
import cProfile


def profile_standard_case():
    import copy
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
    from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen
    # increase the number of simulations so it gives sensible results
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    mcs_config = copy.deepcopy(EXAMPLE_CONFIG_DICT)
    for k in list(mcs_input.keys()):
        mcs_input[k]['phi_teq'] = 1
        mcs_input[k]['n_simulations'] = 1000
        mcs_input[k]['timber_exposed_area'] = 0
        mcs_input[k].pop('beam_position_horizontal')
        mcs_input[k]['beam_position_horizontal:dist'] = 'uniform_'
        mcs_input[k]['beam_position_horizontal:ubound'] = mcs_input[k]['room_depth'] * 0.9
        mcs_input[k]['beam_position_horizontal:lbound'] = mcs_input[k]['room_depth'] * 0.6

    # increase the number of threads so it runs faster
    mcs_config['n_threads'] = 3
    mcs = MCS()
    mcs.define_problem(data=mcs_input, config=mcs_config)
    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)
    mcs.run_mcs()


if __name__ == '__main__':
    # Profile for the standard case input sfeprapy.mcs0.EXAMPLE_INPUT_CSV
    cProfile.run('profile_standard_case()', sort='cumtime')
