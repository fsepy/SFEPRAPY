# -*- coding: utf-8 -*-


def test_standard_case():
    import copy
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
    from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen
    from scipy.interpolate import interp1d
    import numpy as np
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
    mcs_out = mcs.mcs_out
    teq = mcs_out['solver_time_equivalence_solved'] / 60.
    hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.5))
    x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    teq_at_80_percentile = interp1d(y, x)(0.8)
    print(teq_at_80_percentile)
    target, target_tol = 60, 1
    assert target - target_tol < teq_at_80_percentile < target + target_tol


# test non-gui version
def test_arg():
    import copy
    import numpy as np
    from scipy.interpolate import interp1d
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
    from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    mcs_config = copy.deepcopy(EXAMPLE_CONFIG_DICT)
    for k in list(mcs_input.keys()):
        mcs_input[k].pop('phi_teq')
        mcs_input[k]['phi_teq:dist'] = 'lognorm_'
        mcs_input[k]['phi_teq:ubound'] = 3
        mcs_input[k]['phi_teq:lbound'] = 0.00001
        mcs_input[k]['phi_teq:mean'] = 1
        mcs_input[k]['phi_teq:sd'] = 0.25
        mcs_input[k]['n_simulations'] = 1000
        mcs_input[k]['timber_exposed_area'] = 0

    mcs_config['n_threads'] = 3
    mcs = MCS()
    mcs.define_problem(data=mcs_input, config=mcs_config)
    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)
    mcs.run_mcs()
    mcs_out = mcs.mcs_out
    teq = mcs_out['solver_time_equivalence_solved'] / 60.
    hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.5))
    x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    teq_at_80_percentile = interp1d(y, x)(0.8)
    print(teq_at_80_percentile)


if __name__ == '__main__':
    test_standard_case()
    test_arg()
