

def test():
    import copy

    def test_standard_case():
        from sfeprapy.func.mcs_obj import MCS
        from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
        from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
        from sfeprapy.func.mcs_gen import main as gen
        from scipy.interpolate import interp1d
        import numpy as np
        # increase the number of simulations so it gives sensible results
        mcs_input = EXAMPLE_INPUT_DICT
        for k in list(mcs_input.keys()):
            mcs_input[k]['n_simulations'] = 2500
        # increase the number of threads so it runs faster
        mcs_config = copy.copy(EXAMPLE_CONFIG_DICT)
        mcs_config['n_threads'] = 4
        mcs = MCS()
        mcs.define_problem(data=mcs_input, config=mcs_config)
        mcs.define_stochastic_parameter_generator(gen)
        mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)
        mcs.run_mcs()
        mcs_out = mcs.mcs_out
        teq = mcs_out['solver_time_equivalence_solved'] / 60.
        hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.5))
        x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
        print(abs(interp1d(y, x)(0.8) - 60))
        assert abs(interp1d(y, x)(0.8) - 60) < 1

    # test gui version
    def test_gui():
        from sfeprapy.func.mcs_obj import MCS
        from sfeprapy.mcs0.mcs0_calc import teq_main as calc
        from sfeprapy.mcs0.mcs0_calc import teq_main_wrapper as calc_mp
        from sfeprapy.func.mcs_gen import main as gen
        mcs = MCS()
        mcs.define_problem()
        mcs.define_stochastic_parameter_generator(gen)
        mcs.define_calculation_routine(calc, calc_mp)
        mcs.run_mcs()

    # test non-gui version
    def test_arg_dict():
        from sfeprapy.func.mcs_obj import MCS
        from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
        from sfeprapy.mcs0.mcs0_calc import teq_main as calc
        from sfeprapy.mcs0.mcs0_calc import teq_main_wrapper as calc_mp
        from sfeprapy.func.mcs_gen import main as gen
        for k in list(EXAMPLE_INPUT_DICT.keys()):
            EXAMPLE_INPUT_DICT[k]['n_simulations'] = 2
        mcs = MCS()
        mcs.define_problem(data=EXAMPLE_INPUT_DICT, config=EXAMPLE_CONFIG_DICT)
        mcs.define_stochastic_parameter_generator(gen)
        mcs.define_calculation_routine(calc, calc_mp)
        mcs.run_mcs()

    import time
    test_standard_case()
    time.sleep(0.5)
    test_arg_dict()
    time.sleep(0.5)
    test_gui()
    time.sleep(0.5)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    test()
