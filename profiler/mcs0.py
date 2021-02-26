# -*- coding: utf-8 -*-
import cProfile


def profile_standard_case():
    import copy
    from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
    from sfeprapy.mcs0.mcs0_calc import MCS0

    # increase the number of simulations so it gives sensible results
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    mcs_config = copy.deepcopy(EXAMPLE_CONFIG_DICT)
    for k in list(mcs_input.keys()):
        mcs_input[k]["phi_teq"] = 1
        mcs_input[k]["n_simulations"] = 1000
        mcs_input[k]["timber_exposed_area"] = 0
        mcs_input[k].pop("beam_position_horizontal")
        mcs_input[k]["beam_position_horizontal:dist"] = "uniform_"
        mcs_input[k]["beam_position_horizontal:ubound"] = mcs_input[k]["room_depth"] * 0.9
        mcs_input[k]["beam_position_horizontal:lbound"] = mcs_input[k]["room_depth"] * 0.6

    # increase the number of threads so it runs faster
    mcs_config["n_threads"] = 3
    mcs = MCS0()
    mcs.inputs = mcs_input
    mcs.mcs_config = mcs_config
    mcs.run_mcs()


if __name__ == "__main__":
    # Profile for the standard case input sfeprapy.mcs0.EXAMPLE_INPUT_CSV
    cProfile.run("profile_standard_case()", sort="cumtime")
