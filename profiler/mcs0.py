# -*- coding: utf-8 -*-
import cProfile


def profile_standard_case():
    from sfeprapy.mcs0 import MCS0, EXAMPLE_INPUT

    # increase the number of simulations so it gives sensible results
    mcs_input = dict()
    mcs_input['CASE_1'] = EXAMPLE_INPUT['CASE_1'].copy()
    for k in list(mcs_input.keys()):
        mcs_input[k]["phi_teq"] = 1
        mcs_input[k]["n_simulations"] = 10_000
        mcs_input[k]["timber_exposed_area"] = 0
        mcs_input[k].pop("beam_position_horizontal")
        mcs_input[k]["beam_position_horizontal:dist"] = "uniform_"
        mcs_input[k]["beam_position_horizontal:ubound"] = mcs_input[k]["room_depth"] * 0.9
        mcs_input[k]["beam_position_horizontal:lbound"] = mcs_input[k]["room_depth"] * 0.6

    # increase the number of threads so it runs faster
    mcs = MCS0()
    mcs.set_inputs_dict(mcs_input)
    mcs.run(4, save=False)


if __name__ == "__main__":
    # Profile for the standard case input sfeprapy.mcs0.EXAMPLE_INPUT_CSV
    cProfile.run("profile_standard_case()", sort="cumtime")
