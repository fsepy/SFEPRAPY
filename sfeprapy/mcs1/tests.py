import copy

import numpy as np

from sfeprapy.mcs1 import EXAMPLE_INPUT_DICT, MCS1
from sfeprapy.mcs1.calcs import *


def test_teq_phi():
    inputs = dict(
        index=0,
        fire_time_step=1.,
        fire_time_duration=5. * 60 * 60,
        beam_cross_section_area=0.017,
        beam_position_vertical=2.5,
        beam_position_horizontal=18,
        beam_rho=7850.,
        fire_combustion_efficiency=0.8,
        fire_gamma_fi_q=1,
        fire_hrr_density=0.25,
        fire_load_density=420,
        fire_mode=0,
        fire_nft_limit=1050,
        fire_spread_speed=0.01,
        fire_t_alpha=300,
        fire_tlim=0.333,
        protection_c=1700.,
        protection_k=0.2,
        protection_protected_perimeter=2.14,
        protection_rho=800.,
        protection_d_p=0.01,
        room_breadth=16,
        room_depth=31.25,
        room_height=3,
        room_wall_thermal_inertia=720,
        # solver_temperature_goal=620 + 273.15,
        # solver_max_iter=200,
        # solver_thickness_lbound=0.0001,
        # solver_thickness_ubound=0.0500,
        # solver_tol=0.01,
        window_height=2,
        window_open_fraction=0.8,
        window_width=72,
        window_open_fraction_permanent=0,
        timber_charring_rate=0.7,
        timber_exposed_area=0,
        timber_hc=400,
        timber_density=500,
        timber_solver_ilim=20,
        timber_solver_tol=1,
        epsilon_q=0.5,
        t_k_y_theta=3600,
    )

    inputs["phi_teq"] = 1.0
    teq_10 = teq_main(**inputs)[14]

    inputs["phi_teq"] = 0.1
    teq_01 = teq_main(**inputs)[14]

    print(
        f'Time equivalence at phi_teq=0.1: {teq_01:<8.3f}\n'
        f'Time equivalence at phi_teq=1.0: {teq_10:<8.3f}\n'
        f'Ratio between the above:         {teq_10 / teq_01:<8.3f}\n'
    )

    assert abs(teq_10 / teq_01 - 10) < 0.01


def test_standard_case(skip_3: bool = False):
    # increase the number of simulations so it gives sensible results
    mcs_input = EXAMPLE_INPUT_DICT.copy()
    mcs_input['CASE_1']['n_simulations'] = 10_000

    mcs = MCS1(1)
    mcs.set_inputs_dict({
        'CASE_1': mcs_input.pop('CASE_1'),
    })
    mcs.run()

    # 60 minutes based on Kirby et al
    x, y = mcs['CASE_1'].get_cdf()
    assert abs(np.amax(y[x < 60]) - 0.2) > 0.
    assert abs(np.amax(y[x < 60]) - 0.4) > 0.
    assert abs(np.amax(y[x < 60]) - 0.6) > 0.
    assert abs(np.amax(y[x < 60]) - 0.8) > 0.


def test_file_input():
    import tempfile
    import time
    import os

    from sfeprapy.mcs1 import EXAMPLE_INPUT_DF

    # save input as .xlsx
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as dir_work:
        print('A temporary folder has been created:', dir_work)

        time.sleep(0.5)
        fp_in = os.path.join(dir_work, 'input.xlsx')
        EXAMPLE_INPUT_DF.to_excel(fp_in)
        print(f"A temporary input file has been created: {fp_in}")  # 4
        time.sleep(0.5)

        mcs = MCS1(2)
        mcs.set_inputs_file_path(fp_in)
        mcs.run()
        time.sleep(0.5)
        mcs.save_all(False)
        time.sleep(0.5)
        mcs.save_all(True)
        time.sleep(0.5)


if __name__ == '__main__':
    test_teq_phi()
    test_standard_case()
    test_file_input()
