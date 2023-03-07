import copy
import tempfile

import numpy as np
from scipy.interpolate import interp1d

from sfeprapy.mcs1 import EXAMPLE_INPUT_DF
from sfeprapy.mcs1 import EXAMPLE_INPUT_DICT
from sfeprapy.mcs1.mcs1_calc import *


def test_teq_phi():
    fire_time_ = np.arange(0, 2 * 60 * 60, 1)
    fire_temperature_iso834_ = 345.0 * np.log10(fire_time_ / 60. * 8.0 + 1.0) + 293.15

    inputs = dict(
        index=0,
        case_name="Standard 1",
        probability_weight=1.,
        fire_time_step=1.,
        fire_time_duration=5. * 60 * 60,
        n_simulations=1,
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
        fire_temperature_iso834=fire_temperature_iso834_,
        fire_time_iso834=fire_time_,
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
    teq_10 = teq_main(**inputs)["solver_time_equivalence_solved"]

    inputs["phi_teq"] = 0.1
    teq_01 = teq_main(**inputs)["solver_time_equivalence_solved"]

    print(
        f'Time equivalence at phi_teq=0.1: {teq_01:<8.3f}\n'
        f'Time equivalence at phi_teq=1.0: {teq_10:<8.3f}\n'
        f'Ratio between the above:         {teq_10 / teq_01:<8.3f}\n'
    )

    assert abs(teq_10 / teq_01 - 10) < 0.01


def test_standard_case(skip_3: bool = False):
    # increase the number of simulations so it gives sensible results
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    mcs_input['CASE_1']['n_simulations'] = 10_000

    mcs = MCS1()
    mcs.inputs = mcs_input
    mcs.n_threads = 1
    mcs.run(keep_results=True)
    outputs = mcs.outputs

    def frac2teq(data, case_name: str, frac: float):
        data = data.loc[outputs['case_name'] == case_name]["solver_time_equivalence_solved"] / 60.
        hist, edges = np.histogram(data, bins=np.arange(0, 181, 0.5))
        x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
        res = interp1d(y, x)(frac)
        print(res)
        return res

    # 60 minutes based on Kirby et al
    assert abs(frac2teq(outputs, 'CASE_1', 0.2) - 60) > 0
    assert abs(frac2teq(outputs, 'CASE_1', 0.4) - 60) > 0
    assert abs(frac2teq(outputs, 'CASE_1', 0.6) - 60) > 0
    assert abs(frac2teq(outputs, 'CASE_1', 0.8) - 60) > 0


def test_file_input():
    # save input as .xlsx
    temp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    EXAMPLE_INPUT_DF.to_excel(temp)
    fp = f'{temp.name}'
    print(f"A temporary input file has been created: {fp}")  # 4
    temp.close()  # 5

    mcs = MCS1()
    mcs.inputs = fp
    mcs.n_threads = 2
    mcs.run()


if __name__ == '__main__':
    test_teq_phi()
    test_standard_case()
    test_file_input()