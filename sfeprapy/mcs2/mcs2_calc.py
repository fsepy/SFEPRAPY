# -*- coding: utf-8 -*-
from sfeprapy.mcs0.mcs0_calc import MCS0
from sfeprapy.mcs0.mcs0_calc import teq_main as teq_main_mcs0


def teq_main_wrapper(args):
    try:
        kwargs, q = args
        q.put("index: {}".format(kwargs["index"]))
        return teq_main(**kwargs)
    except (ValueError, AttributeError):
        return teq_main(**args)


def teq_main(
        case_name: str,
        n_simulations: int,
        probability_weight: float,
        index: int,
        beam_cross_section_area: float,
        beam_position_vertical: float,
        beam_position_horizontal: float,
        beam_rho: float,
        fire_time_duration: float,
        fire_time_step: float,
        fire_combustion_efficiency: float,
        fire_gamma_fi_q: float,
        fire_hrr_density: float,
        fire_load_density: float,
        fire_mode: int,
        fire_nft_limit: float,
        fire_spread_speed: float,
        fire_t_alpha: float,
        fire_tlim: float,
        protection_c: float,
        protection_k: float,
        protection_protected_perimeter: float,
        protection_rho: float,
        # room_breadth: float,
        # room_depth: float,
        room_wall_thermal_inertia: float,
        room_floor_area: float,
        room_breadth_depth_ratio: float,
        room_height: float,
        solver_temperature_goal: float,
        solver_max_iter: int,
        solver_thickness_lbound: float,
        solver_thickness_ubound: float,
        solver_tol: float,
        # window_width: float,
        window_height: float,
        window_floor_ratio: float,
        window_open_fraction: float,
        window_open_fraction_permanent: float,
        phi_teq: float = 1.0,
        timber_charring_rate=None,
        timber_hc: float = None,
        timber_density: float = None,
        timber_exposed_area: float = None,
        timber_solver_tol: float = None,
        timber_solver_ilim: float = None,
        *_,
        **__,
) -> dict:
    kwargs = locals()
    _ = kwargs.pop('_')
    __ = kwargs.pop('__')

    kwargs.pop('room_floor_area')
    kwargs.pop('room_breadth_depth_ratio')

    # -----------------------------------------
    # Calculate `room_breadth` and `room_depth`
    # -----------------------------------------
    # room_depth * room_breadth = room_floor_area
    # room_breadth / room_depth = room_breadth_depth_ratio

    # room_breadth = room_breadth_depth_ratio * room_depth
    # room_depth * room_breadth_depth_ratio * room_depth = room_floor_area
    room_depth = (room_floor_area / room_breadth_depth_ratio) ** 0.5
    room_breadth = room_breadth_depth_ratio * (room_floor_area / room_breadth_depth_ratio) ** 0.5
    assert room_breadth_depth_ratio <= 1
    assert abs(room_depth * room_breadth - room_floor_area) < 1e-5

    # ------------------------------
    # Calculate window opening width
    # ------------------------------
    window_width = room_floor_area * window_floor_ratio / window_height

    # ----------------------------------
    # Calculate beam horizontal location
    # ----------------------------------

    kwargs.update(dict(
        room_breadth=room_depth,
        room_depth=room_breadth,
        window_width=window_width,
        beam_horizontal_location=0.8 * room_depth
    ))

    outputs = teq_main_mcs0(**kwargs)

    return outputs


class MCS2(MCS0):
    def __init__(self):
        super().__init__()

    def mcs_deterministic_calc(self, *args, **kwargs) -> dict:
        return teq_main(*args, **kwargs)

    def mcs_deterministic_calc_mp(self, *args, **kwargs) -> dict:
        return teq_main_wrapper(*args, **kwargs)


def _test_standard_case():
    import copy
    from sfeprapy.mcs2 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
    from scipy.interpolate import interp1d
    import numpy as np

    # increase the number of simulations so it gives sensible results
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    mcs_config = copy.deepcopy(EXAMPLE_CONFIG_DICT)
    for k in list(mcs_input.keys()):
        mcs_input[k]["phi_teq"] = 1
        mcs_input[k]["n_simulations"] = 10000
        mcs_input[k]["probability_weight"] = 1 / 3.0
        mcs_input[k]["fire_time_duration"] = 10000
        mcs_input[k]["timber_exposed_area"] = 0

    # increase the number of threads so it runs faster
    mcs_config["n_threads"] = 1  # coverage does not support
    mcs2 = MCS2()
    mcs2.mcs_inputs = mcs_input
    mcs2.mcs_config = mcs_config
    mcs2.run_mcs()
    mcs_out = mcs2.mcs_out
    teq = mcs_out["solver_time_equivalence_solved"] / 60.0
    hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.5))
    x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    teq_at_80_percentile = interp1d(y, x)(0.8)
    print(teq_at_80_percentile)
    # target, target_tol = 60, 2
    # assert target - target_tol < teq_at_80_percentile < target + target_tol


if __name__ == '__main__':
    # _test_teq_phi()
    # _test_standard_case()
    _test_standard_case()
