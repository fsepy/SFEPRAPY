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
        # room_breadth: float,  # depreciated from mcs0
        # room_depth: float,  # depreciated from mcs0
        room_floor_area: float,  # new from mcs0
        room_breadth_depth_ratio: float,  # new from mcs0
        # beam_position_horizontal,  # depreciated from mcs0
        beam_position_horizontal_ratio: float,  # new from mcs0
        # window_width: float,  # depreciated from mcs0
        # window_height: float,  # depreciated from mcs0
        window_height_room_height_ratio: float,  # new from mcs0
        window_area_floor_ratio: float,  # new from mcs0
        **kwargs,
) -> dict:
    # -----------------------------------------
    # Calculate `room_breadth` and `room_depth`
    # -----------------------------------------
    # room_depth * room_breadth = room_floor_area
    # room_breadth / room_depth = room_breadth_depth_ratio

    # room_breadth = room_breadth_depth_ratio * room_depth
    # room_depth * room_breadth_depth_ratio * room_depth = room_floor_area
    room_depth = (room_floor_area / room_breadth_depth_ratio) ** 0.5
    room_breadth = room_breadth_depth_ratio * (room_floor_area / room_breadth_depth_ratio) ** 0.5
    assert 0 < room_breadth_depth_ratio <= 1.  # ensure within (0, 1]
    assert abs(room_depth * room_breadth - room_floor_area) < 1e-5  # ensure calculated room floor dimensions match the prescribed floor area

    # -----------------------------------------
    # Calculate window opening width and height
    # -----------------------------------------
    window_height = window_height_room_height_ratio * kwargs['room_height']
    window_width = room_floor_area * window_area_floor_ratio / window_height
    assert 0 < window_height_room_height_ratio <= 1.  # ensure within (0, 1]

    # --------------------------------
    # Calculate beam vertical location
    # --------------------------------
    beam_position_vertical = kwargs['beam_position_vertical']
    kwargs['beam_position_vertical'] = min(beam_position_vertical, kwargs['room_height'])

    kwargs.update(dict(
        room_breadth=room_depth,
        room_depth=room_breadth,
        window_height=window_height,
        window_width=window_width,
        beam_position_horizontal=room_depth * beam_position_horizontal_ratio
    ))

    outputs = teq_main_mcs0(**kwargs)

    return outputs


class MCS2(MCS0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mcs_deterministic_calc(self, *args, **kwargs) -> dict:
        return teq_main(*args, **kwargs)

    def mcs_deterministic_calc_mp(self, *args, **kwargs) -> dict:
        return teq_main_wrapper(*args, **kwargs)


def _test_standard_case():
    import copy
    from sfeprapy.mcs2 import EXAMPLE_INPUT_DICT
    from scipy.interpolate import interp1d
    import numpy as np

    # increase the number of simulations so it gives sensible results
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    for k in list(mcs_input.keys()):
        mcs_input[k]['n_simulations'] = 10_000

    # increase the number of threads so it runs faster
    mcs2 = MCS2()
    mcs2.inputs = mcs_input
    mcs2.n_threads = 2
    mcs2.run_mcs(cases_to_run=['Office'])
    mcs_out = mcs2.mcs_out
    teq = mcs_out.loc[mcs_out['case_name'] == 'Office']["solver_time_equivalence_solved"] / 60.0
    teq = teq[~np.isnan(teq)]
    hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.5))
    x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    func_teq = interp1d(x, y)
    for fire_rating in [30, 45, 60, 75, 90, 105, 120]:
        print(f'{fire_rating:<8.0f}  {func_teq(fire_rating):<.8f}')

    assert abs(func_teq(60) - 0.64109512) <= 5e-3
    assert abs(func_teq(90) - 0.92440939) <= 5e-3


if __name__ == '__main__':
    _test_standard_case()
