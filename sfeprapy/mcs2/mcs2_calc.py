import os

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
    assert abs(
        room_depth * room_breadth - room_floor_area) < 1e-5  # ensure calculated room floor dimensions match the prescribed floor area

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


def cli_main(fp_mcs_in: str, n_threads: int = 1):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS2()
    mcs.inputs = fp_mcs_in
    mcs.n_threads = n_threads
    mcs.run()
