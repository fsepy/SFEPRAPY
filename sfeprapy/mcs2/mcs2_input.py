# -*- coding: utf-8 -*-

import copy as __copy

from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT as EXAMPLE_INPUT_DICT_
from sfeprapy.mcs0.mcs0_input import example_input_df, example_input_csv, example_input_dict


def __example_input_list() -> list:
    # Create base case from `sfeprapy.mcs0`
    # only use "CASE_1"
    base_case = __copy.deepcopy(EXAMPLE_INPUT_DICT_['CASE_1'])
    # remove items which are no longer used in `sfeprapy.mcs2` (comparing to `sfeprapy.mcs0`)
    for i in ['room_breadth', 'room_depth', 'window_width', 'window_height', 'p1', 'p2', 'p3', 'p4',
              'general_room_floor_area', 'beam_position_horizontal']:
        base_case.pop(i)
    # create variable for dumping new inputs
    y = list()

    # Residential
    y.append(__copy.copy(base_case))
    y[-1].update(dict(
        case_name='Residential',
        beam_position_horizontal_ratio=dict(dist='uniform_', lbound=0.6, ubound=0.9),
        fire_load_density=dict(dist="gumbel_r_", lbound=10, ubound=1200, mean=780, sd=234),
        fire_hrr_density=dict(dist="uniform_", lbound=0.32, ubound=0.57),
        phi_teq=dict(dist='lognorm_', mean=1, sd=0.25, ubound=3, lbound=1e-4),
        room_floor_area=dict(dist='uniform_', lbound=9., ubound=30.),
        room_height=dict(dist='constant_', lbound=2.4, ubound=2.4),
        room_breadth_depth_ratio=dict(dist='uniform_', lbound=0.4, ubound=0.6),  # todo
        window_height_room_height_ratio=dict(dist='uniform_', lbound=0.3, ubound=0.9),
        window_area_floor_ratio=dict(dist='uniform_', lbound=0.05, ubound=0.20),
    ))

    # Office
    y.append(__copy.copy(base_case))
    y[-1].update(dict(
        case_name='Office',
        beam_position_horizontal_ratio=dict(dist='uniform_', lbound=0.6, ubound=0.9),
        fire_load_density=dict(dist="gumbel_r_", lbound=10, ubound=1200, mean=420, sd=126),
        fire_hrr_density=dict(dist="uniform_", lbound=0.15, ubound=0.65),
        room_floor_area=dict(dist='uniform_', lbound=50., ubound=1000.),
        room_height=dict(dist='uniform_', lbound=2.8, ubound=4.5),
        room_breadth_depth_ratio=dict(dist='uniform_', lbound=0.4, ubound=0.6),  # todo
        window_height_room_height_ratio=dict(dist='uniform_', lbound=0.3, ubound=0.9),
        window_area_floor_ratio=dict(dist='uniform_', lbound=0.05, ubound=0.40),
        phi_teq=dict(dist='lognorm_', mean=1, sd=0.25, ubound=3, lbound=1e-4),
    ))

    # Retail
    y.append(__copy.copy(base_case))
    y[-1].update(dict(
        case_name='Retail',
        beam_position_horizontal_ratio=dict(dist='uniform_', lbound=0.6, ubound=0.9),
        fire_load_density=dict(dist="gumbel_r_", lbound=10., ubound=2000., mean=600., sd=180.),
        fire_hrr_density=dict(dist="uniform_", lbound=0.27, ubound=1.0),
        room_floor_area=dict(dist='uniform_', lbound=50., ubound=1000.),
        room_height=dict(dist='constant_', lbound=4.5, ubound=7.0),
        room_breadth_depth_ratio=dict(dist='uniform_', lbound=0.4, ubound=0.6),  # todo
        window_height_room_height_ratio=dict(dist='uniform_', lbound=0.5, ubound=1.0),
        window_area_floor_ratio=dict(dist='uniform_', lbound=0.05, ubound=0.40),
        phi_teq=dict(dist='lognorm_', mean=1, sd=0.25, ubound=3, lbound=1e-4),
    ))

    return y


EXAMPLE_INPUT_DICT = example_input_dict(__example_input_list())
EXAMPLE_INPUT_CSV = example_input_csv(__example_input_list())
EXAMPLE_INPUT_DF = example_input_df(__example_input_list())

if __name__ == "__main__":
    import pprint

    pprint.pprint(EXAMPLE_INPUT_DICT)
    pprint.pprint(EXAMPLE_INPUT_CSV)
    pprint.pprint(EXAMPLE_INPUT_DF)
