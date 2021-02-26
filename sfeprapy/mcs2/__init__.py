# -*- coding: utf-8 -*-

import copy

from sfeprapy.mcs0 import __example_config_dict, __example_input_df, __example_input_csv, __example_input_dict


def __example_input_list() -> list:
    # =====================================
    # Create base case from `sfeprapy.mcs0`
    # =====================================
    # use example `sfeprapy.mcs0` example inputs
    from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT as EXAMPLE_INPUT_DICT_
    # only use "Standard Case 1"
    base_case = copy.deepcopy(EXAMPLE_INPUT_DICT_['Standard Case 1'])
    # remove items which are no longer used in `sfeprapy.mcs2` (comparing to `sfeprapy.mcs0`)
    [base_case.pop(i) for i in ['room_breadth', 'room_depth', 'window_width', 'p1', 'p2', 'p3', 'p4', 'general_room_floor_area']]
    # create variable for dumping new inputs
    y = list()

    # ===========
    # Residential
    # ===========
    y.append(copy.copy(base_case))
    y[-1].update(dict(
        fire_mode=0,  # force to use BS EN 1991-1-2 parametric fire
        fire_load_density=dict(dist="gumbel_r_", lbound=10, ubound=1200, mean=780, sd=234),
        fire_hrr_density=dict(dist="uniform_", lbound=0.25 - 0.001, ubound=0.25 + 0.001),
        room_floor_area=dict(dist='uniform_', lbound=9., ubound=30.),
        room_height=dict(dist='constant_', lbound=2.4, ubound=2.4),
        room_breadth_depth_ratio=dict(dist='uniform_', lbound=0.512 - 0.2, ubound=0.512 + 0.2),  # todo
        window_height_room_height_ratio=dict(dist='uniform_', lbound=0.3, ubound=0.9),
        window_area_floor_ratio=dict(dist='uniform_', lbound=0.05, ubound=0.20),
        case_name='Residential',
        phi_teq=1.,
    ))

    # ======
    # Office
    # ======
    y.append(copy.copy(base_case))
    y[-1].update(dict(
        fire_mode=0,  # force to use BS EN 1991-1-2 parametric fire
        fire_load_density=dict(dist="gumbel_r_", lbound=10, ubound=1200, mean=420, sd=126),
        fire_hrr_density=dict(dist="uniform_", lbound=0.25 - 0.001, ubound=0.25 + 0.001),
        room_floor_area=dict(dist='uniform_', lbound=50., ubound=1000.),
        room_height=dict(dist='uniform_', lbound=2.8, ubound=4.5),
        room_breadth_depth_ratio=dict(dist='uniform_', lbound=0.512 - 0.2, ubound=0.512 + 0.2),
        window_height_room_height_ratio=dict(dist='uniform_', lbound=0.3, ubound=0.9),
        window_area_floor_ratio=dict(dist='uniform_', lbound=0.05, ubound=0.40),
        case_name='Office',
        phi_teq=1.,
    ))

    # ======
    # Retail
    # ======
    y.append(copy.copy(base_case))
    y[-1].update(dict(
        fire_mode=0,  # force to use BS EN 1991-1-2 parametric fire
        fire_load_density=dict(dist="gumbel_r_", lbound=10., ubound=2000., mean=600., sd=180.),
        fire_hrr_density=dict(dist="uniform_", lbound=0.25 - 0.001, ubound=0.25 + 0.001),
        room_floor_area=dict(dist='constant_', lbound=400., ubound=400.),
        room_height=dict(dist='uniform_', lbound=4.5, ubound=7.0),
        room_breadth_depth_ratio=dict(dist='uniform_', lbound=0.512 - 0.2, ubound=0.512 + 0.2),
        window_height_room_height_ratio=dict(dist='uniform_', lbound=0.5, ubound=1.0),
        window_area_floor_ratio=dict(dist='uniform_', lbound=0.05, ubound=0.40),
        case_name='Retail',
        phi_teq=1.,
    ))

    return y


EXAMPLE_CONFIG_DICT = __example_config_dict()
EXAMPLE_INPUT_DICT = __example_input_dict(__example_input_list())
EXAMPLE_INPUT_CSV = __example_input_csv(__example_input_list())
EXAMPLE_INPUT_DF = __example_input_df(__example_input_list())

if __name__ == "__main__":
    print(EXAMPLE_CONFIG_DICT, "\n")
    print(EXAMPLE_INPUT_DICT, "\n")
    print(EXAMPLE_INPUT_CSV, "\n")
