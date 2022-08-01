import copy
import os
from typing import Callable

import numpy as np

from ..mcs0 import MCS0, decide_fire, evaluate_fire_temperature, solve_protection_thickness, \
    solve_time_equivalence_iso834


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
        room_breadth: float,
        room_depth: float,
        room_height: float,
        room_wall_thermal_inertia: float,
        solver_temperature_goal: float,
        solver_max_iter: int,
        solver_thickness_lbound: float,
        solver_thickness_ubound: float,
        solver_tol: float,
        window_height: float,
        window_open_fraction: float,
        window_width: float,
        window_open_fraction_permanent: float,
        phi_teq: float = 1.0,
        timber_charring_rate=None,
        timber_charred_depth=None,
        timber_hc: float = None,
        timber_density: float = None,
        timber_exposed_area: float = None,
        timber_depth: float = None,
        timber_solver_tol: float = None,
        timber_solver_ilim: float = None,
        car_cluster_size: int = None,
        *_,
        **__,
) -> dict:
    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    if room_depth < room_breadth:
        room_depth += room_breadth
        room_breadth = room_depth - room_breadth
        room_depth -= room_breadth

    # todo: wip for car park!!!
    if 'occupancy_type' in __ and __['occupancy_type'] == '__CAR_PARK__':
        fire_mode = 1  # force to travelling fire only
        # work out new room_depth_car based on how many cars are involved in fire
        if car_cluster_size is not None and car_cluster_size >= 0:
            car_cluster_size = int(car_cluster_size) + 1
            room_depth_original = float(room_depth)
            parking_bay_width = 2.3
            n_parking_bay_row = 2
            average_area_per_parking_bay = 4283 / 202

            room_depth = car_cluster_size * parking_bay_width / n_parking_bay_row
            room_floor_area = car_cluster_size * average_area_per_parking_bay
            room_breadth = room_floor_area / room_depth

            beam_position_horizontal = (beam_position_horizontal / room_depth_original) * room_depth

    window_open_fraction = (
            window_open_fraction * (1 - window_open_fraction_permanent) + window_open_fraction_permanent)

    # Fix ventilation opening size so it doesn't exceed wall area
    if window_height > room_height:
        window_height = room_height

    # Calculate fire time, this is used for all fire curves in the calculation
    fire_time = np.arange(0, fire_time_duration + fire_time_step, fire_time_step)

    # Calculate ISO 834 fire temperature
    # fire_time_iso834 = fire_time
    # fire_temperature_iso834 = (345.0 * np.log10((fire_time / 60.0) * 8.0 + 1.0) + 20.0) + 273.15  # in [K]

    inputs = copy.deepcopy(locals())
    inputs.pop('_'), inputs.pop('__')

    # initialise solver iteration count for timber fuel contribution
    timber_solver_iter_count = -1
    timber_exposed_duration = 0  # initial condition, timber exposed duration
    _fire_load_density_ = inputs.pop('fire_load_density')  # preserve original fire load density

    while True:
        timber_solver_iter_count += 1
        # the following `if` decide whether to calculate `timber_charred_depth_i` from `timber_charring_rate` or
        # `timber_charred_depth`
        if timber_charred_depth is None:
            # calculate from timber charring rate
            if isinstance(timber_charring_rate, (float, int)):
                timber_charring_rate_i = timber_charring_rate
            elif isinstance(timber_charring_rate, Callable):
                timber_charring_rate_i = timber_charring_rate(timber_exposed_duration)
            else:
                raise TypeError('`timber_charring_rate_i` is not numerical nor Callable type')
            timber_charring_rate_i *= 1. / 1000.  # [mm/min] -> [m/min]
            timber_charring_rate_i *= 1. / 60.  # [m/min] -> [m/s]
            timber_charred_depth_i = timber_charring_rate_i * timber_exposed_duration
        else:
            # calculate from timber charred depth
            if isinstance(timber_charred_depth, (float, int)):
                timber_charred_depth_i = timber_charred_depth
            elif isinstance(timber_charred_depth, Callable):
                timber_charred_depth_i = timber_charred_depth(timber_exposed_duration)
            else:
                raise TypeError('`timber_charring_rate_i` is not numerical nor Callable type')
            timber_charred_depth_i /= 1000.

        # make sure the calculated charred depth does not exceed the the available timber depth
        if timber_depth is not None:
            timber_charred_depth_i = min(timber_charred_depth_i, timber_depth)

        timber_charred_volume = timber_charred_depth_i * timber_exposed_area
        timber_charred_mass = timber_density * timber_charred_volume
        timber_fire_load = timber_charred_mass * timber_hc
        timber_fire_load_density = timber_fire_load / (room_breadth * room_depth)

        inputs['fire_load_density'] = _fire_load_density_ + timber_fire_load_density

        # To check what design fire to use
        inputs.update(decide_fire(**inputs))

        # To calculate design fire temperature
        inputs.update(evaluate_fire_temperature(**inputs))

        # To solve protection thickness at critical temperature
        inputs.update(solve_protection_thickness(**inputs))

        # To solve time equivalence in ISO 834
        inputs.update(solve_time_equivalence_iso834(**inputs))

        # additional fuel contribution from timber
        if timber_exposed_area <= 0 or timber_exposed_area is None:  # no timber exposed
            # Exit timber fuel contribution solver if:
            #     1. no timber exposed
            #     2. timber exposed area undefined
            break
        elif timber_solver_iter_count >= timber_solver_ilim:
            inputs['solver_convergence_status'] = np.nan
            inputs['solver_time_critical_temp_solved'] = np.nan
            inputs['solver_time_equivalence_solved'] = np.nan
            inputs['solver_steel_temperature_solved'] = np.nan
            inputs['solver_protection_thickness'] = np.nan
            inputs['solver_iter_count'] = np.nan
            timber_exposed_duration = np.nan
            break
        elif not -np.inf < inputs["solver_protection_thickness"] < np.inf:
            # no protection thickness solution
            timber_exposed_duration = inputs['solver_protection_thickness']
            break
        elif abs(timber_exposed_duration - inputs["solver_time_equivalence_solved"]) <= timber_solver_tol:
            # convergence sought successfully
            break
        else:
            timber_exposed_duration = inputs["solver_time_equivalence_solved"]

    inputs.update(
        dict(
            timber_charring_rate=timber_charred_depth_i / timber_exposed_duration if timber_exposed_duration else 0,
            timber_exposed_duration=timber_exposed_duration,
            timber_solver_iter_count=timber_solver_iter_count,
            timber_fire_load=timber_fire_load,
            timber_charred_depth=timber_charred_depth_i,
            timber_charred_mass=timber_charred_mass,
            timber_charred_volume=timber_charred_volume,
        )
    )

    return inputs


class MCS1(MCS0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mcs_deterministic_calc(self, *args, **kwargs) -> dict:
        return teq_main(*args, **kwargs)

    def mcs_deterministic_calc_mp(self, *args, **kwargs) -> dict:
        return teq_main_wrapper(*args, **kwargs)


def cli_main(fp_mcs_in: str, n_threads: int = 1):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS1()
    mcs.inputs = fp_mcs_in
    mcs.n_threads = n_threads
    mcs.run()
