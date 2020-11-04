import copy
from typing import Callable

import numpy as np
from fsetools.lib.fse_travelling_fire_flux import heat_flux as _travelling_fire_flux

from sfeprapy.mcs0.mcs0_calc import MCS0
from sfeprapy.mcs0.mcs0_calc import decide_fire, evaluate_fire_temperature, solve_time_equivalence_iso834, solve_protection_thickness


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
        timber_charring_rate=0,
        timber_hc: float = 0,
        timber_density: float = 0,
        timber_solver_tol: float = 0,
        timber_solver_ilim: float = 0,

        # New parameters based upon `sfeprapy.mcs0`
        timber_Q_crit: float = 12.6,
        timber_total_exposed_area: float = 0.,
        timber_exposed_breadth: float = 0.,

        # Depreciated parameters based on `sfeprapy mcs0`
        # timber_exposed_area: float = 0,

        *_,
        **__,
) -> dict:
    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    if room_depth < room_breadth:
        room_depth += room_breadth
        room_breadth = room_depth - room_breadth
        room_depth -= room_breadth

    window_open_fraction = (window_open_fraction * (1 - window_open_fraction_permanent) + window_open_fraction_permanent)

    # Fix ventilation opening size so it doesn't exceed wall area
    if window_height > room_height:
        window_height = room_height

    # Calculate fire time, this is used for all fire curves in the calculation
    fire_time = np.arange(0, fire_time_duration + fire_time_step, fire_time_step)

    # Calculate ISO 834 fire temperature
    fire_time_iso834 = fire_time
    fire_temperature_iso834 = (345.0 * np.log10((fire_time / 60.0) * 8.0 + 1.0) + 20.0) + 273.15  # in [K]

    inputs = copy.deepcopy(locals())
    inputs.pop('_'), inputs.pop('__')

    # initialise solver iteration count for timber fuel contribution
    timber_solver_iter_count = -1
    timber_exposed_duration = 0.  # initial condition, timber exposed duration
    timber_fire_load_density = 0.
    _fire_load_density_ = copy.copy(inputs['fire_load_density'])  # preserve original fire load density

    timber_charring_rate_i = 0.
    timber_fire_load = 0.
    timber_charred_depth = 0.
    timber_charred_mass = 0.
    timber_charred_volume = 0.
    timber_exposed_area = timber_total_exposed_area

    while True:

        # To check what design fire to use
        inputs.update(decide_fire(**inputs))

        # To calculate design fire temperature
        inputs.update(evaluate_fire_temperature(**inputs))

        # To solve protection thickness at critical temperature
        inputs.update(solve_protection_thickness(**inputs))

        # To solve time equivalence in ISO 834
        inputs.update(solve_time_equivalence_iso834(**inputs))

        inputs.update(
            dict(
                timber_charring_rate=timber_charring_rate_i,
                timber_exposed_duration=timber_exposed_duration,
                timber_exposed_area=timber_exposed_area,
                timber_solver_iter_count=timber_solver_iter_count,
                timber_fire_load=timber_fire_load,
                timber_charred_depth=timber_charred_depth,
                timber_charred_mass=timber_charred_mass,
                timber_charred_volume=timber_charred_volume,
            )
        )

        # additional fuel contribution from timber
        if timber_exposed_area <= 0 or timber_exposed_area is None:  # no timber exposed
            # Exit timber fuel contribution solver if:
            #     1. no timber exposed
            #     2. timber exposed area undefined
            break
        elif timber_solver_iter_count >= timber_solver_ilim:
            inputs['solver_convergence_status'] = np.nan
            inputs['solver_steel_temperature_solved'] = np.nan
            inputs['solver_time_solved'] = np.nan
            inputs['solver_protection_thickness'] = np.nan
            inputs['solver_iter_count'] = np.nan
            timber_exposed_duration = np.nan
            break
        elif not -np.inf < inputs["solver_protection_thickness"] < np.inf:
            # no protection thickness solution
            timber_exposed_duration = inputs['solver_protection_thickness']
            break
        elif abs(timber_exposed_duration - inputs["solver_time_solved"]) <= timber_solver_tol:
            # convergence sought successfully
            break
        else:
            timber_solver_iter_count += 1

            timber_exposed_duration = inputs["solver_time_equivalence_solved"]

            # Calculate timber exposed area if travelling fire
            if inputs['fire_type'] == 1:
                heat_flux_kW = _travelling_fire_flux(
                    t=fire_time,
                    fire_load_density_MJm2=(_fire_load_density_ + timber_fire_load_density) * fire_combustion_efficiency,
                    fire_hrr_density_MWm2=fire_hrr_density,
                    room_length_m=room_depth,
                    room_width_m=room_breadth,
                    fire_spread_rate_ms=fire_spread_speed,
                    beam_location_height_m=beam_position_vertical,
                    beam_location_length_m=0.5 * room_breadth,
                    fire_nff_limit_kW=120,  # todo, this should be an input variable
                )
                _time_timber_Q_crit = fire_time[heat_flux_kW > timber_Q_crit]
                try:
                    timber_exposed_area_depth = (max(_time_timber_Q_crit) - min(_time_timber_Q_crit)) * fire_spread_speed
                except ValueError:
                    timber_exposed_area_depth = 0
                timber_exposed_area = min(timber_total_exposed_area, timber_exposed_area_depth * timber_exposed_breadth)
            else:
                timber_exposed_area = timber_total_exposed_area

            # calculate resultant fuel load density with contribution from timber conbustion
            if isinstance(timber_charring_rate, (float, int)):
                timber_charring_rate_i = timber_charring_rate * (1 / 60000)  # [mm/min] -> [m/s]
            elif isinstance(timber_charring_rate, Callable):
                timber_charring_rate_i = timber_charring_rate(timber_exposed_duration) * (1 / 60000)  # [mm/min] -> [m/s]
            else:
                raise TypeError('`timber_charring_rate_i` is not numerical nor Callable type')
            timber_charred_depth = timber_charring_rate_i * timber_exposed_duration
            timber_charred_volume = timber_charred_depth * timber_exposed_area
            timber_charred_mass = timber_density * timber_charred_volume
            timber_fire_load = timber_charred_mass * timber_hc
            timber_fire_load_density = timber_fire_load / (room_breadth * room_depth)

            inputs['fire_load_density'] = _fire_load_density_ + timber_fire_load_density

    # inputs.update(
    #     dict(
    #         timber_charring_rate=0.,
    #         timber_exposed_duration=timber_exposed_duration,
    #         timber_exposed_area=timber_exposed_area,
    #         timber_solver_iter_count=timber_solver_iter_count,
    #         timber_fire_load=0.,
    #         timber_charred_depth=0.,
    #         timber_charred_mass=0.,
    #         timber_charred_volume=0.,
    #     )
    # )

    # Prepare results to be returned, only the items in the list below will be returned
    # add keys accordingly if more parameters are desired to be returned
    outputs = {
        i: inputs[i] for i in
        ['phi_teq', 'fire_spread_speed', 'fire_nft_limit', 'fire_mode', 'fire_load_density', 'fire_hrr_density', 'fire_combustion_efficiency', 'beam_position_horizontal',
         # 'beam_position_vertical', 'index', 'probability_weight', 'case_name', 'fire_type', 'solver_convergence_status', 'solver_time_equivalence_solved',
         'beam_position_vertical', 'index', 'case_name', 'fire_type', 'solver_convergence_status', 'solver_time_equivalence_solved',
         'solver_steel_temperature_solved', 'solver_protection_thickness', 'solver_iter_count', 'window_open_fraction', 'timber_solver_iter_count', 'timber_charred_depth',
         'timber_exposed_area']
    }

    return outputs


class MCS3(MCS0):
    def __init__(self):
        super().__init__()

    def mcs_deterministic_calc(self, *args, **kwargs) -> dict:
        return teq_main(*args, **kwargs)

    def mcs_deterministic_calc_mp(self, *args, **kwargs) -> dict:
        return teq_main_wrapper(*args, **kwargs)


def _test_standard_case():
    import copy
    from sfeprapy.mcs3 import EXAMPLE_INPUT_DICT, EXAMPLE_CONFIG_DICT
    from scipy.interpolate import interp1d
    import numpy as np

    # increase the number of simulations so it gives sensible results
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    mcs_config = copy.deepcopy(EXAMPLE_CONFIG_DICT)

    # increase the number of threads so it runs faster
    mcs_config["n_threads"] = 1  # coverage does not support
    mcs3 = MCS3()
    mcs3.mcs_inputs = mcs_input
    mcs3.mcs_config = mcs_config
    mcs3.run_mcs()
    mcs_out = mcs3.mcs_out

    def get_time_equivalence(data, fractile: float):
        hist, edges = np.histogram(data, bins=np.arange(0, 181, 0.5))
        x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
        return interp1d(y, x)(fractile)

    mcs_out_standard_case_1 = mcs_out.loc[mcs_out['case_name'] == 'Standard Case 1']
    teq = mcs_out_standard_case_1["solver_time_equivalence_solved"] / 60.0
    teq_at_80_percentile = get_time_equivalence(teq, 0.8)
    print(f'Time equivalence at CDF 0.8 is {teq_at_80_percentile:<6.3f} min')
    target, target_tol = 60, 2
    assert target - target_tol < teq_at_80_percentile < target + target_tol

    mcs_out_standard_case_2 = mcs_out.loc[mcs_out['case_name'] == 'Standard Case 2 (with teq_phi)']
    teq = mcs_out_standard_case_2["solver_time_equivalence_solved"] / 60.0
    teq_at_80_percentile = get_time_equivalence(teq, 0.8)
    print(f'Time equivalence at CDF 0.8 is {teq_at_80_percentile:<6.3f} min')
    target, target_tol = 64, 2  # 64 minutes based on a test run on 2nd Oct 2020
    assert target - target_tol < teq_at_80_percentile < target + target_tol

    mcs_out_standard_case_3 = mcs_out.loc[mcs_out['case_name'] == 'Standard Case 3 (with timber)']
    teq = mcs_out_standard_case_3["solver_time_equivalence_solved"] / 60.0
    teq_at_80_percentile = get_time_equivalence(teq, 0.8)
    print(f'Time equivalence at CDF 0.8 is {teq_at_80_percentile:<6.3f} min')
    target, target_tol = 90, 2  # 81 minutes based on a test run on 2nd Oct 2020
    # assert target - target_tol < teq_at_80_percentile < target + target_tol


if __name__ == '__main__':
    _test_standard_case()
