__all__ = (
    'decide_fire', 'evaluate_fire_temperature', 'solve_time_equivalence_iso834',
    'teq_main'
)

import logging
from typing import Callable, Any

import numpy as np
from fsetools.lib.fse_bs_en_1993_1_2_heat_transfer_c import temperature as _steel_temperature
from fsetools.lib.fse_bs_en_1993_1_2_strength_reduction_factor import k_y_theta_prob

from ..mcs0 import decide_fire, evaluate_fire_temperature

logger = logging.getLogger('gui')


def solve_time_equivalence_iso834(
        fire_time: np.ndarray,
        fire_temperature: np.ndarray,
        beam_cross_section_area: float,
        beam_rho: float,
        protection_k: float,
        protection_rho: float,
        protection_c: float,
        protection_protected_perimeter: float,
        solver_protection_thickness: float,
        phi_teq: float,
) -> tuple[Any, Any, float | Any]:
    """
    Calculates equivalent time exposure for a protected steel element member in more realistic fire environment (i.e. travelling fire, parameteric fires)
    opposing to the standard fire curve ISO 834.

    PARAMETERS:
    :param beam_cross_section_area:             [m2], the steel beam element cross-section area
    :param beam_rho:                            [kg/m3], steel beam element density
    :param protection_k:                        [], steel beam element protection material thermal conductivity
    :param protection_rho:                      [kg/m3], steel beam element protection material density
    :param protection_c:                        [], steel beam element protection material specific heat
    :param protection_protected_perimeter:      [m], steel beam element protection material perimeter
    :param solver_max_iter:                     [-], Maximum allowable iteration counts for seeking solution for time equivalence
    :param solver_thickness_ubound:             [m], protection layer thickness upper bound initial condition for solving time equivalence
    :param solver_thickness_lbound:             [m], protection layer thickness lower bound initial condition for solving time equivalence
    :param solver_tol:                          [K], tolerance for solving time equivalence
    :param solver_protection_thickness:         [m], steel section protection layer thickness
    :param phi_teq:                             [-], model uncertainty factor
    :return results:                            A dict containing `solver_time_equivalence_solved` which is ,[s], solved equivalent time exposure
    EXAMPLE:
    """

    steel_temperature = _steel_temperature(
        fire_time=fire_time,
        fire_temperature=fire_temperature,
        beam_rho=beam_rho,
        beam_cross_section_area=beam_cross_section_area,
        protection_k=protection_k,
        protection_rho=protection_rho,
        protection_c=protection_c,
        protection_thickness=solver_protection_thickness,
        protection_protected_perimeter=protection_protected_perimeter,
    )
    solver_temperature_goal = max(steel_temperature)

    # Solve equivalent time exposure in ISO 834
    solver_time_equivalence_solved = 1.06011 * 2.71828182845905 ** (0.00667416 * solver_temperature_goal) - 7.5

    return steel_temperature, solver_temperature_goal, solver_time_equivalence_solved * phi_teq


def teq_main(
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
        protection_d_p: float,
        room_breadth: float,
        room_depth: float,
        room_height: float,
        room_wall_thermal_inertia: float,
        # solver_temperature_goal: float,
        # solver_max_iter: int,
        # solver_thickness_lbound: float,
        # solver_thickness_ubound: float,
        # solver_tol: float,
        window_height: float,
        window_open_fraction: float,
        window_width: float,
        window_open_fraction_permanent: float,
        epsilon_q: float,
        t_k_y_theta: int,
        phi_teq: float = 1.0,
        timber_charring_rate=np.nan,
        timber_charred_depth=np.nan,
        timber_hc: float = np.nan,
        timber_density: float = np.nan,
        timber_exposed_area: float = np.nan,
        timber_depth: float = np.nan,
        timber_solver_tol: float = np.nan,
        timber_solver_ilim: float = np.nan,
        occupancy_type: str = np.nan,
        car_cluster_size: int = np.nan,
) -> tuple:
    # Make the longest dimension between (room_depth, room_breadth) as room_depth
    if room_depth < room_breadth:
        room_depth += room_breadth
        room_breadth = room_depth - room_breadth
        room_depth -= room_breadth

    # todo: wip for car park!!!
    if occupancy_type == '__CAR_PARK__':
        fire_mode = 1  # force to travelling fire only
        # work out new room_depth_car based on how many cars are involved in fire
        if (car_cluster_size is not None or car_cluster_size != np.nan) and car_cluster_size >= 0:
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

    # Fix ventilation opening size, so it doesn't exceed wall area
    if window_height > room_height:
        window_height = room_height

    # Calculate fire time, this is used for all fire curves in the calculation
    fire_time = np.arange(0, fire_time_duration + fire_time_step, fire_time_step)

    # Calculate ISO 834 fire temperature
    # fire_time_iso834 = fire_time
    # fire_temperature_iso834 = (345.0 * np.log10((fire_time / 60.0) * 8.0 + 1.0) + 20.0) + 273.15  # in [K]

    # initialise solver iteration count for timber fuel contribution
    timber_solver_iter_count = -1
    timber_exposed_duration = 0  # initial condition, timber exposed duration
    _fire_load_density_ = float(fire_load_density)  # preserve original fire load density

    while True:
        timber_solver_iter_count += 1
        # the following `if` decide whether to calculate `timber_charred_depth_i` from `timber_charring_rate` or
        # `timber_charred_depth`
        if timber_charred_depth is None or timber_charred_depth != np.nan:
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
        if timber_depth is not None or timber_depth != np.nan:
            timber_charred_depth_i = min(timber_charred_depth_i, timber_depth)

        timber_charred_volume = timber_charred_depth_i * timber_exposed_area
        timber_charred_mass = timber_density * timber_charred_volume
        timber_fire_load = timber_charred_mass * timber_hc
        timber_fire_load_density = timber_fire_load / (room_breadth * room_depth)

        if np.isnan(timber_fire_load_density):
            fire_load_density = _fire_load_density_
        else:
            fire_load_density = _fire_load_density_ + timber_fire_load_density

        # To check what design fire to use
        fire_type = decide_fire(
            window_height=window_height, window_width=window_width, window_open_fraction=window_open_fraction,
            room_breadth=room_breadth, room_depth=room_depth, room_height=room_height, fire_mode=fire_mode,
            fire_load_density=fire_load_density, fire_combustion_efficiency=fire_combustion_efficiency,
            fire_hrr_density=fire_hrr_density, fire_spread_speed=fire_spread_speed
        )

        # To calculate design fire temperature
        fire_temperature, beam_position_horizontal, t1, t2, t3 = evaluate_fire_temperature(
            window_height=window_height, window_width=window_width, window_open_fraction=window_open_fraction,
            room_breadth=room_breadth, room_depth=room_depth, room_height=room_height,
            room_wall_thermal_inertia=room_wall_thermal_inertia, fire_tlim=fire_tlim, fire_type=fire_type,
            fire_time=fire_time, fire_nft_limit=fire_nft_limit, fire_load_density=fire_load_density,
            fire_combustion_efficiency=fire_combustion_efficiency, fire_hrr_density=fire_hrr_density,
            fire_spread_speed=fire_spread_speed, fire_t_alpha=fire_t_alpha, fire_gamma_fi_q=fire_gamma_fi_q,
            beam_position_vertical=beam_position_vertical, beam_position_horizontal=beam_position_horizontal
        )

        # To solve protection thickness at critical temperature
        # inputs.update(solve_protection_thickness(**inputs))

        # To solve time equivalence in ISO 834
        steel_temperature, solver_temperature_goal, solver_time_equivalence_solved = solve_time_equivalence_iso834(
            fire_time=fire_time, fire_temperature=fire_temperature, beam_cross_section_area=beam_cross_section_area,
            beam_rho=beam_rho, protection_k=protection_k, protection_rho=protection_rho, protection_c=protection_c,
            protection_protected_perimeter=protection_protected_perimeter, solver_protection_thickness=protection_d_p,
            phi_teq=phi_teq
        )

        # additional fuel contribution from timber
        if timber_exposed_area <= 0 or timber_exposed_area is None or timber_exposed_area != np.nan:  # no timber exposed
            # Exit timber fuel contribution solver if:
            #     1. no timber exposed
            #     2. timber exposed area undefined
            break
        elif timber_solver_iter_count >= timber_solver_ilim:
            solver_convergence_status = np.nan
            solver_time_critical_temp_solved = np.nan
            solver_time_equivalence_solved = np.nan
            solver_steel_temperature_solved = np.nan
            solver_protection_thickness = np.nan
            solver_iter_count = np.nan
            timber_exposed_duration = np.nan
            break
        elif not -np.inf <= protection_d_p <= np.inf:
            # no protection thickness solution
            timber_exposed_duration = protection_d_p
            break
        elif abs(timber_exposed_duration - solver_time_equivalence_solved) <= timber_solver_tol:
            # convergence sought successfully
            break
        else:
            timber_exposed_duration = solver_time_equivalence_solved

        timber_charring_rate = timber_charred_depth_i / timber_exposed_duration if timber_exposed_duration else 0
        timber_exposed_duration = timber_exposed_duration
        timber_solver_iter_count = timber_solver_iter_count
        timber_fire_load = timber_fire_load
        timber_charred_depth = timber_charred_depth_i
        timber_charred_mass = timber_charred_mass
        timber_charred_volume = timber_charred_volume

    t_k_y_theta = int(t_k_y_theta)
    T_max_t = list()
    k_y_theta_t = list()
    for t_ in range(t_k_y_theta, int(fire_time_duration + t_k_y_theta / 2), t_k_y_theta):
        T_max_t.append(np.max(steel_temperature[fire_time <= t_]))

    for t_ in range(t_k_y_theta, int(fire_time_duration + t_k_y_theta / 2), t_k_y_theta):
        k_y_theta_t.append(k_y_theta_prob(T_max_t[-1], epsilon_q=epsilon_q))

    return (
        index, beam_position_horizontal, fire_combustion_efficiency, fire_hrr_density, fire_nft_limit,
        fire_spread_speed, window_open_fraction, epsilon_q, fire_load_density, fire_type, t1, t2, t3,
        solver_temperature_goal, solver_time_equivalence_solved, timber_charring_rate, timber_exposed_duration,
        timber_solver_iter_count, timber_fire_load, timber_charred_depth, timber_charred_mass, timber_charred_volume,
        *T_max_t, *k_y_theta_t,
    )
