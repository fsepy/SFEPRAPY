__all__ = (
    'EXAMPLE_INPUT',

    'decide_fire', 'evaluate_fire_temperature', 'solve_time_equivalence_iso834', 'solve_protection_thickness',
    'teq_main',

    'MCS0Case'
)

from typing import Tuple

from .calcs import (
    decide_fire, evaluate_fire_temperature, solve_time_equivalence_iso834, solve_protection_thickness, teq_main,
)
from .inputs import EXAMPLE_INPUT
from ..mcs import MCSCase


class MCS0Case(MCSCase):
    @staticmethod
    def get_input_keys() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        return (
            (
                'index', 'beam_cross_section_area', 'beam_position_vertical', 'beam_position_horizontal', 'beam_rho',
                'fire_time_duration', 'fire_time_step', 'fire_combustion_efficiency', 'fire_gamma_fi_q',
                'fire_hrr_density', 'fire_load_density', 'fire_mode', 'fire_nft_limit', 'fire_spread_speed',
                'fire_t_alpha', 'fire_tlim', 'protection_c', 'protection_k', 'protection_protected_perimeter',
                'protection_rho', 'room_breadth', 'room_depth', 'room_height', 'room_wall_thermal_inertia',
                'solver_temperature_goal', 'solver_max_iter', 'solver_thickness_lbound', 'solver_thickness_ubound',
                'solver_tol', 'window_height', 'window_open_fraction', 'window_width', 'window_open_fraction_permanent',
            ), (
                'phi_teq', 'timber_exposed_area', 'timber_charred_depth', 'timber_charring_rate', 'timber_hc',
                'timber_density', 'timber_depth', 'timber_solver_tol', 'timber_solver_ilim', 'occupancy_type',
                'car_cluster_size',
            )
        )

    @staticmethod
    def get_output_keys() -> Tuple[str, ...]:
        return (
            'index', 'fire_type', 't1', 't2', 't3', 'solver_steel_temperature_solved',
            'solver_time_critical_temp_solved', 'solver_protection_thickness,', 'solver_iter_count',
            'solver_time_equivalence_solved', 'solver_time_equivalence_solved_with_uncertainty',
            'timber_exposed_duration', 'timber_solver_iter_count', 'timber_fire_load', 'timber_charred_depth',
            'timber_charred_mass', 'timber_charred_volume',
        )

    @staticmethod
    def main_func(*args, **kwargs) -> tuple:
        return teq_main(*args, **kwargs)
