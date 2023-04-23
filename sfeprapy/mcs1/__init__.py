__all__ = (
    'MCS1Single', 'MCS1', 'cli_main',

)

import os

import numpy as np

from .calcs import teq_main
from .inputs import EXAMPLE_INPUT_DICT, EXAMPLE_INPUT_DF, EXAMPLE_INPUT_CSV
from ..mcs import MCSSingle
from ..mcs0 import MCS0


class MCS1Single(MCSSingle):
    INPUT_KEYS = (
        'index', 'beam_cross_section_area', 'beam_position_vertical', 'beam_position_horizontal', 'beam_rho',
        'fire_time_duration', 'fire_time_step', 'fire_combustion_efficiency', 'fire_gamma_fi_q', 'fire_hrr_density',
        'fire_load_density', 'fire_mode', 'fire_nft_limit', 'fire_spread_speed', 'fire_t_alpha', 'fire_tlim',
        'protection_c', 'protection_k', 'protection_protected_perimeter', 'protection_rho', 'protection_d_p',
        'room_breadth', 'room_depth', 'room_height', 'room_wall_thermal_inertia', 'window_height',
        'window_open_fraction', 'window_width', 'window_open_fraction_permanent', 'epsilon_q', 't_k_y_theta', 'phi_teq',
        'timber_charring_rate', 'timber_charred_depth', 'timber_hc', 'timber_density', 'timber_exposed_area',
        'timber_depth', 'timber_solver_tol', 'timber_solver_ilim', 'occupancy_type', 'car_cluster_size'
    )
    OUTPUT_KEYS = (
        'index', 'beam_position_horizontal', 'fire_combustion_efficiency', 'fire_hrr_density', 'fire_nft_limit',
        'fire_spread_speed', 'window_open_fraction', 'epsilon_q', 'fire_load_density', 'fire_type', 't1', 't2', 't3',
        'solver_temperature_goal', 'solver_time_equivalence_solved', 'timber_charring_rate', 'timber_exposed_duration',
        'timber_solver_iter_count', 'timber_fire_load', 'timber_charred_depth', 'timber_charred_mass',
        'timber_charred_volume', 'T_max_t', 'k_y_theta_t',
    )

    def __init__(self, name, n_simulations, sim_kwargs, save_dir):
        super().__init__(name=name, n_simulations=n_simulations, sim_kwargs=sim_kwargs, save_dir=save_dir)

    @staticmethod
    def worker(args) -> tuple:
        return teq_main(*args)

    def get_pdf(self, bin_width: float = 0.2) -> (np.ndarray, np.ndarray, np.ndarray):
        teq: np.ndarray = None
        for i in range(len(self.output_keys)):
            if self.output_keys[i] == 'solver_time_equivalence_solved':
                teq = self.output[:, i]

        return MCS1Single.make_pdf(teq, bin_width=bin_width)

    def get_cdf(self, bin_width: float = 0.2):
        x, y_pdf = self.get_pdf(bin_width=bin_width)

        return x, np.cumsum(y_pdf)

    @property
    def input_keys(self) -> tuple:
        return MCS1Single.INPUT_KEYS

    @property
    def output_keys(self) -> tuple:
        return MCS1Single.OUTPUT_KEYS


class MCS1(MCS0):
    @property
    def new_mcs_case(self):
        return MCS1Single


def cli_main(fp_mcs_in: str, n_threads: int = 1):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS1()
    mcs.inputs = fp_mcs_in
    mcs.n_threads = n_threads
    mcs.run()
