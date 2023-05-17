__all__ = (
    'MCS1Single', 'MCS1', 'cli_main',

)

import os

import numpy as np

from .calcs import teq_main
from .inputs import EXAMPLE_INPUT
from ..mcs import MCSSingle
from ..mcs0 import MCS0


class MCS1Single(MCSSingle):
    OUTPUT_KEYS = (
        'index', 'beam_position_horizontal', 'fire_combustion_efficiency', 'fire_hrr_density', 'fire_nft_limit',
        'fire_spread_speed', 'window_open_fraction', 'epsilon_q', 'fire_load_density', 'fire_type', 't1', 't2', 't3',
        'solver_temperature_goal', 'solver_time_equivalence_solved', 'timber_charring_rate', 'timber_exposed_duration',
        'timber_solver_iter_count', 'timber_fire_load', 'timber_charred_depth', 'timber_charred_mass',
        'timber_charred_volume', 'T_max_t', 'k_y_theta_t',
    )

    def __init__(self, name, n_simulations, sim_kwargs, save_dir):
        super().__init__(name=name, n_simulations=n_simulations, sim_kwargs=sim_kwargs, save_dir=save_dir)

    @property
    def worker(self):
        return teq_main

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
