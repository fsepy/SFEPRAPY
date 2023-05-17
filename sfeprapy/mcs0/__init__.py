__all__ = (
    'EXAMPLE_INPUT',
    'decide_fire', 'evaluate_fire_temperature', 'solve_time_equivalence_iso834', 'solve_protection_thickness',
    'teq_main',
    'MCS0', 'MCS0Single',
)

import os
from typing import Callable

import numpy as np

from .calcs import (
    decide_fire, evaluate_fire_temperature, solve_time_equivalence_iso834, solve_protection_thickness, teq_main,
)
from .inputs import EXAMPLE_INPUT
from ..mcs import MCSSingle, MCS


class MCS0Single(MCSSingle):
    OUTPUT_KEYS = (
        'index', 'beam_position_horizontal', 'fire_combustion_efficiency', 'fire_hrr_density', 'fire_nft_limit',
        'fire_spread_speed', 'window_open_fraction', 'fire_load_density', 'fire_type', 't1', 't2', 't3',
        'solver_steel_temperature_solved', 'solver_time_critical_temp_solved', 'solver_protection_thickness',
        'solver_iter_count', 'solver_time_equivalence_solved', 'timber_charring_rate', 'timber_exposed_duration',
        'timber_solver_iter_count', 'timber_fire_load', 'timber_charred_depth', 'timber_charred_mass',
        'timber_charred_volume',
    )

    def __init__(self, name, n_simulations, sim_kwargs, save_dir):
        super().__init__(name=name, n_simulations=n_simulations, sim_kwargs=sim_kwargs, save_dir=save_dir)

    @property
    def worker(self) -> Callable:
        return teq_main

    def get_pdf(self, bin_width: float = 0.2) -> (np.ndarray, np.ndarray, np.ndarray):
        teq: np.ndarray = None
        for i in range(len(self.output_keys)):
            if self.output_keys[i] == 'solver_time_equivalence_solved':
                teq = self.output[:, i]
        return MCS0Single.make_pdf(teq, bin_width=bin_width)

    def get_cdf(self, bin_width: float = 0.2):
        x, y_pdf = self.get_pdf(bin_width=bin_width)
        return x, np.cumsum(y_pdf)

    @property
    def output_keys(self) -> tuple:
        return MCS0Single.OUTPUT_KEYS


class MCS0(MCS):
    def __getitem__(self, item) -> MCS0Single:
        return self.mcs_cases[item]

    @property
    def new_mcs_case(self):
        return MCS0Single


def cli_main(fp_mcs_in: str, n_threads: int = 1):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS0()
    mcs.set_inputs_file_path(fp_mcs_in)
    mcs.run(n_proc=n_threads)
    mcs.save_all(True)
