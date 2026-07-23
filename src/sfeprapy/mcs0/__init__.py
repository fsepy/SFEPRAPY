"""sfeprapy.mcs0 - structural fire time-equivalence (next-gen, pure function).

The entire public surface is the deterministic calculation routine ``teq_main`` plus
its supporting helpers and the ``EXAMPLE_INPUT`` cases. There is no Monte Carlo
machinery, no orchestration classes, and no file I/O here -- ``teq_main`` is a plain
function: pass it sampled parameters, get a tuple back.
"""

__all__ = (
    'EXAMPLE_INPUT',
    'decide_fire', 'evaluate_fire_temperature', 'solve_time_equivalence_iso834',
    'solve_protection_thickness', 'teq_main', 'TeqResult',
)

from .calcs import (
    decide_fire, evaluate_fire_temperature, solve_time_equivalence_iso834, solve_protection_thickness,
    teq_main, TeqResult,
)
from .inputs import EXAMPLE_INPUT
