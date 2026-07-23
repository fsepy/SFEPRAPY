"""sfeprapy - structural fire time-equivalence (next-gen, pure function).

The public surface is the deterministic calculation routine ``teq_main`` plus its
supporting helpers and the ``EXAMPLE_INPUT`` cases. Pass it sampled parameters,
get a ``TeqResult`` back.
"""

import logging

logger = logging.getLogger('sfeprapy')
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(
        logging.Formatter(fmt='{asctime} {levelname:8.8s} [{filename:15.15s}:{lineno:05d}] {message:s}',
                          style='{'))
    logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)

__all__ = (
    'EXAMPLE_INPUT',
    'decide_fire', 'evaluate_fire_temperature', 'solve_time_equivalence_iso834',
    'solve_protection_thickness', 'teq_main', 'TeqResult',
    '__version__',
)

from ._version import __version__
from .calcs import (
    decide_fire, evaluate_fire_temperature, solve_time_equivalence_iso834, solve_protection_thickness,
    teq_main, TeqResult,
)
from .inputs import EXAMPLE_INPUT
