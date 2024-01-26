from abc import ABC, abstractmethod
from typing import Tuple

from .dist import TrueToScipy

import numpy as np

class MCSCase(ABC):
    """Describes a single SFEPRAPY case that packs the entry function and necessary tools for ease of use."""

    @staticmethod
    @abstractmethod
    def get_input_keys() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Returns two tuples, first is the required parameter list and second is the optional parameter list"""
        pass

    @staticmethod
    @abstractmethod
    def get_output_keys() -> Tuple[str, ...]:
        """Returns a list of output parameter names"""
        pass

    @staticmethod
    @abstractmethod
    def main_func(*args, **kwargs) -> tuple:
        """The entry function, single iteration of SFEPRAPY calculation"""
        pass

    @classmethod
    def run_mcs(cls, data: dict):
        kwargs_valid = cls.get_input_keys()
        kwargs_valid = kwargs_valid[0] + kwargs_valid[1]

        kwargs_static = dict()
        kwargs_stochastic = dict()
        for k, v in data.items():
            if k not in kwargs_valid:
                continue
            if isinstance(v, (int, float, str)):
                kwargs_static[k] = v
            elif isinstance(v, bytes):
                kwargs_static[k] = v.decode('utf-8')
            else:
                kwargs_stochastic[k] = v

        kwargs_stochastic = [dict(zip(kwargs_stochastic, t)) for t in zip(*kwargs_stochastic.values())]
        for kwargs in kwargs_stochastic:
            kwargs_ = {**kwargs_static, **kwargs}
            yield cls.main_func(**kwargs_)

    @classmethod
    def process_mcs_output(cls, data: list):
        data = np.array(tuple(data))
        assert len(data[0]) == len(cls.get_output_keys())
        return {k: data[:, i] for i, k in enumerate(cls.get_output_keys())}
