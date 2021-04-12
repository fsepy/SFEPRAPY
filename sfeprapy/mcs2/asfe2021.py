from sfeprapy.mcs2 import MCS2
from copy import deepcopy
from scipy.interpolate import interp1d
import numpy as np


class ASFE2021:
    def step_1_benchmark_against_kirby(self):
        fp_inputs = r'D:\projects_fse\!ASFE2021\01 analysis\trial_00\0-trial_00.xlsx'
        mcs2 = MCS2()
        mcs2.inputs = fp_inputs
        mcs2.n_threads = 1
        mcs2.run_mcs()

    def step_2_failure_probability(self):
        pass


if __name__ == '__main__':
    asfe21 = ASFE2021()
    asfe21.step_1_benchmark_against_kirby()
