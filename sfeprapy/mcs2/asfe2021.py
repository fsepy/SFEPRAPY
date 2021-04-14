from sfeprapy.mcs2 import MCS2
import numpy as np
from scipy.interpolate import interp1d


class ASFE2021:
    def step_1_benchmark_against_kirby(self):
        fp_inputs = r'D:\projects_fse\!ASFE2021\01 analysis\trial_00\0-trial_00.xlsx'
        cases_to_run=['Office', 'Residential', 'Retail']
        mcs2 = MCS2()
        mcs2.inputs = fp_inputs
        mcs2.n_threads = 6
        mcs2.run_mcs(cases_to_run=cases_to_run)

        mcs_out = mcs2.mcs_out
        for case in cases_to_run:
            print(case)
            self.print_teq_cfd(mcs_out.loc[mcs_out['case_name'] == case]["solver_time_equivalence_solved"] / 60.0)

    def step_2_failure_probability(self):
        pass

    @staticmethod
    def print_teq_cfd(teq):
        hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.5))
        x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
        func_teq = interp1d(x, y)
        for fire_rating in [30, 45, 60, 75, 90, 105, 120, 150, 135, 180]:
            print(f'{fire_rating:<4.0f}  {func_teq(fire_rating):<.4f}')


if __name__ == '__main__':
    asfe21 = ASFE2021()
    asfe21.step_1_benchmark_against_kirby()
