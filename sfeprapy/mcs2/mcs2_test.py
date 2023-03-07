import copy

import numpy as np
from scipy.interpolate import interp1d

from sfeprapy.mcs2 import EXAMPLE_INPUT_DICT
from sfeprapy.mcs2 import MCS2


def test_standard_case():
    # increase the number of simulations so it gives sensible results
    mcs_input = copy.deepcopy(EXAMPLE_INPUT_DICT)
    mcs_input.pop('Residential')
    mcs_input.pop('Retail')
    for k in list(mcs_input.keys()):
        mcs_input[k]['n_simulations'] = 50_000

    # increase the number of threads so it runs faster
    mcs2 = MCS2()
    mcs2.inputs = mcs_input
    mcs2.n_threads = 1
    mcs2.run(cases_to_run=['Office'], keep_results=True)
    mcs_out = mcs2.outputs
    teq = mcs_out.loc[mcs_out['case_name'] == 'Office']["solver_time_equivalence_solved"] / 60.0
    teq = teq[~np.isnan(teq)]
    hist, edges = np.histogram(teq, bins=np.arange(0, 181, 0.05))
    x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    func_teq = interp1d(x, y)
    for fire_rating in [30, 45, 60, 75, 90, 105, 120]:
        print(f'{fire_rating:<8.0f}  {func_teq(fire_rating):<.8f}')

    # assert abs(func_teq(30) - 0.07519936) <= 5e-3
    # assert abs(func_teq(60) - 0.65458147) <= 5e-3
    # assert abs(func_teq(90) - 0.93229350) <= 5e-3
    assert abs(func_teq(30) - 0.08871437) <= 5e-3
    assert abs(func_teq(60) - 0.65500191) <= 5e-3
    assert abs(func_teq(90) - 0.92701250) <= 5e-3


if __name__ == '__main__':
    test_standard_case()
