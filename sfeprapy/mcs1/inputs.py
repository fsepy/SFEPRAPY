import copy as __copy

from ..mcs0 import EXAMPLE_INPUT as EXAMPLE_INPUT_DICT_
from ..mcs0.inputs import example_input_dict


def __example_input_list() -> list:
    # Create base case from `sfeprapy.mcs0`
    # only use "CASE_1"
    base_case = __copy.deepcopy(EXAMPLE_INPUT_DICT_['CASE_1'])
    # remove items which are no longer used in `sfeprapy.mcs2` (comparing to `sfeprapy.mcs0`)
    for i in [
        'p1', 'p2', 'p3', 'p4', 'solver_temperature_goal', 'solver_max_iter', 'solver_thickness_lbound',
        'solver_thickness_ubound', 'solver_tol'
    ]:
        base_case.pop(i)

    base_case['epsilon_q'] = dict(ubound=1 - 1e-9, lbound=1e-9, dist='uniform_')
    base_case['t_k_y_theta'] = 5 * 60

    # create variable for dumping new inputs
    y = list()

    y.append(__copy.copy(base_case))
    y[-1].update(dict(case_name='CASE_1', protection_d_p=0.01, phi_teq=1))

    return y


EXAMPLE_INPUT = example_input_dict(__example_input_list())

if __name__ == "__main__":
    import pprint

    pprint.pprint(EXAMPLE_INPUT)
