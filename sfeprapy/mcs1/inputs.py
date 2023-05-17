from ..mcs0 import EXAMPLE_INPUT as __EXAMPLE_INPUT

EXAMPLE_INPUT = {'CASE_1': __EXAMPLE_INPUT['CASE_1'].copy()}
for i in (
        'p1', 'p2', 'p3', 'p4', 'solver_temperature_goal', 'solver_max_iter', 'solver_thickness_lbound',
        'solver_thickness_ubound', 'solver_tol'
):
    EXAMPLE_INPUT['CASE_1'].pop(i)
EXAMPLE_INPUT['CASE_1']['epsilon_q'] = dict(ubound=1 - 1e-9, lbound=1e-9, dist='uniform_')
EXAMPLE_INPUT['CASE_1']['t_k_y_theta'] = 5 * 60.
EXAMPLE_INPUT['CASE_1']['protection_d_p'] = 0.01

if __name__ == "__main__":
    import pprint

    pprint.pprint(EXAMPLE_INPUT)
