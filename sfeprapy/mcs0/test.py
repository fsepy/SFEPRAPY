from sfeprapy.mcs.dist import DistSampler
from sfeprapy.mcs0.inputs import EXAMPLE_INPUT
from sfeprapy.mcs0 import MCS0Case


if __name__ == '__main__':
    data_user = EXAMPLE_INPUT.copy()
    data_sampled = dict()
    for case_name, case_data in data_user.items():
        kwargs_from_input = DistSampler(case_data, case_data['n_simulations']).to_dict(suppress_constant=True)
        res = MCS0Case.process_mcs_output(MCS0Case.run_mcs(kwargs_from_input))
        print(res)
