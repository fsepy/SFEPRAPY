# -*- coding: utf-8 -*-
from sfeprapy.mcs0 import __example_input_csv, __example_input_df


def __example_input_dict() -> dict:
    from sfeprapy.mcs0 import EXAMPLE_INPUT_DICT
    inputs_old = EXAMPLE_INPUT_DICT
    inputs_new = dict()
    for k, v in inputs_old.items():
        v.pop('timber_exposed_area')
        if k == "Standard Case 3 (with timber)":
            v["timber_Q_crit"] = 12.6
            v["timber_total_exposed_area"] = 500.
            v["timber_exposed_breadth"] = 16.
        else:
            v["timber_Q_crit"] = 0.
            v["timber_total_exposed_area"] = 0.
            v["timber_exposed_breadth"] = 0.

        inputs_new[k] = v
    return inputs_new


EXAMPLE_INPUT_DICT = __example_input_dict()
EXAMPLE_INPUT_CSV = __example_input_csv(list(EXAMPLE_INPUT_DICT.values()))
EXAMPLE_INPUT_DF = __example_input_df(list(EXAMPLE_INPUT_DICT.values()))

if __name__ == "__main__":
    print(EXAMPLE_INPUT_DICT, "\n")
    print(EXAMPLE_INPUT_CSV, "\n")
    print(EXAMPLE_INPUT_DF, "\n")
