# -*- coding: utf-8 -*-
import sfeprapy.mcs0
from sfeprapy.mcs0 import __example_input_csv, __example_input_df, EXAMPLE_CONFIG_DICT


def __example_input_dict():
    inputs_old = sfeprapy.mcs0.__example_input_dict()
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
    # return {"Standard Case 3 (with timber)": inputs["Standard Case 3 (with timber)"]}
    return inputs_new


EXAMPLE_INPUT_DICT = __example_input_dict()
EXAMPLE_INPUT_CSV = __example_input_csv(__example_input_dict())
EXAMPLE_INPUT_DF = __example_input_df(__example_input_dict())

if __name__ == "__main__":
    print(EXAMPLE_CONFIG_DICT, "\n")
    print(EXAMPLE_INPUT_DICT, "\n")
    print(EXAMPLE_INPUT_CSV, "\n")
