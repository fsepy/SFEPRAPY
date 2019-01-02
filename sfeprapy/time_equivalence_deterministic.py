# coding: utf-8
"""
SCRIPT NAME:    misc_deterministic.py
AUTHOR:         YAN FU
DATE:           30 July 2018
DESCRIPTION:    This code performs time equivalence analysis without probabilistic component.
"""

# DEPENDENCIES
import sys
import copy
import os
import json
import numpy as np
from sfeprapy import time_equivalence_mc as func_core
from sfeprapy.func.temperature_fires import standard_fire_iso834 as _fire_standard
from sfeprapy.dat.steel_carbon import Thermal


def main(dir_wd_list=list()):

    # PARSE INPUT FILES DIRECTORY
    # ===========================

    # if no function parameters
    if len(sys.argv) > 1 > len(dir_wd_list):
        dir_wd_list = copy.copy(sys.argv)
        del dir_wd_list[0]

    else:
        while True:
            project_full_path = input("Work directory (blank to finish): ")
            if len(project_full_path) == 0:
                break
            else:
                dir_wd_list.append(project_full_path)

    # exit function if no work directory obtained
    if len(dir_wd_list) < 1:
        raise ValueError("Aborted. No directories obtained.")

    # otherwise, transform path to absolute real path
    else:
        dir_wd_list = [os.path.abspath(os.path.realpath(i)) for i in dir_wd_list]

    # PARSE INPUT FILE PATHS
    # ======================
    files_directories = []
    for d in dir_wd_list:

        list_files = os.listdir(d)
        list_input_files = []
        for f in list_files:
            if f.endswith('.json'):
                list_input_files.append(os.path.join(d, f))

        files_directories.append(list_input_files)

    # TIME EQUIVALENCE CALCULATION
    # ============================

    # "iso834_time": 0,
    # "iso834_temperature": 0,
    steel_prop = Thermal()
    beam_c = steel_prop.c()
    fire = _fire_standard(np.arange(0, 3*60*60, 1), 273.15+20)
    # iterate through all input work directories
    print(files_directories)
    dirs_calcs = []
    for d in files_directories:

        # iterate all input files in the work directory
        calcs = []
        for f in d:
            with open(str(f), "r") as i:
                kwargs = json.load(i)
                kwargs["iso834_time"] = fire[0]
                kwargs["iso834_temperature"] = fire[1]
                kwargs["beam_c"] = beam_c
                kwargs["is_return_dict"] = True

            c = func_core.calc_time_equivalence(**kwargs)

            calcs.append(func_core.calc_time_equivalence(**kwargs))

            with open(f + ".out.json", 'w') as f_:
                json.dump(c, f_, indent=True)

            print('-'*40)
            print("{:<30}{:<10}".format("Key", 'Value'))
            print('-'*40)
            for k, v in c.items():
                print("{:<30}{:<10}".format(k, v))
            print('-'*40)

        dirs_calcs.append(calcs)

    # func_core.calculation_time_equivalence()

    # WRITE RESULTS
    # =============

    return 0


if __name__ == '__main__':
    main([r"C:\Users\IanFu\Desktop\aaaaa"])
    # main()
