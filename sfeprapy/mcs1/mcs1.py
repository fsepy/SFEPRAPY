# -*- coding: utf-8 -*-
import matplotlib

from sfeprapy.mcs1.mcs1_obj import MonteCarlo

matplotlib.use("agg")


def main(path_master_csv: str = None):
    MC = MonteCarlo()
    MC.select_input_file(path_master_csv)
    MC.make_input_param()
    MC.make_mc_params()
    MC.run_mc()
    MC.out_combined_ky()
    MC.out_combined_T()


if __name__ == "__main__":
    main()
