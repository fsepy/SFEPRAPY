import os

from sfeprapy.mcs3.mcs3_calc import MCS3


def main(fp_mcs_in: str, n_threads: int = 1):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS3()
    mcs.inputs = fp_mcs_in
    mcs.n_threads = n_threads
    mcs.run_mcs()
