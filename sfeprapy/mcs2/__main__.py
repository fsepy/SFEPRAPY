import os

from sfeprapy.mcs2.mcs2_calc import MCS2


def main(fp_mcs_in: str, n_threads: int = 1):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS2()
    mcs.inputs = fp_mcs_in
    mcs.n_threads = n_threads
    mcs.run_mcs()
