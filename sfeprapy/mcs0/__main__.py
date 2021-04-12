import os
import warnings

from sfeprapy.mcs0.mcs0_calc import MCS0

warnings.filterwarnings("ignore")


def main(fp_mcs_in: str, n_threads: int = 1):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS0()
    mcs.inputs = fp_mcs_in
    mcs.n_threads = n_threads
    mcs.run_mcs()


if __name__ == "__main__":
    print("Use `sfeprapy` CLI, `sfeprapy.mcs0:__main__` is depreciated on 22 Oct 2019.")
