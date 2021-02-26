import os
import warnings

from sfeprapy.mcs0.mcs0_calc import MCS0

warnings.filterwarnings("ignore")


def main(fp_mcs_in: str, n_threads: int = None):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS0()
    try:
        mcs.inputs = fp_mcs_in
    except Exception as e:
        raise e

    try:
        if n_threads:
            mcs.mcs_config = (
                dict(n_threads=n_threads)
                if mcs.mcs_config
                else mcs.mcs_config.update(dict(n_threads=n_threads))
            )
    except KeyError:
        pass

    mcs.run_mcs()


if __name__ == "__main__":
    print("Use `sfeprapy` CLI, `sfeprapy.mcs0:__main__` is depreciated on 22 Oct 2019.")
