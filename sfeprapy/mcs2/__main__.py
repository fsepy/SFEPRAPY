import os

from sfeprapy.mcs2.mcs2_calc import MCS2


def main(fp_mcs_in: str, n_threads: int = None):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS2()
    try:
        mcs.mcs_inputs = fp_mcs_in
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
