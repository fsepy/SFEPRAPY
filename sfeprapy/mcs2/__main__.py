import os

import pandas as pd

from sfeprapy.mcs2.mcs2_calc import MCS2


def main(fp_mcs_in: str, n_threads: int = None):
    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS2()
    try:
        if fp_mcs_in.endswith('.csv'):
            mcs.mcs_inputs = pd.read_csv(fp_mcs_in, index_col=0)
        elif fp_mcs_in.endswith('.xlsx'):
            mcs.mcs_inputs = pd.read_excel(fp_mcs_in, index_col=0)
        else:
            raise TypeError('Unknown file format')
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
