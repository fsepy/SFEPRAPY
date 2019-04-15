# -*- coding: utf-8 -*-
import matplotlib
from sfeprapy.mc1.mc_objs_gen2 import MonteCarlo

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os


matplotlib.use('agg')


def plot_dist(id_, df_input, path_input_file, headers):

    names = {'WINDOW OPEN FRACTION []': 'Ao',
             'FIRE LOAD DENSITY [MJ/m2]': 'qfd',
             'FIRE SPREAD SPEED [m/s]': 'spread',
             'BEAM POSITION [m]': 'beam_loc',
             'MAX. NEAR FIELD TEMPERATURE [C]': 'nft'}

    fig, ax = plt.subplots(figsize=(3.94, 2.76))  # (3.94, 2.76) for large and (2.5, 2) for small figure size

    for k, v in names.items():
        x = np.array(df_input[k].values, float)

        if k == 'MAX. NEAR FIELD TEMPERATURE [C]':
            x = x[x < 1200]

        sns.distplot(x, kde=False, rug=True, bins=50, ax=ax, norm_hist=True)

        # Normal plot parameters
        ax.set_ylabel('PDF')
        ax.set_xlabel(k)

        # Small simple plot parameters
        # ax.set_ylabel('')
        # ax.set_yticklabels([])
        # ax.set_xlabel('')

        plt.tight_layout()
        plt.savefig(
            os.path.join(os.path.dirname(path_input_file), '{} - dist - {}.png'.format(id_, v)),
            transparent=True,
            ppi=300
        )
        plt.cla()

    plt.clf()


def run3():
    MC = MonteCarlo()
    MC.select_input_file()
    MC.make_input_param()
    MC.make_mc_params()
    MC.run_mc()
    MC.plot_teq()


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    run3()
