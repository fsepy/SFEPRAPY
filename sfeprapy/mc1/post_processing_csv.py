# -*- coding: utf-8 -*-
import os
from tkinter import Tk, StringVar, filedialog
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def tk_select_path():

    root = Tk()
    root.withdraw()
    folder_path = StringVar()

    path_folder = filedialog.askdirectory(title='Select Input Files')
    folder_path.set(path_folder)
    root.update()

    return path_folder


def get_list_file_path_with_suffix(path_folder, suffix):

    list_path_out_csv_all = []

    for f in os.listdir(path_folder):
        if f.endswith(suffix):
            list_path_out_csv_all.append(os.path.join(path_folder, f))

    return list_path_out_csv_all


def read_csvs_to_dfs(list_path_csv_file):

    list_pd_out_csv = []

    for path_out_csv in list_path_csv_file:
        df_ = pd.read_csv(path_out_csv, index_col='INDEX', dtype=np.float64)

        list_pd_out_csv.append(df_)

    return list_pd_out_csv


def get_teq_for_fr(list_df_out, list_fire_resistance):

    list_teq = []

    for df in list_df_out:

        # obtain x and y values for plot

        mask_teq_sort = np.asarray(df["TIME EQUIVALENCE [s]"].values).argsort()

        x = df["TIME EQUIVALENCE [s]"].values[mask_teq_sort] / 60.  # to minutes
        y = np.arange(1, len(x) + 1) / len(x)

        f_interp = interp1d(x, y, bounds_error=False, fill_value=(min(y), max(y)))

        list_teq.append(f_interp(list_fire_resistance))

    return list_teq


def main():

    path_folder = tk_select_path()

    list_path_files = get_list_file_path_with_suffix(path_folder, '_out.csv')

    list_df = read_csvs_to_dfs(list_path_files)

    list_teq = get_teq_for_fr(list_df, [30, 60, 90, 120, 150, 180])

    fmt = '{:<20.18},{:10.6f},{:10.6f},{:10.6f},{:10.6f},{:10.6f},{:10.6f}'

    for i, path_file in enumerate(list_path_files):

        file_name = os.path.basename(path_file).replace('_out.csv', '')

        print(fmt.format(file_name, *list_teq[i]))


if __name__ == '__main__':

    main()
