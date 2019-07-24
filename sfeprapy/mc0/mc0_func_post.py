# -*- coding: utf-8 -*-
import os
import pandas as pd


def concatenate_dataframe_header(header_name:str, **dict_df:pd.DataFrame):

    dict_res = dict()

    for k, df in dict_df.items():
        dict_res[k] = df[header_name].values

    return pd.DataFrame.from_dict(dict_res)


def concatenate_dataframe_header2(*header_names:str, **dict_df:pd.DataFrame):

    dict_res = dict()

    for header_name in header_names:
        dict_res[header_name] = []

    for k, df in dict_df.items():
        for header_name in header_names:
            dict_res[header_name] += list(df[header_name].values)

    return pd.DataFrame.from_dict(dict_res)


def extract_results(*list_path_csv:str):
    # get out.csv path

    dict_df = dict()

    for path_csv in list_path_csv:
        k = os.path.basename(path_csv).replace('_out.csv', '')
        df = pd.DataFrame.from_csv(path_csv)
        dict_df[k] = df

    dict_df_out = dict()
    dict_df_out['solver_time_equivalence_solved'] = concatenate_dataframe_header('solver_time_equivalence_solved', **dict_df)
    dict_df_out['solver_time_equivalence_solved_merged'] = concatenate_dataframe_header2('solver_time_equivalence_solved', 'probability_weight', **dict_df)

    return dict_df_out
