# -*- coding: utf-8 -*-
import csv
import os


def list_to_csv(file_name, a_list):
    with open(file_name, 'wb') as f:
        output_writer = csv.writer(f, dialect='excel')
        output_writer.writerows(a_list)


def list_all_files_with_suffix(directory_str, suffix_str, is_full_dir=True):
    list_files = []

    for f in os.listdir(directory_str):
        if f.endswith(suffix_str):
            if is_full_dir:
                list_files.append(os.path.join(directory_str, f))
            else:
                list_files.append(f)

    return list_files
