import csv
import json
from os import path, makedirs, remove
from typing import List

import tqdm

from sfeprapy import logger
from sfeprapy.mcs0.calcs import teq_main

EXAMPLE_INPUT_JSON = ''
EXAMPLE_INPUT_CSV = ''


def csv_to_list_of_dicts(fp: str) -> List[dict]:
    """This is used to read csv file containing measurement data exported from Bluebeam.

    :param fp:
    :return:
    """
    with open(fp, newline='', encoding='utf-8-sig') as csvfile:
        # Read the CSV file
        reader = csv.reader(csvfile)

        # Get the headers (first row)
        headers = next(reader)

        # Initialize the result list
        result = []

        # Iterate through the CSV file and create the list of dictionaries
        for row in reader:
            row_dict = {header: float(value) for header, value in zip(headers, row)}
            result.append(row_dict)

    return result


def teq_main_chunk(kwargs_group: List[dict]):
    res = list()
    for kwargs in kwargs_group:
        res.append(teq_main(**kwargs))
    return res


def main_single(work_dir: str):
    case_name = path.basename(work_dir)

    # =========
    # read json
    # =========
    fp_json = path.join(work_dir, f'{case_name}.json')
    try:
        with open(fp_json, 'r', encoding='utf-8') as f:
            kwargs = json.load(f)
        for k in kwargs.keys():
            if isinstance(kwargs[k], (int, float)):
                kwargs[k] = float(kwargs[k])
    except Exception:
        raise FileNotFoundError(f'Error when reading JSON')
    n_simulations = kwargs.pop('n_simulations')

    # ========
    # read csv
    # ========
    fp_csv = path.join(work_dir, f'{case_name}.csv')
    kwargs_list = list()  # Initialize the result list
    try:
        with open(fp_csv, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)  # Read the CSV file
            headers = next(reader)  # Get the headers (first row)
            # Iterate through the CSV file and create the list of dictionaries
            for row in reader:
                row_dict = {header: float(value) for header, value in zip(headers, row)}
                kwargs_list.append(row_dict)
    except Exception:
        raise FileNotFoundError(f'Error when reading CSV')

    # ===============
    # validate inputs
    # ===============
    try:
        assert n_simulations == len(kwargs_list)
    except AssertionError:
        raise ValueError('Error when validating input data: Data length does not match n_simulations')

    # ============
    # run analysis
    # ============
    res = list()
    chunk_size = 800
    for i in tqdm.tqdm(range(0, int(n_simulations), chunk_size)):
        if i + chunk_size > n_simulations:
            kwargs_list_ = kwargs_list[i:]
        else:
            kwargs_list_ = kwargs_list[i: i + chunk_size]
        res.extend(teq_main_chunk([{**kwargs, **kwargs_} for kwargs_ in kwargs_list_]))

    # =========
    # save data
    # =========
    fp_out = path.join(work_dir, f'{path.basename(work_dir)}-out.csv')
    header = (
        'index', 'fire_type', 't1', 't2', 't3',
        'solver_status', 'solver_steel_temperature_solved', 'solver_time_critical_temp_solved',
        'solver_protection_thickness', 'solver_iter_count', 'solver_time_equivalence_solved', 'timber_charring_rate',
        'timber_exposed_duration', 'timber_solver_iter_count', 'timber_fire_load', 'timber_charred_depth',
        'timber_charred_mass', 'timber_charred_volume',
    )
    if res:
        if len(header) != len(res[0]):
            logger.warning(f'Header length {len(header)} does not match output length{len(res)}')
    with open(fp_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(res)


def split_csv(fp_csv, dir_out=None, records_per_file=10000, fn_output_template="~{:04d}.csv"):
    """
    Splits a large CSV file into multiple smaller CSV files.

    Args:
        fp_csv (str): Path to the input CSV file.
        dir_out (str): Path to the directory where the output files will be saved.
        records_per_file (int, optional): Maximum number of records (rows, excluding header)
            in each output file. Defaults to 10000.
        fn_output_template (str, optional):  Template for the output file names.
            Must contain a single integer formatting placeholder (e.g., "{:04d}" for
            four-digit zero-padded numbering).  Defaults to "part_{:04d}.csv".
    """
    if not path.exists(fp_csv):
        raise FileNotFoundError(f"Input file not found: {fp_csv}")
    if not dir_out:
        dir_out = path.dirname(fp_csv)
    if not path.isdir(dir_out):
        makedirs(dir_out)  # Create the output directory if it doesn't exist
    if records_per_file <= 0:
        raise ValueError("records_per_file must be a positive integer.")

    file_number = 1
    # Construct the initial output path using the template
    try:
        output_path = path.join(dir_out, fn_output_template.format(file_number))
    except ValueError:
        raise ValueError(
            f"Invalid fn_output_template: '{fn_output_template}'. Ensure it contains one formatting placeholder like {{:04d}}.")

    try:
        with open(fp_csv, 'r', encoding='utf-8',
                  newline='') as infile:  # Added newline='' for input reading consistency
            reader = csv.reader(infile)

            try:
                header = next(reader)  # Read the header row
            except StopIteration:
                print(f"Warning: Input file '{fp_csv}' is empty or contains only a header.")
                return  # Nothing to split if no data rows

            # Open the first output file
            outfile = open(output_path, 'w', newline='', encoding='utf-8')
            writer = csv.writer(outfile)
            writer.writerow(header)
            last_opened_path = output_path  # Track the first opened file
            records_written = 0

            for row in reader:
                writer.writerow(row)
                records_written += 1

                if records_written >= records_per_file:
                    outfile.close()  # Close the completed file
                    file_number += 1
                    output_path = path.join(dir_out, fn_output_template.format(file_number))

                    # Open the next file and write header
                    outfile = open(output_path, 'w', newline='', encoding='utf-8')
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    last_opened_path = output_path  # Track the newly opened file
                    records_written = 0  # Reset counter for the new file

        # --- After the loop ---
        if outfile:  # Ensure the last file is closed
            outfile.close()

        # --- Fix for empty last file ---
        # Check if the last action was resetting records_written, meaning the last opened file is empty
        if records_written == 0 and file_number > 1:
            # Check if the file actually exists before trying to remove it
            if last_opened_path and path.exists(last_opened_path):
                remove(last_opened_path)
                file_number -= 1  # Decrement count as we removed the empty file
    finally:
        if outfile and not outfile.closed:
            try:
                outfile.close()
            except Exception as close_e:
                logger.debug(f"Error closing output file during unexpected error handling: {close_e}")


if __name__ == '__main__':
    main_single(r'C:\Users\ian\Desktop\new_sfeprapy\case_1')
    # split_csv(r'C:\Users\ian\Desktop\new_sfeprapy\new_case\new_case.csv', records_per_file=1001)
