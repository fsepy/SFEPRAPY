import concurrent.futures
import csv
import json
import multiprocessing as mp
import pathlib
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm

from .calcs import teq_main as calcs_main
from .. import logger


def calculation_function(params: Dict[str, Any]) -> Tuple:
    """
    Call the calculation function with appropriate parameters.

    Args:
        params: Dictionary of parameters (both stochastic and static)

    Returns:
        Tuple containing (index, result1, result2)
    """
    # Create a copy to avoid modifying the input dictionary
    params_copy = deepcopy(params)

    # Extract the index before passing to calculation function
    index = params_copy.pop('index', -1)

    # Call the main calculation function and return results with index
    try:
        return (index, *calcs_main(**params_copy))
    except Exception as e:
        logger.error(f"Calculation error with params {params_copy}: {str(e)}")
        raise


def process_iteration(task_data: Dict[str, Any]) -> Tuple:
    """
    Process a single row of data with shared parameters.

    Args:
        task_data: Dictionary containing combined parameters

    Returns:
        Tuple containing calculation results
    """
    try:
        # Call the calculation function
        return calculation_function(task_data)
    except Exception as e:
        row_index = task_data.get('index', 'unknown')
        logger.error(f"Error processing row {row_index}: {str(e)}")
        # Return None values for results with the correct tuple length
        # Adjust the number of None values to match your actual result structure
        return (row_index, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def convert_numeric(value: str) -> Any:
    """
    Convert string value to int or float if possible.

    Args:
        value: String value to convert

    Returns:
        Converted value (int, float, or original string)
    """
    if not isinstance(value, str):
        return value

    value = value.strip()

    # Check for empty strings or null values
    if not value or value.lower() in ('null', 'nan', 'none'):
        return None

    try:
        # Try to convert to integer
        return int(value)
    except ValueError:
        try:
            # Try float conversion
            return float(value)
        except ValueError:
            # Return original if conversion fails
            return value


def read_input_files(csv_path: pathlib.Path, json_path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Read and combine input files into a list of parameter dictionaries.

    Args:
        csv_path: Path to CSV file with stochastic parameters
        json_path: Path to JSON file with static parameters

    Returns:
        List of combined parameter dictionaries
    """
    # Read static parameters
    try:
        with open(json_path, 'r') as jsonfile:
            static_params = json.load(jsonfile)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading static parameters: {str(e)}")
        raise

    # Read stochastic parameters
    stochastic_params = []
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Validate required columns
            required_cols = ['index']  # Add your essential columns here
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]

            if missing_cols:
                logger.error(f"Missing required columns in CSV: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Process each row
            for i, row in enumerate(reader):
                # Add row number as index if not present
                if 'index' not in row or not row['index']:
                    row['index'] = i

                # Convert string values to appropriate types
                numeric_row = {field: convert_numeric(value) for field, value in row.items()}
                stochastic_params.append(numeric_row)
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {str(e)}")
        raise

    # Combine parameters for processing
    combined_params = [dict(**row, **static_params) for row in stochastic_params]
    return combined_params


def write_results_to_csv(results: List[Tuple], output_path: pathlib.Path, column_names: List[str]) -> None:
    """
    Write calculation results to CSV file.

    Args:
        results: List of result tuples
        output_path: Path to output CSV file
        column_names: Names of columns for CSV header
    """
    # Sort results by index (first element of each tuple)
    sorted_results = sorted(results, key=lambda x: x[0])

    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)  # Write header

            # Write data rows with progress bar
            for result in sorted_results:
                formatted_result = list()
                for value in result:
                    try:
                        # Format the float to the specified number of decimal places
                        # f-string formatting: :.nf where n is decimal_places
                        formatted_value = f"{value:.{5}f}"
                        formatted_result.append(formatted_value)
                    except ValueError:
                        # In case formatting fails for some unexpected reason
                        # Append the original value and let csv.writer handle it
                        formatted_result.append(value)
                writer.writerow(result)
    except IOError as e:
        logger.error(f"Error writing results to CSV: {str(e)}")
        raise


def process_single_case(case_dir: pathlib.Path, executor: concurrent.futures.ProcessPoolExecutor) -> None:
    """
    Process a single simulation case using the provided executor.

    Args:
        case_dir: Directory containing simulation case files
        executor: ProcessPoolExecutor to use for parallel processing
    """
    case_name = case_dir.name
    logger.info(f"Processing simulation case: {case_name}")

    # Define file paths
    fp_in_stochastic = case_dir / f'{case_name}.csv'
    fp_in_static = case_dir / f'{case_name}.json'
    fp_out = case_dir / f'{case_name}_out.csv'

    # Read input files and combine parameters
    try:
        process_args = read_input_files(fp_in_stochastic, fp_in_static)
    except Exception as e:
        logger.error(f"Failed to read input files for case {case_name}: {str(e)}")
        return

    # Process using the provided ProcessPoolExecutor with batching for large datasets
    batch_size = 10000  # Adjust based on your memory constraints
    results = []

    # Process in batches to avoid memory issues
    for i in range(0, len(process_args), batch_size):
        batch = process_args[i:i + batch_size]

        # Submit all tasks in batch and get futures
        futures = [executor.submit(process_iteration, arg | dict(dir_temp=case_dir)) for arg in batch]

        # Process results as they complete with tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc=f"Batch {i // batch_size + 1}", ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in future: {str(e)}")

    # Define result column names - replace with appropriate names
    result_cols = ['index', 'q_inc', 't_ig_ftp', 'ftp', 't_ig_safir', 't_max_safir', 'T_max_safir', 't_d', ]

    # Save results to CSV
    try:
        write_results_to_csv(results, fp_out, result_cols)
    except Exception as e:
        logger.error(f"Failed to write results for case {case_name}: {str(e)}")


def process_multiple_cases(case_dirs: List[pathlib.Path], n_proc: Optional[int] = 0) -> None:
    """
    Process multiple simulation cases with a single process pool.

    Args:
        case_dirs: List of directories containing simulation cases
    """
    # Determine number of processes (leave one core free for the OS)
    num_cores = n_proc or max(1, mp.cpu_count() - 2)
    logger.info(f"Using {num_cores} worker processes")

    # Create a single ProcessPoolExecutor for all cases
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for case_dir in case_dirs:
            if not case_dir.is_dir():
                logger.warning(f"Skipping non-directory: {case_dir}")
                continue

            try:
                # Process each case with the same executor
                process_single_case(case_dir, executor)
            except Exception as e:
                logger.error(f"Failed to process case {case_dir.name}: {str(e)}")


def process_single_case_2(case_dir: pathlib.Path) -> None:
    """Process a single simulation case in its entirety within one process."""
    case_name = case_dir.name

    # Define file paths
    fp_in_stochastic = case_dir / f'{case_name}.csv'
    fp_in_static = case_dir / f'{case_name}.json'
    fp_out = case_dir / f'{case_name}_out.csv'

    try:
        # Read input files and combine parameters
        process_args = read_input_files(fp_in_stochastic, fp_in_static)

        # Process all rows sequentially within this process, without progress tracking
        results = []
        for arg in process_args:
            # Add the directory path to the arguments
            arg_with_dir = arg | dict(dir_temp=case_dir)
            result = process_iteration(arg_with_dir)
            results.append(result)

        # Define result column names - replace with appropriate names
        result_cols = ['index', 'q_inc', 't_ig_ftp', 'ftp', 't_ig_safir', 't_max_safir', 'T_max_safir', 't_d']

        # Save results to CSV
        write_results_to_csv(results, fp_out, result_cols)

    except Exception as e:
        logger.error(f"Failed to process case {case_name}: {str(e)}")


def process_multiple_cases_2(case_dirs: List[pathlib.Path], n_proc: Optional[int] = None) -> None:
    """
    Process multiple simulation cases in parallel with one dedicated process per case.

    Instead of processing rows of a case in parallel, this function processes each
    entire case in its own process.

    Args:
        case_dirs: List of directories containing simulation cases
        n_proc: Maximum number of concurrent processes to use. If None, will use available cores.
    """
    # Determine number of processes (leave some cores free for the OS)
    max_processes = n_proc or max(1, mp.cpu_count() - 2)

    # Filter valid directories
    valid_case_dirs = [case_dir for case_dir in case_dirs if case_dir.is_dir()]

    if not valid_case_dirs:
        logger.warning("No valid case directories found!")
        return

    if n_proc == 1:
        for case_dir in tqdm(valid_case_dirs):
            process_single_case_2(case_dir)
        return

    # Use ProcessPoolExecutor to run each case as a separate process
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        # Submit all cases to the executor
        futures = {executor.submit(process_single_case_2, case_dir): case_dir.name for case_dir in valid_case_dirs}

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            case_name = futures[future]
            try:
                # Get the result (None for the processing function)
                future.result()
            except Exception as e:
                logger.error(f"Error processing case {case_name}: {str(e)}")
