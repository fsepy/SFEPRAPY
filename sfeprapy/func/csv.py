import csv
from typing import List


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
            row_dict = {header: value for header, value in zip(headers, row)}
            result.append(row_dict)

    return result


def csv_to_dict_of_lists(file_path):
    """
    Convert a CSV file to a dictionary of lists.

    Each column header in the CSV file becomes a key in the dictionary,
    and the items in that column become the values in the list.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    dict: A dictionary where keys are column headers and values are lists of column data.
    """
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        result = {}

        # Initialize lists for each header
        for header in reader.fieldnames:
            result[header] = []

        # Populate the lists with column data
        for row in reader:
            for header in reader.fieldnames:
                result[header].append(row[header])

        return result


def dict_of_ndarray_to_csv(fp: str, data: dict):
    # Assuming all arrays have the same length
    length = len(next(iter(data.values())))

    # Save dictionary to CSV
    with open(fp, 'w+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers
        writer.writerow(data.keys())

        # Write data
        for i in range(length):
            writer.writerow([data[key][i] for key in data])
