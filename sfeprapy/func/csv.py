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
