import csv

from sfeprapy.func.xlsx import xlsx_to_dict
from sfeprapy.mcs import InputParser


def dict_to_csv(csv_filename, data_dict):
    # Get the headers (column names) from the dictionary keys
    headers = list(data_dict.keys())

    # Check if the dictionary is empty or has no data
    # Assume all value lists have the same length to determine the number of rows
    num_rows = len(data_dict[headers[0]])

    # Verify that all lists have the same length (optional but recommended)
    for header in headers:
        if len(data_dict[header]) != num_rows:
            raise ValueError(
                f"Inconsistent list lengths found for header '{header}'. All lists must have the same length.")

    # Open the CSV file in write mode ('w')
    # newline='' prevents extra blank rows on Windows
    # encoding='utf-8' is generally recommended for compatibility
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(headers)

        # Write the data rows
        for i in range(num_rows):
            # Construct the row for the current index 'i'
            # This list comprehension iterates through the headers
            # and picks the i-th element from the corresponding list in the dictionary
            row_data = [data_dict[header][i] for header in headers]
            writer.writerow(row_data)

    print(f"Dictionary successfully saved to '{csv_filename}'")

    # except FileNotFoundError:
    #     print(f"Error: Could not write to file. Check path permissions for '{csv_filename}'.")
    # except KeyError as e:
    #     print(f"Error: Key '{e}' not found. This shouldn't happen if headers are derived from keys.")
    # except ValueError as e:
    #     print(f"Error: {e}")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    _ = xlsx_to_dict(r'C:\Users\ian\Desktop\new_sfeprapy\test.xlsx')
    print(_)
    __, __2 = InputParser(_['CASE_1'], 100).to_dict2()
    print(__)
    dict_to_csv(r'C:\Users\ian\Desktop\new_sfeprapy\test2.csv', __)

    import json

    with open(r'C:\Users\ian\Desktop\new_sfeprapy\test2.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(__2, jsonfile, ensure_ascii=False, indent=4)
