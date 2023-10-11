import csv
import inspect
import json
import shutil
from io import BytesIO
from os import path, mkdir
from typing import Optional, List

import numpy as np
from xlrd import open_workbook

from sfeprapy.func.csv import dict_of_ndarray_to_csv
from sfeprapy.mcs import InputParser
from sfeprapy.mcs0 import teq_main


class InputFile:
    @staticmethod
    def make_simulation_files(fp_user_input: str):
        data = InputFile._user_input_to_data(fp_user_input=fp_user_input)
        dir_dest = path.join(path.dirname(fp_user_input), 'in')

        for case_name_, kwargs_ in data.items():
            stochastic = dict()
            constant = dict()
            kwargs_from_input = InputParser(kwargs_, kwargs_['n_simulations']).to_dict(suppress_constant=True)
            for k in list(kwargs_from_input.keys()):
                if k in kwargs_.keys():
                    constant[k] = kwargs_from_input.pop(k)
                else:
                    stochastic[k] = kwargs_from_input.pop(k)

            if not path.exists(dir_dest):
                mkdir(dir_dest)

            dict_of_ndarray_to_csv(path.join(dir_dest, f'{case_name_}.csv'), stochastic)

            constant = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in
                        constant.items()}
            with open(path.join(dir_dest, f'{case_name_}.json'), 'w+', encoding='utf-8') as f:
                json.dump(constant, f, indent='\t')

    @staticmethod
    def _user_input_to_data(fp_user_input: str):
        fp_user_input = path.realpath(fp_user_input)
        assert path.exists(fp_user_input)

        if fp_user_input.endswith(".xlsx"):
            from openpyxl import load_workbook
            # Get the values from the worksheet
            rows = load_workbook(fp_user_input).active.values

            # Convert the rows to a dictionary
            data = {}
            keys = next(rows)
            for k in keys[1:]:
                data[k] = dict(case_name=k)
            for row in rows:
                for key_, row_ in zip(keys[1:], row[1:]):
                    data[key_][row[0]] = row_

        elif fp_user_input.endswith(".xls"):
            # Get the first worksheet
            worksheet = open_workbook(fp_user_input).sheet_by_index(0)

            # Extract the headers from the first row
            headers = [worksheet.cell_value(0, col) for col in range(worksheet.ncols)][1:]

            # Convert the rows to a dictionary
            data = {}
            for col_index, case_name_ in enumerate(headers):
                data_ = dict(case_name=case_name_)
                for row_index in range(1, worksheet.nrows):
                    data_[worksheet.cell_value(row_index, 0)] = worksheet.cell_value(row_index, col_index + 1)
                data[case_name_] = data_

        elif fp_user_input.endswith(".csv"):
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown input file format, {path.basename(fp_user_input)}")

        if len(set((data.keys()))) != len(data.keys()):
            raise ValueError(f'case_name not unique')

        return data


class MCS:
    OUTPUT_KEYS = (
        'index', 'fire_type', 't1', 't2', 't3',
        'solver_steel_temperature_solved', 'solver_time_critical_temp_solved', 'solver_protection_thickness',
        'solver_iter_count', 'solver_time_equivalence_solved', 'timber_charring_rate',
        'timber_exposed_duration', 'timber_solver_iter_count', 'timber_fire_load', 'timber_charred_depth',
        'timber_charred_mass', 'timber_charred_volume',
    )

    def __init__(self, csv_file_path, json_file_path):
        self.csv_file_path = csv_file_path
        self.json_parameters = self._load_json_parameters(json_file_path)
        self.valid_params = self._get_valid_params_for_function()

    def _load_json_parameters(self, json_file_path):
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)

    def _get_valid_params_for_function(self):
        # Using introspection to get the names of the parameters that the function accepts
        sig = inspect.signature(teq_main)
        return set(sig.parameters.keys())

    def run(self):
        results = []
        with open(self.csv_file_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Convert all values in the CSV to float for demonstration.
                # Adjust this if you have different data types.
                csv_kwargs = {k: float(v) for k, v in row.items()}

                # Merge parameters from CSV and JSON
                all_kwargs = {**self.json_parameters, **csv_kwargs}

                # Filter out any parameters not accepted by the calculation function
                valid_kwargs = {k: v for k, v in all_kwargs.items() if k in self.valid_params}

                result = teq_main(**valid_kwargs)
                results.append(result)

            case_name = path.splitext(path.basename(self.csv_file_path))[0]
            self.save_csv(
                data=results,
                fp_save=path.join(path.dirname(self.csv_file_path), f'{case_name}.out.csv'),
            )

        return results

    def save_csv(self, data: List[List], fp_save: Optional[str] = None, archive: bool = True):
        """Saves simulation output as a csv file, either in a folder (if `dir_name` is a folder) or in a zip file (if
        `dir_name` is a zip file path). `dir_name` should be cleaned properly before passing into this method."""
        assert fp_save
        # assert path.isfile(fp_save), f'Directory does not exist {fp_save}'
        assert data is not None

        # create byte object representing the save data/results
        if isinstance(data, (np.ndarray, tuple, list)):
            content = BytesIO()
            np.savetxt(content, data, delimiter=",", header=','.join(self.OUTPUT_KEYS), fmt='%g', comments='')
            content.seek(0)
        elif data is None:
            raise ValueError('No results to save')
        else:
            raise ValueError(f'Unknown results data type {type(data)}')

        # save result to file
        # in a folder
        with open(fp_save, 'wb+') as f:
            shutil.copyfileobj(content, f)
        return


if __name__ == '__main__':
    InputFile.make_simulation_files(
        fp_user_input=r'C:\Users\ian\Desktop\sfeprapy_test\test.xlsx',
    )

    CalculationHelper(
        r'C:\Users\ian\Desktop\sfeprapy_test\in\CASE_1.csv',
        r'C:\Users\ian\Desktop\sfeprapy_test\in\CASE_1.json'
    ).run()
