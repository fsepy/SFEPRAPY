# Monte Carlo Simulation Multi-Process Implementation
# Yan Fu, October 2017

import multiprocessing as mp
import random
import shutil
import time
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from inspect import getfullargspec
from io import BytesIO, TextIOWrapper
from os import path, remove, makedirs
from typing import Callable, Optional, Dict

import numpy as np

from sfeprapy.input_parser import InputParser


class MCSSingle(ABC):
    def __init__(self, name: str, n_simulations: int, sim_kwargs: dict, save_dir: Optional[str] = None):
        assert n_simulations > 0

        self.name: str = name  # case name
        self.n_sim: int = n_simulations  # number of simulation
        self.input = InputParser(sim_kwargs, n_simulations)
        self.save_dir: Optional[str] = save_dir

        # {'var1': [0, ...], 'var2': [2, 3, ...], ...}
        self.__output: Optional[np.ndarray] = None  # {'res1': [...], 'res2': [...], ...}

    @property
    def input_keys(self) -> tuple[tuple, tuple]:
        """Placeholder method. The input value names from the Monte Carlo Simulation deterministic calculation
        routine.
        :return:    (args, n_args) where `args` is a list of argument names and `n_args` is the number of compulsory
                    arguments
        """
        _ = getfullargspec(self.worker)
        return tuple(_.args), _.defaults

    def run(self, p: mp.Pool = None, set_progress: Optional[Callable] = None, progress_0: int = 0, save: bool = False,
            save_archive: bool = False):
        # ======================
        # prepare input iterable
        # ======================
        kwargs_from_input = self.input.to_dict()
        keys_from_worker, defaults_from_worker = self.input_keys
        n_keys_from_worker = len(keys_from_worker) - (0 if defaults_from_worker is None else len(defaults_from_worker))
        keys_from_worker_required = keys_from_worker[:n_keys_from_worker]
        keys_from_worker_optional = keys_from_worker[n_keys_from_worker:]

        # check if all the required arguments are provided
        missing_args = list()
        for k in keys_from_worker_required:
            if k not in keys_from_worker:
                missing_args.append(k)
        if len(missing_args) > 0:
            raise ValueError(f'Missing arguments: {missing_args}.')

        nested_args = list(kwargs_from_input[k] for k in keys_from_worker_required)

        for i, k in enumerate(keys_from_worker_optional):
            if k in kwargs_from_input:
                nested_args.append(kwargs_from_input[k])
            else:
                nested_args.append((defaults_from_worker[i] for __ in range(self.n_sim)))

        # ===============
        # start processes
        # ===============
        output = list()
        if p is None:
            if set_progress is not None:
                for i, arg in enumerate(zip(*nested_args)):
                    print(i, arg)
                    output.append(self.worker(*arg))
                    set_progress(progress_0 + i)
            else:
                for arg in zip(*nested_args):
                    output.append(self.worker(*arg))
        else:
            futures = {p.submit(self.worker, *arg) for arg in zip(*nested_args)}
            if set_progress is not None:
                for i, future in enumerate(as_completed(futures), start=1):
                    output.append(future.result())
                    set_progress(progress_0 + i)
            else:
                for i, future in enumerate(as_completed(futures), start=1):
                    output.append(future.result())

        self.__output = np.array(output)

        if save:
            self.save_csv(dir_save=None, archive=save_archive)

        return self.__output

    @property
    @abstractmethod
    def worker(self) -> Callable:
        """Placeholder method. The Monte Carlo Simulation deterministic calculation routine.
        :return:
        """
        raise NotImplementedError('This method should be overridden by a child class')

    @property
    @abstractmethod
    def output_keys(self) -> tuple:
        """Placeholder method. The returned value names from the Monte Carlo Simulation deterministic calculation
        routine.
        :return:
        """
        raise NotImplementedError('This method should be overridden by a child class')

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, d: np.ndarray):
        self.__output = d

    @staticmethod
    def make_pdf(data: np.ndarray, bin_width: float = 0.2) -> (np.ndarray, np.ndarray, np.ndarray):
        # Set all time equivalence to be no more than 5 hours (18000 seconds)
        data[data >= 18000.] = 17999.999

        # Set all zero time equivalence to 1e-3 (cannot recall why I did this)
        data[data <= 0] = 1e-3

        # [s] -> [min]
        data /= 60.

        assert np.nanmax(data) < 300.
        assert np.nanmin(data) > 0.

        edges = np.arange(0, 300 + bin_width, bin_width)
        x = (edges[1:] + edges[:-1]) / 2  # make x-axis values, i.e. time equivalence

        y_pdf = np.histogram(data, edges)[0] / len(data)

        return x, y_pdf

    @staticmethod
    def make_cdf(data: np.ndarray, bin_width: float = 0.2):
        x, y_pdf = MCSSingle.make_pdf(data=data, bin_width=bin_width)
        return x, np.cumsum(y_pdf)

    def save_csv(self, dir_save: Optional[str] = None, archive: bool = True):
        """Saves simulation output as a csv file, either in a folder (if `dir_name` is a folder) or in a zip file (if
        `dir_name` is a zip file path). `dir_name` should be cleaned properly before passing into this method."""
        if dir_save is None:
            dir_save = self.save_dir
        assert dir_save
        assert path.exists(dir_save), f'Directory does not exist {dir_save}'
        assert self.__output is not None

        # create byte object representing the save data/results
        if isinstance(self.__output, np.ndarray):
            content = BytesIO()
            np.savetxt(content, self.__output, delimiter=",", header=','.join(self.output_keys), fmt='%g', comments='')
            content.seek(0)
        elif self.__output is None:
            raise ValueError('No results to save')
        else:
            raise ValueError(f'Unknown results data type {type(self.__output)}')

        # save result to file
        if archive:
            # in a zip file
            for i in range(40):
                try:
                    with zipfile.ZipFile(dir_save, 'a', compression=zipfile.ZIP_DEFLATED) as f_zip:
                        f_zip.writestr(f'{self.name}.csv', content.read(), compress_type=zipfile.ZIP_DEFLATED)
                    return
                except Exception:
                    time.sleep(random.randint(1, 5))
        else:
            # in a folder
            with open(path.join(dir_save, f'{self.name}.csv'), 'wb+') as f:
                shutil.copyfileobj(content, f)
            return

    def load_output_from_file(self, fp: str):
        fp = path.realpath(fp)

        if zipfile.is_zipfile(fp):
            with zipfile.ZipFile(fp, 'r') as f_zip:
                with f_zip.open(f'{self.name}.csv') as f:
                    self.__output = np.genfromtxt(TextIOWrapper(f), delimiter=',', skip_header=1, )
        else:
            fp = path.join(fp, f'{self.name}.csv')
            self.__output = np.genfromtxt(fp, delimiter=',', skip_header=1, )

        assert (self.n_sim, len(self.output_keys)) == tuple(self.__output.shape)


class MCS(ABC):
    """An abstract class purposed to provide infrastructure in supporting time equivalence stochastic analysis utilising
    Monte Carlo simulation.

    To help to understand this class, a brief process of a MCS is outlined below:

        1. stochastic problem definition (i.e. raw input parameters from the user).
        2. sample deterministic parameters from the stochastic inputs
        3. iterate the sampled parameters and run deterministic calculation
        4. summarise results

    Following above, this object consists of four primary methods, each matching one of the above step:

        `MCS.mcs_inputs`
            A method to intake user inputs, restructure data so usable within MCS.
        `MCS.mcs_sampler`
            A method to sample deterministic parameters from the stochastic parameters, produces input to be used in
            the next step.
        `MCS.mcs_deterministic_calc`
            A method to carry out deterministic calculation.
            NOTE! This method needs to be re-defined in a child class.
        `MCS.mcs_post_per_case`
            A method to post-processing results.
            NOTE! This method needs to be re-defined in a child class.
    """

    # DEFAULT_SAVE_FOLDER_NAME = "{}.pra"  #

    def __init__(self):
        self.__in_fp: str = ''
        self.__in_dict: Optional[dict] = None  # input parameters
        self.__mp_pool: Optional[np.Pool] = None
        self.mcs_cases: Dict[str, MCSSingle] = dict()

    def get_save_dir(self):
        return path.join(path.dirname(self.__in_fp), f'{path.splitext(path.basename(self.__in_fp))[0]}.out')

    def get_inputs_dict(self):
        return self.__in_dict

    def get_inputs_file_path(self):
        return self.__in_fp

    def set_inputs_dict(self, data: dict):
        for case_name_, kwargs_ in data.items():
            self.mcs_cases[case_name_] = self.new_mcs_case(
                name=case_name_, n_simulations=kwargs_['n_simulations'], sim_kwargs=kwargs_,
                save_dir=path.join(path.dirname(self.__in_fp), self.get_save_dir())
            )

        self.__in_dict = data

    def set_inputs_file_path(self, fp):
        fp = path.realpath(fp)
        assert path.exists(fp)

        if fp.endswith(".xlsx"):
            from openpyxl import load_workbook
            # Get the values from the worksheet
            rows = load_workbook(fp).active.values

            # Convert the rows to a dictionary
            data = {}
            keys = next(rows)
            for k in keys[1:]:
                data[k] = dict(case_name=k)
            for row in rows:
                for key_, row_ in zip(keys[1:], row[1:]):
                    data[key_][row[0]] = row_
            self.__in_fp = fp
            self.__in_dict = data

        elif fp.endswith(".xls"):
            from xlrd import open_workbook

            # Get the first worksheet
            worksheet = open_workbook(fp).sheet_by_index(0)

            # Extract the headers from the first row
            headers = [worksheet.cell_value(0, col) for col in range(worksheet.ncols)][1:]

            # Convert the rows to a dictionary
            data = {}
            for col_index, case_name_ in enumerate(headers):
                data_ = dict(case_name=case_name_)
                for row_index in range(1, worksheet.nrows):
                    data_[worksheet.cell_value(row_index, 0)] = worksheet.cell_value(row_index, col_index + 1)
                data[case_name_] = data_

        elif fp.endswith(".csv"):
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown input file format, {path.basename(fp)}")

        if len(set((data.keys()))) != len(data.keys()):
            raise ValueError(f'case_name not unique')

        self.__in_fp = fp

        self.set_inputs_dict(data)

        return data

    @property
    @abstractmethod
    def new_mcs_case(self) -> MCSSingle:
        raise NotImplementedError('This method should be implemented by the child class.')

    def run(
            self,
            n_proc: int = 1,
            set_progress: Optional[Callable] = None,
            set_progress_max: Optional[Callable] = None,
            save: bool = False,
            save_archive: bool = False,
            cases_to_run: Optional[list] = None,
            concurrency_strategy: Optional[int] = 0,
    ):
        # check if all `cases_to_run` exist in `self.mcs_cases`
        if cases_to_run:
            undefined_case_name_by_user = list()
            for case_name in cases_to_run:
                if case_name not in self.mcs_cases:
                    undefined_case_name_by_user.append(case_name)
            if undefined_case_name_by_user:
                raise ValueError(
                    f'The following case are specified to run but they do not exist: {undefined_case_name_by_user}. '
                    f'Available case names: {self.mcs_cases.keys()}.'
                )
            else:
                del undefined_case_name_by_user

        '''Concurrency Strategy

        Procedure description:

            1.  Go through each simulation case and to the following.
                1.1 Task 1: Prepare arguments for simulation iterations.
                1.2 Go through each simulation iteration in the simulation case, for each simulation iteration do the 
                    following.
                    1.2.1   Task 2: Perform the computation.
                1.3 Write output to disk.

        Terminology:

            1.  Simulation iteration: This refers to a single run of the simulation. The computation-intensive task (task 
                2) happens here. 
            2.  Simulation case: This refers to a set of simulation iterations. It involves multiple iterations of 
                task 2 but also includes the preparation/sampling of inputs (task 1) and the output saving operation 
                (task 3).
            3.  The E/W ratio, which reflects the balance of computational workloads across available processing units. 
            4.  The overhead cost (O), which involves the expenses associated with managing multiple processes. 
            5.  The blocking cost (B), which reflects the performance impact of I/O bound operations blocking CPU-bound 
                tasks.

        In the application of this class, many simulation cases are typically present. Each simulation case consists 
        of multiple simulation iterations. To improve computation performance, multiprocessing is used to leverage 
        the multi-core nature of modern CPUs. There are two possible strategies to apply multiprocessing, 
        based on the level of parallelism:

        Strategy 1: Case-level Parallelism

            In this strategy, each simulation case is allocated to a separate process. This strategy is beneficial 
            when the ratio of executions (simulation iterations) to available workers (E/W) is significant. It can 
            efficiently utilise CPU resources when execution times within each case are roughly equal. However, 
            if there's significant variation in execution times within each case, some CPU cores might be 
            underutilized if they finish their tasks much earlier than others.

        Strategy 2: Iteration-level Parallelism

            In this strategy, each simulation iteration is allocated to a separate process. This can provide a more 
            balanced workload across processes, especially when there's high variability in the execution times of 
            iterations. However, there's additional overhead involved in managing more processes, and the main 
            process could become a bottleneck if it's blocked by I/O operations.

        Things to Consider:

            Workload Distribution and Balance: Workload balance is a critical aspect in multiprocessing. A skewed 
            workload can lead to idle CPU resources when some processes finish much earlier than others. For better 
            load balancing, dynamic task scheduling strategies might be required and this is done through the
            `ProcessPoolExecutor` but may raise cost in strategy 1.

            Overhead of Multiprocessing: While multiprocessing can speed up computation, it also introduces overhead 
            when perform `submit()` in `ProcessPoolExecutor`. 
            It's crucial to ensure that the gain from multiprocessing outweighs the overhead costs. Otherwise, 
            the overall performance might be worse than a single-process implementation.

            I/O Bound Operations: Task 3 (saving output to disk) is I/O bound and could potentially block CPU-bound 
            tasks. Consider moving the I/O operations to a separate process, or use asynchronous I/O if possible, 
            to prevent blocking the CPU-bound tasks.

            Execution Time Variation: The estimation of E/W seems to be based on the assumption that each execution 
            takes roughly the same amount of time. If the execution times vary significantly, this estimation might 
            not accurately reflect the workload.

        Strategy Selection:

        To assist in this decision-making process, it can be useful to represent these factors in a cost function, 
        where the total cost is the sum of the E/W ratio, overhead cost, and blocking cost:

        Cost = E/W + O + B

        The strategy with the lowest computed cost could be selected as the optimal strategy for a given situation. 
        However, keep in mind that quantifying costs like O and B can be challenging, and they may vary depending on 
        factors such as hardware characteristics, operating system behavior, and the specific nature of the tasks.

        As a temporary measure, an E/W ratio threshold of 1.5 is currently being used as the sole determinant of the 
        strategy, with the understanding that this approach may not fully capture the complexities of the costs 
        involved. Further research and testing are required to refine the cost function and its use in strategy 
        selection.

        Moreover, it can be beneficial to experiment with different strategies and configurations, as well as to 
        perform benchmark tests under a variety of conditions, to find the optimal solution for your particular 
        scenario. Remember that this cost function and threshold might not be optimal for all scenarios and may need 
        to be adjusted based on empirical data.'''

        if cases_to_run:
            n_case = sum([1 if k in cases_to_run else 0 for k, v in self.mcs_cases.items()])
            n_sim = sum([v.n_sim if k in cases_to_run else 0 for k, v in self.mcs_cases.items()])
        else:
            n_case = len(self.mcs_cases)
            n_sim = sum([v.n_sim for k, v in self.mcs_cases.items()])

        if concurrency_strategy == 0:
            if 4 < n_proc < n_case:
                concurrency_strategy = 1
            else:
                concurrency_strategy = 2

        if save:
            self.save_init(archive=save_archive)

        progress_0 = None
        if n_proc == 0:
            for mcs_case_name, mcs_case in self.mcs_cases.items():  # Reuse the executor for 3 sets of tasks
                if cases_to_run and mcs_case_name not in cases_to_run:
                    continue
                progress_0 = 0 if progress_0 is None else progress_0 + mcs_case.n_sim
                mcs_case.run()
                if save:
                    mcs_case.save_csv(archive=save_archive)
        elif concurrency_strategy == 1:
            try:
                set_progress_max(n_case)
            except TypeError:
                pass

            output = [None] * len(self.mcs_cases)  # Pre-allocate a list to hold results in order
            futures_to_case = {}  # Map each future to its corresponding case name

            with ThreadPoolExecutor(max_workers=1) as t_save_output:
                with ProcessPoolExecutor(max_workers=min(n_proc, n_case)) as p_executor:
                    # Submit all tasks and remember their order
                    mcs_cases_keys = list(self.mcs_cases.keys())  # Get a list of all case names
                    for i, case_name in enumerate(mcs_cases_keys):
                        future = p_executor.submit(self.mcs_cases[case_name].run, save=False)
                        futures_to_case[future] = case_name

                    if set_progress is not None:
                        # Wait for the futures to complete and collect their results in submission order
                        for future in as_completed(futures_to_case):
                            case_name = futures_to_case[future]
                            index = mcs_cases_keys.index(case_name)  # Find the index of the case based on its name
                            result = future.result()
                            output[index] = result
                            self.mcs_cases[case_name].output = result  # Directly access the case by its name
                            if save:
                                t_save_output.submit(self.mcs_cases[case_name].save_csv, None, save_archive)
                            set_progress(index + 1)
                    else:
                        for future in as_completed(futures_to_case):
                            case_name = futures_to_case[future]
                            index = mcs_cases_keys.index(case_name)
                            result = future.result()
                            output[index] = result
                            self.mcs_cases[case_name].output = result  # Directly access the case by its name
                            if save:
                                t_save_output.submit(self.mcs_cases[case_name].save_csv, None, save_archive)

        elif concurrency_strategy == 2:
            try:
                set_progress_max(n_sim)
            except TypeError:
                pass

            with ThreadPoolExecutor(max_workers=1) as t_executor:
                with ProcessPoolExecutor(max_workers=n_proc) as p_executor:
                    for mcs_case_name, mcs_case in self.mcs_cases.items():  # Reuse the executor for 3 sets of tasks
                        if cases_to_run and mcs_case_name not in cases_to_run:
                            continue
                        progress_0 = 0 if progress_0 is None else progress_0 + mcs_case.n_sim
                        mcs_case.run(p_executor, set_progress=set_progress, progress_0=progress_0)
                        if save:
                            t_executor.submit(mcs_case.save_csv, None, save_archive)
        else:
            raise NotImplementedError(f'Unknown `concurrency_strategy` {concurrency_strategy}.')

    def save_init(self, archive: bool):
        # clean existing files
        try:
            remove(self.get_save_dir())
        except:
            pass
        try:
            shutil.rmtree(self.get_save_dir())
        except:
            pass

        # create empty folder or zip
        if archive:
            with open(self.get_save_dir(), 'wb+') as f:
                f.write(b'PK\x05\x06\x00l\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        else:
            makedirs(self.get_save_dir())

    def save_all(self, archive: bool = True):
        self.save_init(archive=archive)

        # write results
        for k, v in self.mcs_cases.items():
            v.save_csv(archive=archive)
        return

    def load_from_file(self, fp_in: str, fp_out: str = None):
        self.__in_fp: str = path.realpath(fp_in)
        self.set_inputs_file_path(fp_in)  # input parameters
        for name, mcs_case in self.mcs_cases.items():
            mcs_case.load_output_from_file(fp_out)
