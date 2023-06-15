# Monte Carlo Simulation Multiple Process Implementation
# Yan Fu, October 2017

import io
import multiprocessing as mp
import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from inspect import getfullargspec
from io import StringIO
from typing import Callable, Optional, Dict, Union, Any

import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d

from sfeprapy.func.xlsx import dict_to_xlsx


class TrueToScipy:
    """Converts 'normal' distribution parameters, e.g. normal, standard deviation etc., to Scipy recognisable
    parameters, e.g. loc, scale etc.
    """

    @staticmethod
    def gumbel_r_(mean: float, sd: float, **_):
        # parameters Gumbel W&S
        alpha = 1.282 / sd
        u = mean - 0.5772 / alpha

        # parameters Gumbel scipy
        scale = 1 / alpha
        loc = u

        return dict(loc=loc, scale=scale)

    @staticmethod
    def lognorm_(mean: float, sd: float, **_):
        cov = sd / mean

        sigma_ln = np.sqrt(np.log(1 + cov ** 2))
        miu_ln = np.log(mean) - 1 / 2 * sigma_ln ** 2

        s = sigma_ln
        loc = 0
        scale = np.exp(miu_ln)

        return dict(s=s, loc=loc, scale=scale)

    @staticmethod
    def lognorm_mod_(mean: float, sd: float, **_):
        return TrueToScipy.lognorm_(mean, sd, **_)

    @staticmethod
    def norm_(mean: float, sd: float, **_):
        loc = mean
        scale = sd

        return dict(loc=loc, scale=scale)

    @staticmethod
    def uniform_(ubound: float, lbound: float, **_):
        if lbound > ubound:
            lbound += ubound
            ubound = lbound - ubound
            lbound -= ubound

        loc = lbound
        scale = ubound - lbound

        return dict(loc=loc, scale=scale)


class InputParser:
    """Converts """

    def __init__(self, dist_params: dict, n: int):
        assert isinstance(dist_params, dict)
        assert isinstance(n, int)

        self.__n = n
        self.__in_raw = dist_params
        self.__in = InputParser.unflatten_dict(dist_params)

    def to_dict(self):
        n = self.__n
        dist_params = self.__in
        dict_out = dict()

        for k, v in dist_params.items():
            if isinstance(v, float) or isinstance(v, int) or isinstance(v, float):
                dict_out[k] = np.full((n,), v, dtype=float)
            elif isinstance(v, str):
                dict_out[k] = np.full(
                    (n,), v, dtype=np.dtype("U{:d}".format(len(v)))
                )
            elif isinstance(v, np.ndarray) or isinstance(v, list):
                dict_out[k] = list(np.full((n, len(v)), v, dtype=float))
            elif isinstance(v, dict):
                if "dist" in v:
                    try:
                        dict_out[k] = InputParser._sampling(v, n)
                    except KeyError:
                        raise KeyError(f"Missing parameters in input variable {k}.")
                elif "ramp" in v:
                    s_ = StringIO(v["ramp"])
                    d_ = np.loadtxt(s_, delimiter=',')
                    t_ = d_[:, 0]
                    v_ = d_[:, 1]
                    if all(v_ == v_[0]):
                        f_interp = v_[0]
                    else:
                        f_interp = interp1d(t_, v_, bounds_error=False, fill_value=0)
                    dict_out[k] = np.full((n,), f_interp)
                else:
                    raise ValueError("Unknown input data type for {}.".format(k))
            elif v is None:
                dict_out[k] = np.full((n,), np.nan, dtype=float)
            else:
                raise TypeError(f"Unknown input data type for {k}.")

        dict_out["index"] = np.arange(0, n, 1)
        return dict_out

    def to_xlsx(self, fp: str):
        dict_to_xlsx({i: InputParser.flatten_dict(v) for i, v in self.to_dict().items()}, fp)

    @staticmethod
    def unflatten_dict(dict_in: dict) -> dict:
        """Invert flatten_dict.

        :param dict_in:
        :return dict_out:
        """
        dict_out = dict()

        for k, v in dict_in.items():
            InputParser.__unflatten_dict(k, v, dict_out)

        return dict_out

    @staticmethod
    def __unflatten_dict(k: str, v: Any, dict_out: dict):
        if ":" in k:
            k1, *k2 = k.split(':')
            if k1 not in dict_out:
                dict_out[k1] = dict()
            InputParser.__unflatten_dict(':'.join(k2), v, dict_out[k1])
        else:
            dict_out[k] = v

    @staticmethod
    def flatten_dict(dict_in: dict) -> dict:
        dict_out = dict()
        InputParser.__flatten_dict(dict_in, dict_out)
        return dict_out

    @staticmethod
    def __flatten_dict(dict_in: dict, dict_out: dict, history: str = None):
        """Converts two levels dict to single level dict. Example input and output see _test_dict_flatten.
        >>> dict_in = {
        >>>             'a': 1,
        >>>             'b': {'b1': 21, 'b2': 22},
        >>>             'c': {'c1': 31, 'c2': 32, 'c3': 33}
        >>>         }
        >>> output = {
        >>>             'a': 1,
        >>>             'b:b1': 21,
        >>>             'b:b2': 22,
        >>>             'c:c1': 31,
        >>>             'c:c2': 32,
        >>>             'c:c3': 33,
        >>>         }
        >>> assert InputParser.flatten_dict(dict_in) == output  # True

        :param dict_in:     Any two levels (or less) dict.
        :return dict_out:   Single level dict.
        """
        for k, v in dict_in.items():
            if isinstance(v, dict):
                InputParser.__flatten_dict(v, dict_out=dict_out, history=k if history is None else f'{history}:{k}')
            else:
                dict_out[f'{k}' if history is None else f'{history}:{k}'] = v
        # for k in list(dict_in.keys()):
        #     if isinstance(dict_in[k], dict):
        #         for kk, vv in dict_in[k].items():
        #             dict_out[f"{k}:{kk}"] = vv
        #     else:
        #         dict_out[k] = dict_in[k]
        # return dict_out

    @staticmethod
    def _sampling(dist_params: dict, num_samples: int, randomise: bool = True) -> Union[float, np.ndarray]:
        """Evacuate sampled values based on a defined distribution. This is build upon `scipy.stats` library.

        :param dist_params: Distribution inputs, required keys are distribution dependent, should be aligned with inputs
                            required in the scipy.stats. Additional compulsory keys are:
                                `dist`: str, distribution type.
        :param num_samples: Number of samples to be generated.
        :param randomise:   Whether to randomise the sampled values.
        :return samples:    Sampled values based upon `dist` in the range [`lbound`, `ubound`] with `num_samples` number
                            of values.
        """
        if dist_params['dist'] == 'discrete_':
            v_ = dist_params['values']
            if isinstance(v_, str):
                assert ',' in v_, f'`discrete_ distribution `values` parameter is not a list separated by comma.'
                v_ = [float(i.strip()) for i in v_.split(',')]

            w_ = dist_params['weights']
            if isinstance(w_, str):
                assert ',' in w_, f'`discrete_`:`weights` is not a list of numbers separated by comma.'
                w_ = [float(i.strip()) for i in w_.split(',')]

            assert len(v_) == len(w_), f'Length of values ({len(v_)}) and weights ({len(v_)}) do not match.'
            assert sum(w_) == 1., f'Sum of all weights should be unity, got {sum(w_)}.'

            w_ = [int(round(i * num_samples)) for i in w_]
            if (sum_sampled := sum(w_)) < num_samples:
                for i in np.random.choice(np.arange(len(w_)), size=sum_sampled - num_samples):
                    w_[i] += 1
            elif sum_sampled > num_samples:
                for i in np.random.choice(np.arange(len(w_)), size=sum_sampled - num_samples):
                    w_[i] -= 1
            w_ = np.cumsum(w_)
            assert w_[-1] == num_samples, f'Total weight length does not match `num_samples`.'
            samples = np.empty((num_samples,), dtype=float)
            for i, v__ in enumerate((v_)):
                if i == 0:
                    samples[0:w_[i]] = v__
                else:
                    samples[w_[i - 1]:w_[i]] = v__

            if randomise:
                np.random.shuffle(samples)

            return samples

        if dist_params['dist'] == 'constant_':
            return np.full((num_samples,), (dist_params['lbound'] + dist_params['ubound']) / 2, dtype=float)

        # sample CDF points (y-axis value)
        def generate_cfd_q(dist, dist_params_scipy, lbound, ubound, num_samples_=None):
            num_samples_ = num_samples if num_samples_ is None else num_samples_
            cfd_q_ = np.linspace(
                getattr(stats, dist).cdf(x=lbound, **dist_params_scipy),
                getattr(stats, dist).cdf(x=ubound, **dist_params_scipy),
                num_samples_,
            )
            samples_ = getattr(stats, dist).ppf(q=cfd_q_, **dist_params_scipy)
            return samples_

        # convert true distribution parameters to scipy distribution parameters
        try:
            if dist_params['dist'] == 'lognorm_mod_':
                dist_params_scipy = getattr(TrueToScipy, 'lognorm_')(
                    **dist_params
                )
                samples = generate_cfd_q(
                    dist='lognorm', dist_params_scipy=dist_params_scipy, lbound=dist_params['lbound'],
                    ubound=dist_params['ubound']
                )
                samples = 1 - samples
            elif dist_params['dist'] == 'br187_fuel_load_density_':
                dist_params_list = list()
                dist_params_list.append(
                    dict(dist='gumbel_r_', lbound=dist_params['lbound'], ubound=dist_params['ubound'], mean=780,
                         sd=234))
                dist_params_list.append(
                    dict(dist='gumbel_r_', lbound=dist_params['lbound'], ubound=dist_params['ubound'], mean=420,
                         sd=126))
                samples_ = list()
                for dist_params in dist_params_list:
                    dist_params_scipy = getattr(TrueToScipy, dist_params['dist'])(**dist_params)
                    samples__ = generate_cfd_q(
                        dist=dist_params['dist'].rstrip('_'), dist_params_scipy=dist_params_scipy,
                        lbound=dist_params['lbound'], ubound=dist_params['ubound']
                    )
                    samples_.append(samples__)
                samples = np.random.choice(np.append(*samples_), num_samples, replace=False)
            elif dist_params['dist'] == 'br187_hrr_density_':
                dist_params_list = list()
                dist_params_list.append(dict(dist='uniform_', lbound=0.32, ubound=0.57))
                dist_params_list.append(dict(dist='uniform_', lbound=0.15, ubound=0.65))
                samples_ = list()
                for dist_params in dist_params_list:
                    dist_params_scipy = getattr(TrueToScipy, dist_params['dist'])(**dist_params)
                    samples__ = generate_cfd_q(
                        dist=dist_params['dist'].rstrip('_'), dist_params_scipy=dist_params_scipy,
                        lbound=dist_params['lbound'], ubound=dist_params['ubound']
                    )
                    samples_.append(samples__)
                samples = np.random.choice(np.append(*samples_), num_samples, replace=False)
            else:
                dist_params_scipy = getattr(TrueToScipy, dist_params['dist'])(**dist_params)
                samples = generate_cfd_q(
                    dist=dist_params['dist'].rstrip('_'), dist_params_scipy=dist_params_scipy,
                    lbound=dist_params['lbound'], ubound=dist_params['ubound']
                )

        except Exception as e:
            try:
                samples = generate_cfd_q(
                    dist=dist_params['dist'], dist_params_scipy=dist_params, lbound=dist_params['lbound'],
                    ubound=dist_params['ubound']
                )
            except AttributeError:
                raise ValueError(f"Unknown distribution type {dist_params['dist']}, {e}")

        samples[samples == np.inf] = dist_params['ubound']
        samples[samples == -np.inf] = dist_params['lbound']

        if "permanent" in dist_params:
            samples += dist_params["permanent"]

        if randomise:
            np.random.shuffle(samples)

        return samples


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

        nested_args = list()
        for k in keys_from_worker_required:
            if k not in kwargs_from_input:
                missing_args.append(k)
                continue
            nested_args.append(kwargs_from_input[k])

        if len(missing_args) > 0:
            raise ValueError(f'Missing required arguments: {missing_args}.')

        for i, k in enumerate(keys_from_worker_optional):
            if k in keys_from_worker:
                nested_args.append(kwargs_from_input[k])
            else:
                nested_args.append(defaults_from_worker[i])

        # ===============
        # start processes
        # ===============
        output = list()
        if p is None:
            if set_progress is not None:
                for i, arg in enumerate(zip(*nested_args)):
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
        assert os.path.exists(dir_save), f'Directory does not exist {dir_save}'
        assert self.__output is not None

        # create byte object representing the save data/results
        if isinstance(self.__output, np.ndarray):
            content = io.BytesIO()
            np.savetxt(content, self.__output, delimiter=",", header=','.join(self.output_keys), fmt='%g', comments='')
            content.seek(0)
        elif self.__output is None:
            raise ValueError('No results to save')
        else:
            raise ValueError(f'Unknown results data type {type(self.__output)}')

        # save result to file
        if archive:
            # in a zip file
            with zipfile.ZipFile(dir_save, 'a', compression=zipfile.ZIP_DEFLATED) as f_zip:
                f_zip.writestr(f'{self.name}.csv', content.read(), compress_type=zipfile.ZIP_DEFLATED)
            return
        else:
            # in a folder
            with open(os.path.join(dir_save, f'{self.name}.csv'), 'wb+') as f:
                shutil.copyfileobj(content, f)
            return

    def load_output_from_file(self, fp: str):
        fp = os.path.realpath(fp)

        if zipfile.is_zipfile(fp):
            with zipfile.ZipFile(fp, 'r') as f_zip:
                with f_zip.open(f'{self.name}.csv') as f:
                    self.__output = np.genfromtxt(io.TextIOWrapper(f), delimiter=',', skip_header=1, )
        else:
            fp = os.path.join(fp, f'{self.name}.csv')
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
        return os.path.join(os.path.dirname(self.__in_fp), f'{os.path.splitext(os.path.basename(self.__in_fp))[0]}.out')

    def get_inputs_dict(self):
        return self.__in_dict

    def get_inputs_file_path(self):
        return self.__in_fp

    def set_inputs_dict(self, data: dict):
        for case_name_, kwargs_ in data.items():
            self.mcs_cases[case_name_] = self.new_mcs_case(
                name=case_name_, n_simulations=kwargs_['n_simulations'], sim_kwargs=kwargs_,
                save_dir=os.path.join(os.path.dirname(self.__in_fp), self.get_save_dir())
            )

        self.__in_dict = data

    def set_inputs_file_path(self, fp):
        fp = os.path.realpath(fp)
        assert os.path.exists(fp)

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
            raise ValueError(f"Unknown input file format, {os.path.basename(fp)}")

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

        if cases_to_run:
            n_case = sum([1 if k in cases_to_run else 0 for k, v in self.mcs_cases.items()])
            n_sim = sum([v.n_sim if k in cases_to_run else 0 for k, v in self.mcs_cases.items()])
        else:
            n_case = sum([1 for k, v in self.mcs_cases.items()])
            n_sim = sum([v.n_sim for k, v in self.mcs_cases.items()])

        # concurrency strategy
        # terminology:
        #   simulation iteration: simply a single simulation iteration, e.g., calling a function and get results
        #   simulation case: describes a monte carlo simulation case, consists of a defined number of simulation
        #   iterations.
        #
        # specific to the application of this class, many simulation cases may present and each simulation case consists
        # many simulation iteratoins to be run. multiprocessing is utilised to speed up the computation performance
        # given modern CPUs are supplied with multiple logical processors. However, multiprocessing can be applied in
        # different ways in terms which level of the loops to chip in. Below is an example of the loop pattern:
        #
        #   for each_case in mcs_cases:
        #           pass  # task 1: prepare/sampling inputs for `each_case`...
        #       for each_iteration in each_case:
        #           pass  # task 2: computation happen here...
        #       pass # task 3: save output to disk...
        #
        # task 1 and 2 are CPU bound and task 3 is I/O bound.
        #
        # strategy 1:
        #   each case get allocated to a process. the benefit is clear as all the tasks are non-blocking to other cases.
        #   however, the computational time will be dictated by the case that takes the longest time. this strategy
        #   would be beneficial when the E/W (executions and number of available workers ratio) is significant.
        # strategy 2:
        #   each computation/execution get allocated to a process. following strategy 1, this is
        # CPU-bounded task as the only I/O task would be after each simulation case
        concurrency_strategy: int = 0

        if save:
            self.save_init(archive=save_archive)

        try:
            set_progress_max(n_sim)
        except TypeError:
            pass

        progress_0 = None
        if n_proc > 3:
            with ProcessPoolExecutor(max_workers=n_proc) as p_executor:
                # futures = {p.submit(self.worker, *arg) for arg in zip(*nested_args)}
                futures = {
                    p_executor.submit(mcs.run, save=True, save_archive=save_archive)
                    for mcs in self.mcs_cases.values()
                }
                if set_progress is not None:
                    for i, future in enumerate(as_completed(futures), start=1):
                        set_progress(0 + i)
        else:
            with ThreadPoolExecutor(max_workers=2) as t_executor:
                with ProcessPoolExecutor(max_workers=n_proc) as p_executor:
                    for mcs_case_name, mcs_case in self.mcs_cases.items():  # Reuse the executor for 3 sets of tasks
                        if cases_to_run and mcs_case_name not in cases_to_run:
                            continue
                        progress_0 = 0 if progress_0 is None else progress_0 + mcs_case.n_sim
                        mcs_case.run(p_executor, set_progress=set_progress, progress_0=progress_0)
                        if save:
                            t_executor.submit(mcs_case.save_csv, None, save_archive)

    def save_init(self, archive: bool):
        # clean existing files
        try:
            os.remove(self.get_save_dir())
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
            os.makedirs(self.get_save_dir())

    def save_all(self, archive: bool = True):
        self.save_init(archive=archive)

        # write results
        for k, v in self.mcs_cases.items():
            v.save_csv(archive=archive)
        return

    def load_from_file(self, fp_in: str, fp_out: str = None):
        self.__in_fp: str = os.path.realpath(fp_in)
        self.set_inputs_file_path(fp_in)  # input parameters
        for name, mcs_case in self.mcs_cases.items():
            mcs_case.load_output_from_file(fp_out)
