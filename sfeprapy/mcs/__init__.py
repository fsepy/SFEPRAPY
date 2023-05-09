# Monte Carlo Simulation Multiple Process Implementation
# Yan Fu, October 2017

import concurrent.futures
import io
import multiprocessing as mp
import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from io import StringIO
from typing import Callable, Optional, Dict, Union

import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d


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
                        raise (f"Missing parameters in input variable {k}.")
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

    @staticmethod
    def unflatten_dict(dict_in: dict) -> dict:
        """Invert flatten_dict.

        :param dict_in:
        :return dict_out:
        """
        dict_out = dict()

        for k in list(dict_in.keys()):
            if ":" in k:
                k1, k2 = k.split(":")

                if k1 in dict_out:
                    dict_out[k1][k2] = dict_in[k]
                else:
                    dict_out[k1] = {k2: dict_in[k]}
            else:
                dict_out[k] = dict_in[k]

        return dict_out

    @staticmethod
    def flatten_dict(dict_in: dict) -> dict:
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

        dict_out = dict()

        for k in list(dict_in.keys()):
            if isinstance(dict_in[k], dict):
                for kk, vv in dict_in[k].items():
                    dict_out[f"{k}:{kk}"] = vv
            else:
                dict_out[k] = dict_in[k]

        return dict_out

    @staticmethod
    def _sampling(dist_params: dict, num_samples: int, randomise: bool = True) -> Union[float, np.ndarray]:
        """Evacuate sampled values based on a defined distribution. This is build upon `scipy.stats` library.

        :param dist_params: Distribution inputs, required keys are distribution dependent, should be aligned with inputs
                            required in the scipy.stats. Additional compulsory keys are:
                                `dist`: str, distribution type;
                                `ubound`: float, upper bound of the sampled values; and
                                `lbound`: float, lower bound of the sampled values.
        :param num_samples: Number of samples to be generated.
        :param randomise:   Whether to randomise the sampled values.
        :return samples:    Sampled values based upon `dist` in the range [`lbound`, `ubound`] with `num_samples` number
                            of values.
        """
        if dist_params['dist'] == 'constant_':
            return np.full((num_samples,), (dist_params['lbound'] + dist_params['ubound']) / 2, dtype=float)

        # sample CDF points (y-axis value)
        def generate_cfd_q(dist, dist_params_scipy, lbound, ubound):
            cfd_q_ = np.linspace(
                getattr(stats, dist).cdf(x=lbound, **dist_params_scipy),
                getattr(stats, dist).cdf(x=ubound, **dist_params_scipy),
                num_samples,
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
    @abstractmethod
    def input_keys(self) -> tuple:
        """Placeholder method. The input value names from the Monte Carlo Simulation deterministic calculation
        routine.
        :return:
        """
        raise NotImplementedError('This method should be overridden by a child class')

    def run(self, p: mp.Pool, set_progress: Optional[Callable] = None, progress_0: int = 0):
        kwargs = self.input.to_dict()
        args = list(zip(*[kwargs[k] for k in self.input_keys]))
        output = list()

        futures = {p.submit(self.worker, arg) for arg in zip(*[kwargs[k] for k in self.input_keys])}

        if set_progress is not None:
            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                output.append(future.result())
                set_progress(progress_0 + i)
        else:
            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                output.append(future.result())

        self.__output = np.array(output)

    @staticmethod
    @abstractmethod
    def worker(args) -> dict:
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

        print(np.amax(data), np.amin(data))
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
        assert os.path.exists(dir_save)
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
                self.__output = np.frombuffer(f_zip.read(f'{self.name}.csv'), delimiter=',', skip_header=1, )
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
            worksheet = open_workbook(r'C:\Users\IanFu\Desktop\pra-test\mcs0\mcs0.xls').sheet_by_index(0)

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
        raise NotImplementedError()

    def run(self, n_proc: int = 1, set_progress: Optional[Callable] = None, set_progress_max: Optional[Callable] = None,
            save: bool = False, save_archive: bool = False):

        if set_progress_max is not None:
            set_progress_max(sum([v.n_sim for k, v in self.mcs_cases.items()]))

        if save:
            self.save_init(archive=save_archive)

        progress_0 = None
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as executor:
            for mcs_case_name, mcs_case in self.mcs_cases.items():  # Reuse the executor for 3 sets of tasks
                progress_0 = 0 if progress_0 is None else progress_0 + mcs_case.n_sim
                mcs_case.run(executor, set_progress=set_progress, progress_0=progress_0)
                if save:
                    mcs_case.save_csv(archive=save_archive)

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
            mcs_case.load_output_from_file(self.__in_fp)
