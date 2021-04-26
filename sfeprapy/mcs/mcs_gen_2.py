# -*- coding: utf-8 -*-

from io import StringIO

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d


class DistParamsTrue2Scipy:
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

    def lognorm_mod_(self, mean: float, sd: float, **_):
        return self.lognorm_(mean, sd, **_)

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


class Params2Samples:
    @staticmethod
    def args2samples_dist(dist_params: dict, num_samples: int):
        """Generates samples of defined distribution. This is build upon scipy.stats library.

        :param dist_params: Distribution inputs, required keys are distribution dependent, should be align with inputs
                            required in the scipy.stats. Additional compulsory keys are:
                                `dist`: str, distribution type;
                                `ubound`: float, upper bound of the sampled values; and
                                `lbound`: float, lower bound of the sampled values.
        :param num_samples: Number of samples to be generated.
        :return samples:    Sampled values based upon `dist` in the range [`lbound`, `ubound`] with `num_samples` number
                            of values.
        """

        # assign distribution type
        dist_0: str = dist_params["dist"]
        dist = dist_params["dist"]

        # assign distribution boundary (for samples)
        ubound = dist_params["ubound"]
        lbound = dist_params["lbound"]

        # sample CDF points (y-axis value)
        def generate_cfd_q(dist_, dist_kw_, lbound_, ubound_):
            cfd_q_ = np.linspace(
                getattr(stats, dist_).cdf(x=lbound_, **dist_kw_),
                getattr(stats, dist_).cdf(x=ubound_, **dist_kw_),
                num_samples,
            )
            samples_ = getattr(stats, dist_).ppf(q=cfd_q_, **dist_kw_)
            return samples_

        # convert human distribution parameters to scipy distribution parameters
        try:
            if dist_0 == "constant_":
                samples = np.full((num_samples,), np.average([lbound, ubound]))
            elif dist_0 == 'lognorm_mod_':
                dist_kw = getattr(DistParamsTrue2Scipy, 'lognorm_')(**dist_params)
                samples = generate_cfd_q(
                    dist_='lognorm', dist_kw_=dist_kw, lbound_=lbound, ubound_=ubound
                )
                samples = 1 - samples
            else:
                dist_kw = getattr(DistParamsTrue2Scipy, dist_0)(**dist_params)
                samples = generate_cfd_q(
                    dist_=dist_0.rstrip('_'), dist_kw_=dist_kw, lbound_=lbound, ubound_=ubound
                )

        except Exception as e:
            try:
                dist_params.pop("dist")
                dist_params.pop("ubound")
                dist_params.pop("lbound")
                samples = generate_cfd_q(
                    dist_=dist, dist_kw_=dist_params, lbound_=lbound, ubound_=ubound
                )
            except AttributeError:
                raise ValueError(f"Unknown distribution type {dist}, {e}")

        samples[samples == np.inf] = ubound
        samples[samples == -np.inf] = lbound

        if "permanent" in dist_params:
            samples += dist_params["permanent"]

        np.random.shuffle(samples)

        return samples


class InputParser(Params2Samples):
    """Converts """

    def __init__(self):
        super().__init__()

    def inputs2samples(self, dist_params: dict, num_samples: int) -> pd.DataFrame:
        """Generates samples based upon prescribed distribution types.
    
        :param dist_params: description of distribution function.
        :param num_samples: number of samples to be produced.
        :return df_out:
        """

        dict_out = dict()

        for k, v in dist_params.items():

            if isinstance(v, float) or isinstance(v, int) or isinstance(v, np.float):
                dict_out[k] = np.full((num_samples,), v, dtype=float)

            elif isinstance(v, str):
                dict_out[k] = np.full(
                    (num_samples,), v, dtype=np.dtype("U{:d}".format(len(v)))
                )

            elif isinstance(v, np.ndarray) or isinstance(v, list):
                dict_out[k] = list(np.full((num_samples, len(v)), v, dtype=float))

            elif isinstance(v, dict):
                if "dist" in v:
                    try:
                        dict_out[k] = self.args2samples_dist(v, num_samples)
                    except KeyError:
                        raise ("Missing parameters in input variable {}.".format(k))
                elif "ramp" in v:
                    s_ = StringIO(v["ramp"])
                    d_ = pd.read_csv(
                        s_,
                        names=["x", "y"],
                        dtype=float,
                        skip_blank_lines=True,
                        skipinitialspace=True,
                    )
                    t_ = d_.iloc[:, 0]
                    v_ = d_.iloc[:, 1]
                    if all(v_ == v_[0]):
                        f_interp = v_[0]
                    else:
                        f_interp = interp1d(t_, v_, bounds_error=False, fill_value=0)
                    dict_out[k] = np.full((num_samples,), f_interp)
                else:
                    raise ValueError("Unknown input data type for {}.".format(k))
            else:
                raise TypeError("Unknown input data type for {}.".format(k))

        dict_out["index"] = np.arange(0, num_samples, 1)
        df_out = pd.DataFrame.from_dict(dict_out, orient="columns")

        return df_out

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


class TestInputParser:
    def __init__(self):
        self.test_cls = InputParser()

        self.inputs2samples()
        self.flatten_dict()
        self.unflatten_dict()

    def inputs2samples(self):
        dist_params = dict(
            string="hello world",
            number=10.,
            dist_uniform_=dict(dist="uniform_", lbound=0., ubound=100.),
            dist_gumbel_r_=dict(dist="gumbel_r_", lbound=10, ubound=1500, mean=420, sd=126),
            dist_norm_=dict(dist="norm_", lbound=623.15, ubound=2023.15, mean=1323.15, sd=93),
            dist_lognorm_mod_=dict(dist="lognorm_mod_", ubound=0.9999, lbound=0.0001, mean=0.2, sd=0.2),
        )
        df_out = self.test_cls.inputs2samples(dist_params=dist_params, num_samples=10)
        print(df_out)

    def unflatten_dict(self):
        x = {"A:a": 0, "A:b": 1, "B:c": 2, "B:d": 3}
        y = self.test_cls.unflatten_dict(x)
        y_expected = dict(A=dict(a=0, b=1), B=dict(c=2, d=3))
        print(y)
        print(y_expected)
        assert y == y_expected

    def flatten_dict(self):
        x = dict(A=dict(a=0, b=1), B=dict(c=2, d=3))
        y_expected = {"A:a": 0, "A:b": 1, "B:c": 2, "B:d": 3}
        y = self.test_cls.flatten_dict(x)
        print(y)
        print(y_expected)
        assert y == y_expected


if __name__ == "__main__":
    # TestInputParser()

    a = dict(
        a=1,
        b=dict(
            b1=21,
            b2=22,
        ),
        c=dict(
            c31=dict(
                c311=3111,
                c312=3112,
                c313=3113,
            )
        )
    )

    print(a)


    def test(dict_in):
        dict_out = dict()
        for k in list(dict_in.keys()):
            if isinstance(dict_in[k], dict):
                for kk, vv in dict_in[k].items():
                    dict_out[f"{k}:{kk}"] = vv
            else:
                dict_out[k] = dict_in[k]
        return dict_out


    print(test(a))
