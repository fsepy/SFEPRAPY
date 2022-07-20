from sfeprapy.mcs.mcs_gen import *


def test_dict_flatten():
    x = dict(A=dict(a=0, b=1), B=dict(c=2, d=3))
    y_expected = {"A:a": 0, "A:b": 1, "B:c": 2, "B:d": 3}
    y = dict_flatten(x)
    assert y == y_expected


def test_random_variable_generator():
    x = dict(v=np.pi)
    y = main(x, 1000)
    assert len(y["v"].values) == 1000
    assert all([v == np.pi for v in y["v"].values])

    x = dict(v="hello world.")
    y = main(x, 1000)
    assert len(y["v"].values) == 1000
    assert all([v == "hello world." for v in y["v"].values])

    x = dict(v=[0.0, 1.0, 2.0])
    y = main(x, 1000)
    assert len(y["v"].values) == 1000
    assert all([all(v == np.array([0.0, 1.0, 2.0])) for v in y["v"].values])

    x = dict(v=dict(dist="uniform_", ubound=10, lbound=-1))
    y = main(x, 1000)
    assert len(y["v"].values) == 1000
    assert np.max(y["v"].values) == 10
    assert np.min(y["v"].values) == -1
    assert abs(np.mean(y["v"].values) - (10 - 1) / 2) <= 0.00001

    x = dict(v=dict(dist="norm_", ubound=5 + 1, lbound=5 - 1, mean=5, sd=1))
    y = main(x, 1000)
    assert len(y["v"].values) == 1000
    assert np.max(y["v"].values) == 6
    assert np.min(y["v"].values) == 4
    assert abs(np.mean(y["v"].values) - 5) <= 0.00001

    x = dict(v=dict(dist="gumbel_r_", ubound=2000, lbound=50, mean=420, sd=126))
    y = main(x, 1000)
    assert len(y["v"].values) == 1000
    assert abs(np.max(y["v"].values) - 2000) <= 1
    assert abs(np.min(y["v"].values) - 50) <= 1
    assert abs(np.mean(y["v"].values) - 420) <= 1


if __name__ == '__main__':
    test_dict_flatten()
    test_random_variable_generator()
