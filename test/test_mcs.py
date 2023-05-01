from sfeprapy.mcs import *


def test_input_parser_flatten():
    x = dict(A=dict(a=0, b=1), B=dict(c=2, d=3))
    y_expected = {"A:a": 0, "A:b": 1, "B:c": 2, "B:d": 3}
    y = InputParser.flatten_dict(x)
    assert y == y_expected


def test_input_parser_unflatten():
    x = {"A:a": 0, "A:b": 1, "B:c": 2, "B:d": 3}
    y_expected = dict(A=dict(a=0, b=1), B=dict(c=2, d=3))
    y = InputParser.unflatten_dict(x)
    assert y == y_expected


def test_input_parser_sampling():
    y = InputParser(dict(v=np.pi), 1000).to_dict()
    assert len(y["v"]) == 1000
    assert all([v == np.pi for v in y["v"]])

    y = InputParser(dict(v="hello world."), 1000).to_dict()
    assert len(y["v"]) == 1000
    assert all([v == "hello world." for v in y["v"]])

    y = InputParser(dict(v=[0.0, 1.0, 2.0]), 1000).to_dict()
    assert len(y["v"]) == 1000
    assert all([all(v == np.array([0.0, 1.0, 2.0])) for v in y["v"]])

    y = InputParser(dict(v=dict(dist="uniform_", ubound=10, lbound=-1)), 1000).to_dict()
    assert len(y["v"]) == 1000
    assert np.max(y["v"]) == 10
    assert np.min(y["v"]) == -1
    assert abs(np.mean(y["v"]) - (10 - 1) / 2) <= 0.00001

    y = InputParser(dict(v=dict(dist="norm_", ubound=5 + 1, lbound=5 - 1, mean=5, sd=1)), 1000).to_dict()
    assert len(y["v"]) == 1000
    assert np.max(y["v"]) == 6
    assert np.min(y["v"]) == 4
    assert abs(np.mean(y["v"]) - 5) <= 0.00001

    y = InputParser(dict(v=dict(dist="gumbel_r_", ubound=2000, lbound=50, mean=420, sd=126)), 1000).to_dict()
    assert len(y["v"]) == 1000
    assert abs(np.max(y["v"]) - 2000) <= 1
    assert abs(np.min(y["v"]) - 50) <= 1
    assert abs(np.mean(y["v"]) - 420) <= 1

    InputParser(dict(
        string="hello world",
        number=10.,
        dist_uniform_=dict(dist="uniform_", lbound=0., ubound=100.),
        dist_gumbel_r_=dict(dist="gumbel_r_", lbound=10, ubound=1500, mean=420, sd=126),
        dist_norm_=dict(dist="norm_", lbound=623.15, ubound=2023.15, mean=1323.15, sd=93),
        dist_lognorm_mod_=dict(dist="lognorm_mod_", ubound=0.9999, lbound=0.0001, mean=0.2, sd=0.2),
    ), n=1000).to_dict()


if __name__ == '__main__':
    test_input_parser_flatten()
    test_input_parser_unflatten()
    test_input_parser_sampling()
