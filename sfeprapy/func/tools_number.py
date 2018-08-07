# -*- coding: utf-8 -*-
import numpy as np
import itertools


def distribute_numbers_cartesian_product(n):
    b = []
    for v in n:
        a = np.linspace(0, 1, v+2)
        a = a[0:-2] + a[1]
        b.append(a)

    c = itertools.product(*b)
    c = list(c)
    c = np.asarray(c)

    return c


if __name__ == "__main__":
    print(distribute_numbers_cartesian_product([9]*3))
