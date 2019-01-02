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


def permuatation_list(a, b):

    if isinstance(a[0], list):
        a = [[i] for i in a]

    if isinstance(b[0], list):
        b = [[i] for i in b]

    r = []
    for i in a:
        for j in b:
            r.append(i+j)


if __name__ == "__main__":
    # print(distribute_numbers_cartesian_product([9]*3))

    a = [[1],[2],[3]]

    b = [[11],[22],[33]]

    r = permuatation_list(a, b)

    print(r)
