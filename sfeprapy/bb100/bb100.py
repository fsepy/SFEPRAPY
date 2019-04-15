# -*- coding: utf-8 -*-


def door_escape_capacity_bb100(width_mm):

    if width_mm < 750:
        exit_capacity = 0
    elif width_mm < 850:
        exit_capacity = 60
    elif width_mm < 1050:
        exit_capacity = 110
    elif width_mm == 1050:
        exit_capacity = 220
    else:
        exit_capacity = int((width_mm - 1050)/50) * 10 + 220

    return exit_capacity


if __name__ == '__main__':

    w = 1730

    print(door_escape_capacity_bb100(w))
