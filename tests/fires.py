# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')


def travelling_fire():
    from sfeprapy.func.temperature_fires import travelling_fire as fire
    import matplotlib.pyplot as plt
    import numpy as np
    time, temperature, data = fire(
        T_0=293.15,
        q_fd=900e6,
        RHRf=0.15e6,
        l=150,
        w=17.4,
        s=0.012,
        h_s=3.5,
        l_s=105,
        time_step=60,
        time_ubound=22080
    )
    fig, ax = plt.subplots()
    ax.plot(time/60, temperature-273.15)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Temperature [C]')
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(0, 1201, 100))
    plt.show()


def parametric_fire():
    import copy
    import numpy as np
    import matplotlib.pyplot as plt
    from sfeprapy.func.temperature_fires import parametric_eurocode1 as fire

    kwargs = {"A_t": 360,
              "A_f": 100,
              "A_v": 72,
              "h_eq": 1,
              "q_fd": 600e6,
              "lambda_": 1,
              "rho": 1,
              "c": 2250000,
              "t_lim": 20*60,
              "time_end": 2*60*60,
              "time_step": 1,
              "time_start": 0,
              "temperature_initial": 293.15}

    fig, ax = plt.subplots()

    for i in [72, 50.4, 36.000000001, 32.4, 21.6, 14.4, 7.2]:
        if i == 36:
            print("stop")
        kwargs["A_v"] = i
        x, y = fire(**copy.copy(kwargs))
        ax.plot(x/60, y-273.15, label="O={:03.2f}".format(kwargs["A_v"]*kwargs["h_eq"]**0.5/kwargs["A_t"]))

    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Temperature [C]')
    ax.set_xticks(np.arange(0, 121, 10))
    ax.set_yticks(np.arange(0, 1201, 100))
    ax.grid(True)
    plt.legend(loc=1)
    plt.show()
    return 0


def iso834():
    import pandas as pd
    from sfeprapy.func.temperature_fires import standard_fire_iso834 as fire
    import numpy as np
    t = np.arange(0, 30*60, 5)
    res = {}
    res["TIME"], res["TEMPERATURE"] = fire(t, 273.15)
    res["TEMPERATURE"] -= 273.15
    df = pd.DataFrame(res)
    df.to_csv("out.csv", index=False)


def t_square():
    import pandas as pd
    import numpy as np
    from sfeprapy.func.temperature_fires import t_square as fire

    growth = "fast"
    cap_time = 300

    dict_res = dict()
    dict_res["TIME [s]"] = np.arange(0, 30*60, 5)
    dict_res["HRR [kW]"] = fire(dict_res["TIME [s]"], growth, cap_hrr_to_time=cap_time) / 1000
    df = pd.DataFrame(dict_res)
    df = df[["TIME [s]", "HRR [kW]"]]
    df.to_csv("fire "+growth+str(cap_time)+".csv", index=False)


if __name__ == "__main__":
    travelling_fire()
    # parametric_fire()
    # t_square()

    pass
