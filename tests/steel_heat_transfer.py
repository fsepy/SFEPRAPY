
import matplotlib
matplotlib.use('TkAgg')

def protected_steel():
    from sfeprapy.func.temperature_steel_section import protected_steel_eurocode as steel
    from sfeprapy.func.temperature_fires import standard_fire_iso834 as fire
    from sfeprapy.dat.steel_carbon import Thermal
    import matplotlib.pyplot as plt
    import numpy as np

    steel_prop = Thermal()
    c = steel_prop.c()
    rho = 7850
    t, T = fire(np.arange(0,10080,60), 20+273.15)

    for d_p in np.arange(0.001, 0.2+0.02, 0.02):
        t_s, T_s, d = steel(
            time=t,
            temperature_ambient=T,
            rho_steel=rho,
            c_steel_T=c,
            area_steel_section=0.017,
            k_protection=0.2,
            rho_protection=800,
            c_protection=1700,
            thickness_protection=d_p,
            perimeter_protected=2.14
        )
        plt.plot(t_s, T_s, label="d_p={:5.3f}".format(d_p))

    plt.legend(loc=1)
    plt.show()


def protected_steel2():
    """
    This test is for protected steel in eurocode under parametric fire curve.
    """
    import copy
    import matplotlib.pyplot as plt
    from sfeprapy.func.temperature_fires import parametric_eurocode1 as fire
    from sfeprapy.func.temperature_steel_section import protected_steel_eurocode as steel
    from sfeprapy.dat.steel_carbon import Thermal

    steel_prop = Thermal()
    c = steel_prop.c()
    rho = steel_prop.rho()

    kwargs_fire = {"A_t": 360,
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
    time_fire, temp_fire = fire(**kwargs_fire)

    kwargs_steel = {
        "time": time_fire,
        "temperature_ambient": temp_fire,
        "rho_steel_T": rho,
        "c_steel_T": c,
        "area_steel_section": 0.017,
        "k_protection": 0.2,
        "rho_protection": 800,
        "c_protection": 1700,
        "thickness_protection": 0.01,
        "perimeter_protected": 2.14
    }
    time_steel, temp_steel, c = steel(**kwargs_steel)

    plt.plot(time_fire, temp_fire)
    plt.plot(time_steel, temp_steel)
    plt.show()

    # for i in [72,50.4,36.000000001,32.4,21.6,14.4,7.2]:
    #     if i == 36:
    #         print("stop")
    #     kwargs_fire["A_v"] = i
    #     x, y = fire(**copy.copy(kwargs_fire))
    #     plt.plot(x/60, y-273.15, label="O={:03.2f}".format(kwargs_fire["A_v"]*kwargs_fire["h_eq"]**0.5/kwargs_fire["A_t"]))
    # plt.legend(loc=1)
    # plt.show()


if __name__ == "__main__":
    protected_steel()
    # protected_steel2()

    pass
