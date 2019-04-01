# -*- coding: utf-8 -*-
import numpy as np
from sfeprapy.func.fire_iso834 import fire

if __name__ == "__main__":
    time = np.arange(0, 2*60*60, 30)
    temperature_cap = 450

    temperature = fire(time=np.arange(0, 2*60*60+1), temperature_initial=20+273.15)

    temperature -= 273.15
    temperature[temperature > temperature_cap] = temperature_cap

    import seaborn as sb
    import matplotlib.pyplot as plt

    sb.scatterplot(time, temperature)
    plt.show()

    time_temperature = list(zip(time, temperature))

    print(time_temperature)

    time_temperature = ['{:.0f},{:.1f}'.format(*i) for i in time_temperature]

    with open('o.txt', 'w+') as f:
        f.write('\n'.join(time_temperature))
    # t_square()
