# -*- coding: utf-8 -*-
import os
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='ticks')

from sfeprapy.dat.steel_carbon import Thermal
from sfeprapy.func.temperature_fires import standard_fire_iso834 as _fire_iso
from sfeprapy.func.temperature_fires import parametric_eurocode1 as _fire_param
from sfeprapy.func.temperature_fires import travelling_fire as _fire_travelling
from sfeprapy.func.temperature_steel_section import protected_steel_eurocode as _steel_heat_transfer

# try:
#     from project.func.temperature_fires import standard_fire_iso834 as _fire_standard
# except ModuleNotFoundError:
#     import sys
#
#     __path_cwf = os.path.realpath(__file__)
#     __dir_project = os.path.dirname(os.path.dirname(__path_cwf))
#
#     sys.path.insert(0, __dir_project)
#
#     from project.func.temperature_fires import standard_fire_iso834 as _fire_standard
#
# from project.func.temperature_fires import standard_fire_iso834 as _fire_iso
# from project.func.temperature_fires import parametric_eurocode1 as _fire_param
# from project.func.temperature_fires import travelling_fire as _fire_travelling
# from project.dat.steel_carbon import Thermal
# from project.func.temperature_steel_section import protected_steel_eurocode as _steel_heat_transfer


def main():
    print('hello world')

    t = np.arange(0, 60*180, 1, dtype=float)
    steel_prop = Thermal()

    # PARAMETRIC FIRE
    inputs_parametric_fire = {
        "A_t": 360,
        "A_f": 100,
        "A_v": 21.6,
        "h_eq": 1,
        "q_fd": 600e6,
        "lambda_": 1,
        "rho": 1,
        "c": 2250000,
        "t_lim": 20 * 60,
        "time_end": 3 * 60 * 60,
        "time_step": 1,
        "time_start": 0,
        "temperature_initial": 273.15
        }
    time_fire_param, temp_fire_param = _fire_param(**inputs_parametric_fire)

    time_steel_param, temp_steel_param, _ = _steel_heat_transfer(
        time=time_fire_param,
        temperature_ambient=temp_fire_param,
        rho_steel=7850,
        c_steel_T=steel_prop.c(),
        area_steel_section=0.017,
        k_protection=0.2,
        rho_protection=800,
        c_protection=1700,
        thickness_protection=0.00938,
        perimeter_protected=2.14,
        is_terminate_peak=False
    )

    # TRAVELLING FIRE

    inputs_travelling_fire = {
        "T_0": 273.15,
        "q_fd": 600e6,
        "RHRf": 0.15e6,
        "l": 25,
        "w": 17.4,
        "s": 0.012,
        "h_s": 4.5,
        "l_s": 20,
        "time_step": 1,
        "time_ubound": 22080
    }
    time_fire_travel, temp_fire_travel, _ = _fire_travelling(**inputs_travelling_fire)

    time_steel_travel, temp_steel_travel, _ = _steel_heat_transfer(
        time=time_fire_travel,
        temperature_ambient=temp_fire_travel,
        rho_steel=7850,
        c_steel_T=steel_prop.c(),
        area_steel_section=0.017,
        k_protection=0.2,
        rho_protection=800,
        c_protection=1700,
        thickness_protection=0.01007,
        perimeter_protected=2.14,
        is_terminate_peak=False
    )

    # MAKE STEEL TEMPERATURE FOR PARAMETRIC FIRE


    # print(np.max(temp_steel_param)-273.15)
    # plt.plot(time_fire_param/60, temp_fire_param)
    # plt.plot(time_steel_param/60, temp_steel_param)
    # plt.show()

    # MAKE STEEL TEMPERATURE FOR TRAVELLING FIRE
    # print(np.max(temp_steel_travel)-273.15)
    # plt.plot(time_fire_travel/60, temp_fire_travel)
    # plt.plot(time_steel_travel/60, temp_steel_travel)
    # plt.show()

    # MAKE ISO 834 FIRE
    time_fire_iso, temp_fire_iso = _fire_iso(np.arange(0, 21601, 1), 20 + 273.15)

    time_steel_iso_param, temp_steel_iso_param, _ = _steel_heat_transfer(
        time=time_fire_iso,
        temperature_ambient=temp_fire_iso,
        rho_steel=7850,
        c_steel_T=steel_prop.c(),
        area_steel_section=0.017,
        k_protection=0.2,
        rho_protection=800,
        c_protection=1700,
        thickness_protection=0.00938,
        perimeter_protected=2.14,
        is_terminate_peak=False
    )

    time_steel_iso_travel, temp_steel_iso_travel, _ = _steel_heat_transfer(
        time=time_fire_iso,
        temperature_ambient=temp_fire_iso,
        rho_steel=7850,
        c_steel_T=steel_prop.c(),
        area_steel_section=0.017,
        k_protection=0.2,
        rho_protection=800,
        c_protection=1700,
        thickness_protection=0.01007,
        perimeter_protected=2.14,
        is_terminate_peak=False
    )

    # fig, ax = plt.subplots(figsize=(2.5, 2))
    fig, ax = plt.subplots(figsize=(3.94, 2.76))

    def mini_plot(fig_, ax_, x, y, fig_name, xlabel='Time', ylabel='Temperature', xlim=(0, 120), ylim=(0, 1400)):
        ax_.plot(x, y)
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel(ylabel)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.set_xticks([])
        ax_.set_yticks([])
        fig_.tight_layout()
        fig_.savefig(fig_name, ppi=300)
        ax_.cla()

    def mini_plot2(fig_, ax_, x, y, fig_name, xlabel='Time', ylabel='Temperature', xlim=(0, 120), ylim=(0, 1400)):
        for i, xx in enumerate(x):
            yy = y[i]
            ax_.plot(xx, yy)
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel(ylabel)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.set_xticks([])
        ax_.set_yticks([])
        fig_.tight_layout()
        fig_.savefig(fig_name, ppi=300)
        ax_.cla()

    def mini_plot3(fig_, ax_, x, y, legend, fig_name, xlabel='Time', ylabel='Temperature', xlim=(0, 120), ylim=(0, 1400)):
        for i, xx in enumerate(x):
            yy = y[i]
            l = legend[i]
            ax_.plot(xx, yy, lable=l)
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel(ylabel)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.set_xticks([])
        ax_.set_yticks([])
        fig_.tight_layout()
        fig_.savefig(fig_name, ppi=300)
        ax_.cla()

    mini_plot(fig, ax, time_fire_param/60., temp_fire_param - 273.15, 'fire_param.png')

    mini_plot(fig, ax, time_fire_iso/60., temp_fire_iso - 273.15, 'fire_iso.png')

    mini_plot(fig, ax, time_fire_travel/60., temp_fire_travel - 273.15, 'fire_travel.png')

    # mini_plot2(fig, ax,
    #            [time_fire_param/60., time_steel_param/60.],
    #            [temp_fire_param - 273.15, temp_steel_param-273.5],
    #            'steel_param.png')

    # mini_plot2(fig, ax,
    #           [time_fire_travel/60., time_steel_travel/60.],
    #           [temp_fire_travel - 273.15, temp_steel_travel-273.5],
    #           'steel_travel.png')

    # mini_plot2(fig, ax,
    #            [time_fire_iso/60., time_steel_iso_travel/60.],
    #            [temp_fire_iso - 273.15, temp_steel_iso_travel-273.5],
    #            'steel_iso.png')

    # mini_plot2(fig, ax,
    #            [time_fire_iso/60., time_steel_iso_travel/60.],
    #            [temp_fire_iso - 273.15, temp_steel_iso_travel-273.15],
    #            'steel_iso_travel.png')

    # mini_plot2(fig, ax,
    #            [time_fire_travel / 60., time_steel_travel / 60.],
    #            [temp_steel_travel - 273.15, temp_steel_iso_travel - 273.15],
    #            'steel_iso_travel.png')

    mini_plot2(fig, ax,
               [time_fire_param/60., time_fire_travel/60.],
               [temp_fire_param - 273.15, temp_fire_travel - 273.15],
               ['Parametric fire', 'Travelling fire'],
               'fire_param_and_travel.png')

    # plt.plot(time_fire_iso/60, temp_fire_iso)
    # plt.plot(time_steel_iso_param/60, temp_steel_iso_param, '--')
    # plt.plot(time_fire_param/60, temp_fire_param)
    # plt.plot(time_steel_param/60, temp_steel_param, '--')
    # plt.show()

    # plt.plot(time_fire_iso/60, temp_fire_iso)
    # plt.plot(time_steel_iso_travel/60, temp_steel_iso_travel, '--')
    # plt.plot(time_fire_travel/60, temp_fire_travel)
    # plt.plot(time_steel_travel/60, temp_steel_travel, '--')
    # plt.show()

    # MAKE STEEL TEMPERATURE FOR ISO 834 FIRE
