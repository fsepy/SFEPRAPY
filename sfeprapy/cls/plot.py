# -*- coding: utf-8 -*-
# AUTHOR: YAN FU
# VERSION: 0.1
# DATE: 09/01/2018

# import numpy as np
# import os, time, matplotlib, inspect
# import matplotlib.pyplot as plt
# from inspect import currentframe, getframeinfo  # for error handling, get current line number
# from scipy.interpolate import interp1d


class Scatter2D(object):
    def __init__(self, fixed_data_points = 0, fixed_data_points_range = (0, 0)):
        pass

    def self_delete(self):
        pass

    def plot(self, xyl1, xyl2=None):
        pass

    def plot2(self, x, y, label="", second_axis=False, alpha=1.):
        pass

    def plot_vertical_line(self, x, line_color="black", line_linewidth=1):
        pass

    def plot_horizontal_line(self, y, line_color="black", line_linewidth=1):
        pass

    def format(self, **kwargs):
        pass

    def format_figure(self,
                      figure_size_width=8.,
                      figure_size_height=6.,
                      figure_size_scale=1.,
                      figure_title="",
                      figure_title_font_size=15.):

        pass

    def format_axes(self,
                    axis_label_x="",
                    axis_label_y1="",
                    axis_label_y2="",
                    axis_label_font_size=9.,
                    axis_tick_font_size=8.,
                    axis_lim_x=None,
                    axis_lim_y1=None,
                    axis_lim_y2=None,
                    axis_linewidth=1.,
                    axis_scientific_format_x=False,
                    axis_scientific_format_y1=False,
                    axis_scientific_format_y2=False,
                    axis_tick_width=.5,
                    axis_tick_length=2.5,
                    axis_xtick_major_loc=None,
                    axis_xtick_minor_loc=None,
                    axis_ytick_major_loc=None,
                    axis_ytick_minor_loc=None,
                    axis_grid_show=False,
                    axis_grid_linestyle = "--",
                    axis_grid_linewidth = 0.25,
                    axis_grid_linecolour = "black",):
        pass

    def format_lines(self,
                     marker_size=3,
                     mark_every=100,
                     marker_fill_style="none",
                     marker_edge_width=.5,
                     line_width=1.,
                     line_style="-",
                     line_colours=None,
                     line_alpha=0):

        pass

    def format_legend(self,
                      legend_is_shown=True,
                      legend_loc=0,
                      legend_font_size=8,
                      legend_colour="black",
                      legend_alpha=1.0,
                      legend_is_fancybox=False,
                      legend_line_width=1.):

        pass

    def add_lines(self, xyl, axis=0):
        pass

    def update_legend(self, **kwargs):
       pass

    def update_line_format(self, line_name, **kwargs):
        pass

    def add_text(self, x, y, s, va="center", ha="center", fontsize=6):
        pass

    def update_line(self, line_name, x, y, label=None):
        pass

    def remove_line(self, line_name):
        pass

    # def adjust_text(self):
    #
    #     # adjust_text(self._texts_added, arrowprops=dict(arrowstyle="->", color='r', lw=0.5),
    #     #             autoalign='xy',
    #     #             # only_move={'points': 'y', 'text': 'y'},
    #     #             expand_points=(5, 5),
    #     #             force_points=0.1)
    #
    #     adjust_text(self._texts_added,
    #                 # x=None,
    #                 # y=None,
    #                 # add_objects=None,
    #                 # ax=None,
    #                 expand_text=(1.2, 2.2),
    #                 expand_points=(5.2, 5.2),
    #                 expand_objects=(1.2, 1.2),
    #                 expand_align=(0.9, 0.9),
    #                 autoalign='xy',
    #                 va='center',
    #                 ha='center',
    #                 force_text=0.5,
    #                 force_points=0.5,
    #                 force_objects=0.5,
    #                 lim=100,
    #                 precision=0,
    #                 only_move={},
    #                 text_from_text=True,
    #                 text_from_points=True,
    #                 save_steps=False,
    #                 save_prefix='',
    #                 save_format='png',
    #                 add_step_numbers=True,
    #                 draggable=True,
    #                 on_basemap=False,
    #                 arrowprops=dict(arrowstyle="->", color='b', lw=0.5),)

    def save_figure2(self, path_file, dpi=300):
        pass

    def show(self):
        pass

    @property
    def figure(self):
        return None

    @property
    def axes_primary(self):
        return None

    @property
    def axes_secondary(self):
        return None


if __name__ == "__main__":
    # x = np.arange(-2*np.pi, 2*np.pi, 0.01)
    # y_sin = np.sin(x)
    # y_cos = np.cos(x)
    # y_tan = np.tan(x)
    #
    # p = Scatter2D()
    # p.plot2(x, y_sin, 'sin(x)')
    # p.plot2(x, y_cos, 'cos(x)')
    # p.plot2(x, y_tan, 'tan(x)')
    # p.plot2(x+np.pi/2, y_sin, 'sin(x+0.5pi)', second_axis=True)
    #
    # # default format
    # plt_format = {
    #     'figure_size_width': 8.,
    #     'figure_size_height': 6.,
    #     'figure_size_scale': 1.,
    #     'figure_title': "",
    #     'figure_title_font_size': 15.,
    #     'axis_label_x': "",
    #     'axis_label_y1': "",
    #     'axis_label_y2': "",
    #     'axis_label_font_size': 9.,
    #     'axis_tick_font_size': 8.,
    #     'axis_lim_x': None,
    #     'axis_lim_y1': None,
    #     'axis_lim_y2': None,
    #     'axis_linewidth': 1.,
    #     'axis_scientific_format_x': False,
    #     'axis_scientific_format_y1': False,
    #     'axis_scientific_format_y2': False,
    #     'axis_tick_width': .5,
    #     'axis_tick_length': 2.5,
    #     'axis_xtick_major_loc': None,
    #     'axis_xtick_minor_loc': None,
    #     'axis_ytick_major_loc': None,
    #     'axis_ytick_minor_loc': None,
    #     'axis_grid_show': True,
    #     'axis_grid_linestyle': "-",
    #     'axis_grid_linewidth': 0.5,
    #     'axis_grid_linecolour': "black",
    #     'marker_size': 3,
    #     'mark_every': 100,
    #     'marker_fill_style': "none",
    #     'marker_edge_width': .5,
    #     'line_width': 1.,
    #     'line_style': "-",
    #     'line_colours': None,
    # }
    #
    # # re-define some of the values
    # plt_format_ = {
    #     'axis_lim_x': (0., 2*np.pi),
    #     'axis_lim_y1': (-1., 1.),
    #     'axis_lim_y2': (-2., 2.),
    # }
    #
    # # update format dict
    # plt_format.update(plt_format_)
    #
    # p.format(**plt_format)
    # # p.save_figure(r"C:\hello")
    # figure_file_path = os.path.abspath(r'hello.png')
    # figure_file_path = os.path.realpath(figure_file_path)
    # p.save_figure2(figure_file_path)
    pass
