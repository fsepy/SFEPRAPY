# -*- coding: utf-8 -*-
# AUTHOR: YAN FU
# VERSION: 0.1
# DATE: 09/01/2018

import numpy as np
import os, time, matplotlib, inspect
import matplotlib.pyplot as plt
from inspect import currentframe, getframeinfo  # for error handling, get current line number
from scipy.interpolate import interp1d


class Scatter2D(object):
    def __init__(self, fixed_data_points = 0, fixed_data_points_range = (0, 0)):
        # todo: add plot vertical and horizontal lines feature
        # todo: re-think the overall logic

        self._figure = plt.figure()
        self._axes = []
        self._lines = []
        self._vlines = []
        self._hlines = []
        self._texts = []
        self._texts_added = []
        self._format = None

        self.__data_points = fixed_data_points
        self.__data_points_range = fixed_data_points_range

    def self_delete(self):
        self._figure.clf()
        plt.close(self._figure)
        # self._figure = None
        # self._axes = None
        # self._lines = None
        # self._texts = None
        del self

    def plot(self, xyl1, xyl2=None):
        # create _axes
        self._axes.append(self._figure.add_subplot(111))
        self._axes.append(self._axes[0].twinx()) if xyl2 is not None else None

        # create _lines
        for i in tuple(xyl1):
            x, y, l = tuple(i)
            line = self._axes[0].plot(x, y, label=l)
            self._lines.append(line[0])
        if xyl2 is not None:
            for i in xyl2:
                x, y, l = tuple(i)
                line = self._axes[1].plot(x, y, label=l)
                self._lines.append(line[0])

    def plot2(self, x, y, label="", second_axis=False, alpha=1.):

        # DATA POINTS NORMALISATION
        # =========================

        # Check if set by user

        if self.__data_points > 0:

            # interpolate new coordinates

            f_x = interp1d(x, y)

            if self.__data_points_range == (0, 0):
                x_new = np.linspace(np.min(x), np.max(x), self.__data_points)
            else:
                x_new = np.linspace(self.__data_points_range[0], self.__data_points_range[1], self.__data_points)

            y_new = f_x(x_new)

        else:
            x_new, y_new = x, y

        # PLOT
        # ====

        if len(self._axes) == 0:
            self._axes.append(self._figure.add_subplot(111))

        if second_axis:
            self._axes.append(self._axes[0].twinx())
            line = self._axes[1].plot(x_new, y_new, label=label, alpha=alpha)
        else:
            line = self._axes[0].plot(x_new, y_new, label=label, alpha=alpha)

        self._lines.append(line[0])

    def plot_vertical_line(self, x, line_color="black", line_linewidth=1):
        l = self._axes[0].axvline(x=x, color=line_color, linewidth=line_linewidth)
        self._vlines.append(l)

    def plot_horizontal_line(self, y, line_color="black", line_linewidth=1):
        l = self._axes[0].axhline(y=y, color=line_color, linewidth=line_linewidth)
        self._hlines.append(l)

    def format(self, **kwargs):
        def map_dictionary(list_, dict_master):
            dict_new = dict()
            for key in list_:
                if key in dict_master:
                    dict_new[key] = dict_master[key]
            return dict_new

        dict_inputs_figure = map_dictionary(inspect.signature(self.format_figure).parameters, kwargs)
        dict_inputs_axes = map_dictionary(inspect.signature(self.format_axes).parameters, kwargs)
        dict_inputs_lines = map_dictionary(inspect.signature(self.format_lines).parameters, kwargs)
        dict_inputs_legend = map_dictionary(inspect.signature(self.format_legend).parameters, kwargs)

        # set format
        self.format_figure(**dict_inputs_figure)
        self.format_axes(**dict_inputs_axes)
        self.format_lines(**dict_inputs_lines)
        self.format_legend(**dict_inputs_legend)

        self._figure.tight_layout()

    def format_figure(self,
                      figure_size_width=8.,
                      figure_size_height=6.,
                      figure_size_scale=1.,
                      figure_title="",
                      figure_title_font_size=15.):

        self._figure.set_size_inches(w=figure_size_width * figure_size_scale, h=figure_size_height * figure_size_scale)
        self._figure.suptitle(figure_title, fontsize=figure_title_font_size)
        self._figure.set_facecolor((1 / 237., 1 / 237., 1 / 237., 1.0))

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
        has_secondary = len(self._axes) > 1

        self._axes[0].set_xlim(axis_lim_x)
        self._axes[0].set_ylim(axis_lim_y1)
        self._axes[1].set_ylim(axis_lim_y2) if has_secondary else None
        self._axes[0].set_xlabel(axis_label_x, fontsize=axis_label_font_size)
        self._axes[0].set_ylabel(axis_label_y1, fontsize=axis_label_font_size)
        self._axes[1].set_ylabel(axis_label_y2, fontsize=axis_label_font_size) if has_secondary else None
        self._axes[0].get_xaxis().get_major_formatter().set_useOffset(axis_scientific_format_x)
        self._axes[0].get_yaxis().get_major_formatter().set_useOffset(axis_scientific_format_y1)
        self._axes[1].get_yaxis().get_major_formatter().set_useOffset(axis_scientific_format_y2) if has_secondary else None

        [i.set_linewidth(axis_linewidth) for i in self._axes[0].spines.values()]
        [i.set_linewidth(axis_linewidth) for i in self._axes[1].spines.values()] if has_secondary else None

        self._axes[0].tick_params(axis='both', which='major', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in')
        self._axes[0].tick_params(axis='both', which='minor', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in')
        self._axes[1].tick_params(axis='both', which='major', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in') if has_secondary else None
        self._axes[1].tick_params(axis='both', which='minor', labelsize=axis_tick_font_size, width=axis_tick_width, length=axis_tick_length, direction='in') if has_secondary else None

        if axis_xtick_major_loc is not None:
            self._axes[0].set_xticks(axis_xtick_major_loc)
        if axis_xtick_minor_loc is not None:
            self._axes[0].set_xticks(axis_xtick_minor_loc, minor=True)

        if axis_ytick_major_loc is not None:
            self._axes[0].set_yticks(axis_ytick_major_loc)
        if axis_ytick_minor_loc is not None:
            self._axes[0].set_yticks(axis_xtick_minor_loc, minor=True)

        if axis_grid_show:
            self._axes[0].grid(axis_grid_show, linestyle=axis_grid_linestyle, linewidth=axis_grid_linewidth, color=axis_grid_linecolour)
        else:
            self._axes[0].grid(axis_grid_show)


        # tick_lines = self._axes[0].get_xticklines() + self._axes[0].get_yticklines()
        # [line.set_linewidth(3) for line in tick_lines]
        #
        # tick_labels = self._axes[0].get_xticklabels() + self._axes[0].get_yticklabels()
        # [label.set_fontsize("medium") for label in tick_labels]

    def format_lines(self,
                     marker_size=3,
                     mark_every=100,
                     marker_fill_style="none",
                     marker_edge_width=.5,
                     line_width=1.,
                     line_style="-",
                     line_colours=None,
                     line_alpha=0):

        if line_colours is None:
            c = [(80, 82, 199), (30, 206, 214), (179, 232, 35), (245, 198, 0), (255, 89, 87)]
            c = [(colour[0] / 255., colour[1] / 255., colour[2] / 255.) for colour in c] * 100
        else:
            c = line_colours * 500

        m = ['o', '^', 's', 'v', 'p', '*', 'D', 'd', '8', '1', 'h', '+', 'H'] * 40

        for i, line in enumerate(self._lines):
            line.set_marker(m[i])
            line.set_color(c[i])
            line.set_markersize(marker_size)
            line.set_markevery(mark_every)
            line.set_markeredgecolor(c[i])
            line.set_markeredgewidth(marker_edge_width)
            line.set_fillstyle(marker_fill_style)
            line.set_linestyle(line_style)
            line.set_linewidth(line_width)

    def format_legend(self,
                      legend_is_shown=True,
                      legend_loc=0,
                      legend_font_size=8,
                      legend_colour="black",
                      legend_alpha=1.0,
                      legend_is_fancybox=False,
                      legend_line_width=1.):

        line_labels = [l.get_label() for l in self._lines]
        legend = self._axes[len(self._axes) - 1].legend(
            self._lines,
            line_labels,
            loc=legend_loc,
            fancybox=legend_is_fancybox,
            prop={'size': legend_font_size}
            )
        legend.set_visible(legend_is_shown)
        legend.get_frame().set_alpha(legend_alpha)
        legend.get_frame().set_linewidth(legend_line_width)
        legend.get_frame().set_edgecolor(legend_colour)

        self._texts.append(legend)

    def add_lines(self, xyl, axis=0):
        for i in xyl:
            x, y, l = tuple(i)
            line = self._axes[axis].plot(x, y, label=l)
            self._lines.append(line[0])

    def update_legend(self, **kwargs):
        """
        refresh the legend to the existing recent plotted _lines.
        """
        self._texts[0].remove()
        self.format_legend(**kwargs)

    def update_line_format(self, line_name, **kwargs):
        lines_index = {}
        for i,v in enumerate(self._lines):
            lines_index.update({v.get_label(): i})
        i = lines_index[line_name] if line_name in lines_index else None

        if i is None:
            frame_info = getframeinfo(currentframe())
            print("ERROR: {}; LINE: {:d}; FILE: {}".format("Line name does not exist", frame_info.lineno, frame_info.filename))
            return None

        line = self._lines[i]

        line_style = line.get_linestyle() if 'line_style' not in kwargs else kwargs['line_style']
        line_width = line.get_linewidth() if 'line_width' not in kwargs else kwargs['line_width']
        color = line.get_color() if 'color' not in kwargs else kwargs['color']
        marker = line.get_marker() if 'marker' not in kwargs else kwargs['marker']
        marker_size = line.get_markersize() if 'marker_size' not in kwargs else kwargs['marker_size']
        mark_every = line.get_markevery() if 'mark_every' not in kwargs else kwargs['mark_every']
        marker_edge_color = line.get_markeredgecolor() if 'marker_edge_color' not in kwargs else kwargs['marker_edge_color']
        marker_edge_width = line.get_markeredgewidth() if 'marker_edge_width' not in kwargs else kwargs['marker_edge_width']
        marker_fill_style = line.get_fillstyle() if 'marker_fill_style' not in kwargs else kwargs['marker_fill_style']

        line.set_linestyle(line_style)
        line.set_linewidth(line_width)
        line.set_color(color)
        line.set_marker(marker)
        line.set_markersize(marker_size)
        line.set_markevery(mark_every)
        line.set_markeredgecolor(marker_edge_color)
        line.set_markeredgewidth(marker_edge_width)
        line.set_fillstyle(marker_fill_style)

    def add_text(self, x, y, s, va="center", ha="center", fontsize=6):
        text_ = self._axes[0].text(x=x, y=y, s=s, va=va, ha=ha, fontsize=6)
        self._texts_added.append(text_)
        # self.adjust_text()

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

    def __save_figure(self, file_name="_figure", file_format=".pdf", name_prefix="", name_suffix="", dir_folder="", dpi=300):
        # WARNING: DEPRECIATED!!!
        time_suffix = False
        str_time = time.strftime("%m%d.%H%M%S")
        if name_suffix == "time":
            name_suffix = str_time
        if name_prefix == "time":
            name_prefix = str_time
        file_name = "".join([name_prefix, file_name, name_suffix])
        self._figure.tight_layout()
        file_name += file_format
        # self.adjust_text()
        file_name = os.path.join(dir_folder, file_name)
        self._figure.savefig("/".join([dir_folder, file_name]), bbox_inches='tight', dpi=dpi)

    def save_figure2(self, path_file, dpi=300):
        self._figure.tight_layout()
        # self.adjust_text()
        self._figure.savefig(path_file, bbox_inches='tight', dpi=dpi)

    def show(self):
        self._figure.show(warn=True)

    @property
    def figure(self):
        return self.figure

    @property
    def axes_primary(self):
        return self._axes[0]

    @property
    def axes_secondary(self):
        return self._axes[1]


if __name__ == "__main__":
    x = np.arange(-2*np.pi, 2*np.pi, 0.01)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    y_tan = np.tan(x)

    p = Scatter2D()
    p.plot2(x, y_sin, 'sin(x)')
    p.plot2(x, y_cos, 'cos(x)')
    p.plot2(x, y_tan, 'tan(x)')
    p.plot2(x+np.pi/2, y_sin, 'sin(x+0.5pi)', second_axis=True)

    # default format
    plt_format = {
        'figure_size_width': 8.,
        'figure_size_height': 6.,
        'figure_size_scale': 1.,
        'figure_title': "",
        'figure_title_font_size': 15.,
        'axis_label_x': "",
        'axis_label_y1': "",
        'axis_label_y2': "",
        'axis_label_font_size': 9.,
        'axis_tick_font_size': 8.,
        'axis_lim_x': None,
        'axis_lim_y1': None,
        'axis_lim_y2': None,
        'axis_linewidth': 1.,
        'axis_scientific_format_x': False,
        'axis_scientific_format_y1': False,
        'axis_scientific_format_y2': False,
        'axis_tick_width': .5,
        'axis_tick_length': 2.5,
        'axis_xtick_major_loc': None,
        'axis_xtick_minor_loc': None,
        'axis_ytick_major_loc': None,
        'axis_ytick_minor_loc': None,
        'axis_grid_show': True,
        'axis_grid_linestyle': "-",
        'axis_grid_linewidth': 0.5,
        'axis_grid_linecolour': "black",
        'marker_size': 3,
        'mark_every': 100,
        'marker_fill_style': "none",
        'marker_edge_width': .5,
        'line_width': 1.,
        'line_style': "-",
        'line_colours': None,
    }

    # re-define some of the values
    plt_format_ = {
        'axis_lim_x': (0., 2*np.pi),
        'axis_lim_y1': (-1., 1.),
        'axis_lim_y2': (-2., 2.),
    }

    # update format dict
    plt_format.update(plt_format_)

    p.format(**plt_format)
    # p.save_figure(r"C:\hello")
    figure_file_path = os.path.abspath(r'hello.png')
    figure_file_path = os.path.realpath(figure_file_path)
    p.save_figure2(figure_file_path)
