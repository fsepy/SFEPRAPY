# -*- coding: utf-8 -*-

from typing import Union, Tuple

import numpy as np


class AsciiPlot:

    def __init__(self, size: Tuple[float, float] = (80, 35)):
        self.__size = None
        self.__xlim = None
        self.__ylim = None
        self.__plot_canvas = None
        self.__plot_yaxis = None
        self.__plot_xaxis = None
        self.__plot = None

        self.size = size

    def plot(self, x, y, xlim: tuple = None, ylim: tuple = None):

        if xlim is None:
            xlim = [min(x), max(x)]
        if ylim is None:
            ylim = [min(y), max(y)]
            if ylim[0] == ylim[1] == 0:
                ylim = -1., 1.
            elif ylim[0] == ylim[1]:
                ylim[0] -= ylim[0] * 0.1
                ylim[1] += ylim[1] * 0.11

        yaxis_size = (None, self.size[1] - 2)
        self.__plot_yaxis = self.__make_yaxis(ylim, yaxis_size)
        xaxis_size = (self.size[0] - self.__plot_yaxis.shape[1], None)
        self.__plot_xaxis = self.__make_xaxis(xlim, xaxis_size)
        canvas_size = (self.size[0] - self.__plot_yaxis.shape[1], self.size[1] - self.__plot_xaxis.shape[0])
        self.__plot_canvas = self._make_canvas(x, y, xlim, ylim, canvas_size)
        self.__plot = self.__assemble(self.__plot_canvas, self.__plot_xaxis, self.__plot_yaxis)

        return self

    def str(self):
        return self.__list2str(self.__plot)

    def show(self):
        try:
            assert self.__plot is not None
        except AssertionError:
            ValueError('No plot to show')

        print(self.__list2str(self.__plot))

    def save(self, fp: str):
        try:
            assert self.__plot is not None
        except AssertionError:
            ValueError('No plot to save')

        with open(fp, 'w+') as f:
            f.write(self.__list2str(self.__plot))

    @staticmethod
    def __list2str(v: Union[list, np.ndarray]) -> str:
        return '\n'.join([''.join([str(j) for j in i]) for i in v])

    @staticmethod
    def _make_canvas(
            x: Union[list, np.ndarray],
            y: Union[list, np.ndarray],
            xlim: tuple = None,
            ylim: tuple = None,
            size: tuple = (80, 40),
            mat_plot_canvas: np.ndarray = None
    ) -> np.ndarray:
        # correct data type if necessary
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if mat_plot_canvas is not None:
            assert mat_plot_canvas.shape[0] == size[::-1][0]
            assert mat_plot_canvas.shape[1] == size[::-1][1]
        else:
            mat_plot_canvas = np.zeros(shape=size[::-1])  # canvas matrix containing the plot

        # workout data plot boundaries
        if xlim:
            x1, x2 = xlim
        else:
            x1, x2 = min(x), max(x)
        if ylim:
            y1, y2 = ylim
        else:
            y1, y2 = min(y), max(y)

        # workout translator arrays
        # example
        #   mat_plot_canvas[i, j] equates location ii[i], jj[i] in data coordination system
        ii = np.linspace(y1, y2, size[1])
        jj = np.linspace(x1, x2, size[0])
        dx, dy = jj[1] - jj[0], ii[1] - ii[0]

        # interpolate x and y, i.e. to increase resolution of the provided x, y
        xi = np.arange(min(x), max(x) + dx / 2, dx / 2)
        yi = np.interp(xi, x, y)

        # plot, assign value 1 to cells in `mat_plot_canvas` the x, y line lands
        for i in range(len(xi)):
            x_, y_ = xi[i], yi[i]

            # cells that the line lands
            jjp = np.argwhere(np.logical_and(jj > x_ - 0.5 * dx, jj < x_ + 0.5 * dx))
            iip = np.argwhere(np.logical_and(ii > y_ - 0.5 * dy, ii < y_ + 0.5 * dy))

            # assign value 1 to the cells
            for ip in iip:
                for jp in jjp:
                    mat_plot_canvas[ip, jp] = 1

        return mat_plot_canvas

    @staticmethod
    def __make_yaxis(ylim, size, label_fmt: str = None) -> np.ndarray:

        # workout data plot boundaries
        y1, y2 = ylim

        # workout width and height per cell of the canvas matrix
        ii = np.linspace(y1, y2, size[1])

        if label_fmt is None:
            max_digits = 0
            for i in ii:
                if len(str(int(i))) > max_digits:
                    max_digits = len(str(int(i)))
            length = max_digits + 4  # (potential) minus sign + max_length + decimal symbol + 2 decimal places
            label_fmt = '{:' + str(int(length)) + '.2f}'
        else:
            length = len(label_fmt.format(0.1))

        mat_yaxis = np.full(shape=(size[1], length + 1), fill_value=' ', dtype='<U1')
        mat_yaxis[:, -1] = '┫'

        for i in range(len(ii)):
            mat_yaxis[i, 0:length] = np.array(list(label_fmt.format(ii[i])), dtype='<U1')

        return mat_yaxis

    @staticmethod
    def __make_xaxis(xlim, size, label_fmt: str = None) -> np.ndarray:

        jj = np.linspace(min(xlim), max(xlim), size[0])

        if label_fmt is None:
            max_digits = 0
            for j in jj:
                if len(str(int(j))) > max_digits:
                    max_digits = len(str(int(j)))
            length = max_digits + 4  # (potential) minus sign + max_length + decimal symbol + 2 decimal places
            label_fmt = '{:' + str(int(length)) + '.2f}'
        else:
            length = len(label_fmt.format(0.1))
        list_x_tick_labels_exhaustive = [label_fmt.format(j) for j in jj]

        dx_label = max([len(i) for i in list_x_tick_labels_exhaustive]) + 2

        mat_xaxis = np.full(shape=(2, size[0]), fill_value=' ', dtype='<U1')
        xaxis_label_index = np.arange(int(dx_label / 2), mat_xaxis.shape[1] - 0.5 * dx_label, dx_label, dtype=int)
        xaxis_label_value = [list_x_tick_labels_exhaustive[i] for i in xaxis_label_index]
        xaxis_label_tick = np.full(shape=(size[0],), fill_value='━', dtype='<U1')
        for i in xaxis_label_index:
            xaxis_label_tick[i] = '┳'

        mat_xaxis[0, :] = xaxis_label_tick

        for i, v in enumerate(' ' + '  '.join(xaxis_label_value) + ' '):
            mat_xaxis[1, i] = v

        return mat_xaxis

    @staticmethod
    def __assemble(mat_plot_canvas, mat_xaxis, mat_yaxis) -> np.ndarray:
        mat_canvas = np.full_like(mat_plot_canvas, fill_value=' ', dtype='<U1')
        mat_canvas[mat_plot_canvas > 0] = '*'

        # construct figure matrix
        mat_figure = np.full(
            # shape=(mat_plot_canvas.shape[0] + mat_xaxis.shape[0], mat_yaxis.shape[1] + mat_plot_canvas.shape[1]),
            shape=(mat_plot_canvas.shape[0] + mat_xaxis.shape[0], mat_plot_canvas.shape[1] + mat_yaxis.shape[1]),
            fill_value=' ', dtype='<U1')

        # assign canvas to figure
        j1 = mat_yaxis.shape[1]
        j2 = mat_yaxis.shape[1] + mat_plot_canvas.shape[1]
        i1 = 0
        i2 = mat_plot_canvas.shape[0]
        mat_figure[i1:i2, j1:j2] = mat_canvas[::-1, :]

        # assign yaxis to figure
        j1 = 0
        j2 = mat_yaxis.shape[1]
        i1 = 0
        i2 = mat_yaxis.shape[0]
        mat_figure[i1:i2, j1:j2] = mat_yaxis[::-1, :]

        # assign xaxis to figure
        j1 = mat_yaxis.shape[1]
        j2 = mat_yaxis.shape[1] + mat_xaxis.shape[1]
        i1 = mat_canvas.shape[0]
        i2 = mat_canvas.shape[0] + mat_xaxis.shape[0]
        mat_figure[i1:i2, j1:j2] = mat_xaxis[:, :]

        # final touches
        i = mat_yaxis.shape[0]
        j = mat_yaxis.shape[1] - 1
        mat_figure[i, j] = '┗'

        return mat_figure

    @property
    def size(self) -> Tuple[float, float]:
        return self.__size

    @size.setter
    def size(self, v: Tuple[float, float]):
        assert any([isinstance(v, tuple), isinstance(v, list)])
        assert len(v) == 2
        self.__size = v

    @property
    def xlim(self):
        return self.__xlim

    @xlim.setter
    def xlim(self, v: Tuple[float, float]):
        assert any([isinstance(v, tuple), isinstance(v, list)])
        assert len(v) == 2

        self.__xlim = v

    @property
    def ylim(self):
        return self.__ylim

    @ylim.setter
    def ylim(self, v: Tuple[float, float]):
        try:
            assert all([isinstance(v, tuple), isinstance(v, list), len(v) == 2])
        except:
            raise ValueError('`ylim` should be a tuple with length equal to 2')

        self.__ylim = v


def _test_asciiplot():
    size = (80, 25)
    xlim = (-2 * np.pi, 2 * np.pi)
    ylim = (-1.1, 1.1)

    aplot = AsciiPlot(size=size)

    x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    aplot.plot(
        x=x,
        y=np.sin(x),
        xlim=xlim,
        ylim=ylim,
    ).show()


if __name__ == '__main__':
    _test_asciiplot()
