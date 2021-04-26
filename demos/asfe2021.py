from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d


# Helper function
def get_fr_func(teq: np.ndarray) -> Callable:
    hist, edges = np.histogram(teq, bins=np.arange(0, max(361, max(teq[teq < np.inf]) + 0.5), 0.5))
    x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    func_fr = interp1d(y, x, bounds_error=False)
    return func_fr


# Helper function: get a function to return teq = f(fractile)
def get_teq_fractile_func(teq: np.ndarray) -> Callable:
    hist, edges = np.histogram(teq, bins=np.arange(0, max(361, max(teq[teq < np.inf]) + 0.5), 0.5))
    x, y = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    func_teq = interp1d(x, y, bounds_error=False)
    return func_teq


# Helper function: get fractile for a specific teq
def get_teq_fractile(teq: np.ndarray, fr: list = tuple(range(30, 181, 15))) -> np.ndarray:
    return get_teq_fractile_func(teq=teq)(fr)


# Helper function: get
def get_teq_fractile_dict(teq_dict: dict, fr: list = tuple(range(30, 181, 15)), is_print: bool = True) -> dict:
    """Prints CFD of given time equivalence values"""
    res_dict = dict(Time=fr)
    for case, teq in teq_dict.items():
        res_dict[case] = get_teq_fractile(teq=teq, fr=fr)
    if is_print is True:
        print(pd.DataFrame.from_dict(res_dict))
    return res_dict


# Helper function: standard figure format
def format_ax(
        ax,
        xlabel: str = None, ylabel: str = None, legend_title: str = None,
        xlabel_fontsize='small', ylabel_fontsize='small',
        xscale=None, yscale=None,
        xticks=None, yticks=None,
        xticklabels=None, yticklabels=None,
        ticks_labelsize='x-small',
        xlim=None, ylim=None,
        ticklabel_format_style=None, ticklabel_format_axis='y', ticklabel_format_scilimits=(0, 0),
        legend_loc: int = 0, legend_ncol=1, legend_fontsize='x-small', legend_title_fontsize='x-small',
        legend_visible: bool = True,
        grid_which='both', grid_ls='--'
):
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if xscale is not None: ax.set_xscale(xscale)
    if yscale is not None: ax.set_yscale(yscale)
    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)
    if xticklabels is not None: ax.set_xticklabels(xticklabels)
    if yticklabels is not None: ax.set_yticklabels(yticklabels)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if legend_visible is True:
        ax.legend(
            title=legend_title, loc=legend_loc, ncol=legend_ncol, frameon=True, fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize
        ).set_visible(legend_visible)
    ax.grid(which=grid_which, ls=grid_ls)
    ax.tick_params(labelsize=ticks_labelsize)

    if ticklabel_format_style is not None:
        ax.ticklabel_format(
            style=ticklabel_format_style,
            scilimits=ticklabel_format_scilimits,
            axis=ticklabel_format_axis,
            useMathText=True
        )

    if ticklabel_format_style is not None:
        ax.yaxis.offsetText.set_fontsize('x-small')


def func_cdf_teq_from_area_height(
        area: np.ndarray,
        height: np.ndarray,
        func_beta_from_height: Callable,
        p_1: float, p_2: float, p_3: float, p_4: float,
):
    """
    Solve:
                             P_f_fi = P_a_fi
    p_1 * A * p_2 * p_3 * p_4 * p_5 = P_a_fi(H)
                                p_5 = P_a_fi(H) / (p_1 * A * p_2 * p_3 * p_4)
                        1 - teq_cdf = P_a_fi(H) / (p_1 * A * p_2 * p_3 * p_4)
                            teq_cdf = 1 - P_a_fi(H) / (p_1 * A * p_2 * p_3 * p_4)
    """
    # solve/interpolate teq cdf
    P_a_fi = stats.norm(loc=0, scale=1).cdf(-func_beta_from_height(height))
    teq_cdf = 1 - np.divide(P_a_fi, p_1 * area * p_2 * p_3 * p_4, dtype=np.float64)
    return teq_cdf


# helper function
def func_teq_from_area_height(
        area: np.ndarray,
        height: np.ndarray,
        teq_cdf: np.ndarray,
        teq: np.ndarray,
):
    area.astype(np.float64)
    height.astype(np.float64)

    # work out teq = f(fractile)
    hist, edges = np.histogram(teq, bins=np.arange(0, max(361, max(teq[teq < np.inf]) + 0.5), 0.5))
    teq_i, cdf_i = (edges[:-1] + edges[1:]) / 2, np.cumsum(hist / np.sum(hist))
    func_teq_from_fractile = interp1d(cdf_i, teq_i, bounds_error=False, fill_value=(0., 9999.))

    # teq_cdf = func_cdf_teq_from_area_height(
    #     area=area, height=height, func_beta_from_height=func_beta_from_height, p_1=p_1, p_2=p_2, p_3=p_3, p_4=p_4
    # )
    return func_teq_from_fractile(teq_cdf)


def plot_contour(
        ax, xx, yy, zz,
        xlabel: str, xticks=None, xticks_minor=None, xticklabels=None, xlim=None,
        levels=None, clabel_fmt=lambda x: f'{x:.0f}', clabel_manual=False,
):
    cs = ax.contour(xx, yy, zz, levels=levels, linewidths=0.5, colors='k', antialiased=True)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap='coolwarm', alpha=0.6, extend='both')
    ax.clabel(
        cs, cs.levels, inline=True, fmt=clabel_fmt, fontsize='x-small',
        use_clabeltext=True, inline_spacing=5, manual=clabel_manual
    )
    format_ax(
        ax=ax, xlabel=xlabel, xticks=xticks, xticklabels=xticklabels, xlim=xlim, ticklabel_format_style='sci',
        legend_visible=False
    )
    if xticks_minor is not None: ax.set_xticks(xticks_minor, minor=True)

    return cf


def plot_contour_text_p_i(ax, title, p_i, x=0.95, y=0.97, va='top', ha='right', bbox_pad=-0.01, bbox_fc=(1, 1, 1, 0.5)):
    ax.text(
        x, y,
        f'{title}\n'
        f'$p_1$={p_i["p_1"]}\n'
        f'$p_2$={p_i["p_2"]}\n'
        f'$p_3$={p_i["p_3"]}\n'
        f'$p_4$={p_i["p_4"]}',
        transform=ax.transAxes, ha=ha, va=va, ma='left', fontsize='x-small',
        bbox=dict(boxstyle=f'square,pad={bbox_pad}', fc=bbox_fc, ec='none')
    )


def ax_annotate(
        ax, text, xy, xytext, fontsize='x-small', ha='center', va='bottom', ma='left',
        arrowprops_style='->', arrowprops_lw=.5, arrowprops_lc='k',
        bbox_pad=-0.01, bbox_fc=(1, 1, 1, 0.5), bbox_ec='none'
):
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=fontsize, ha=ha, va=va, ma=ma,
        arrowprops=dict(arrowstyle=arrowprops_style, lw=arrowprops_lw),
        bbox=dict(boxstyle=f'square,pad={bbox_pad}', fc=bbox_fc, ec=bbox_ec)
    )

