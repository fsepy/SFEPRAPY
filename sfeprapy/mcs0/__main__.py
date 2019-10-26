#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def save_figure(fp_mcs0_out: str):
    import os
    import numpy as np
    import pandas as pd
    import plotly.io
    import plotly.graph_objects as go

    mcs_out = pd.read_csv(fp_mcs0_out)

    # A helper function to produce (x, y) line plot based upon samples
    def cdf_xy(v, xlim, bin_width=0.1, weights=None):
        edges = np.arange(*xlim, bin_width)
        x = (edges[1:] + edges[:-1]) / 2
        y_pdf = np.histogram(v, bins=edges, weights=weights)[0] / len(v)
        y_cdf = np.cumsum(y_pdf)
        return x, y_cdf

    # Settings
    # fig_size = np.array([3.5, 3.5]) * 1  # in inch
    fig_x_limit = (0, 180)
    fig_y_limit = (0, 1)
    bin_width = 0.5

    # Process time equivalence value, obtain `probability_weight` and `n_simulations`
    list_case_name = sorted(list(set(mcs_out["case_name"].values)))
    list_t_eq = list()
    list_weight = list()
    list_n_simulation = list()
    for case_name in list_case_name:
        teq = np.asarray(
            mcs_out[mcs_out["case_name"] == case_name][
                "solver_time_equivalence_solved"
            ].values,
            float,
        )
        teq[teq == np.inf] = np.max(teq[teq != np.inf])
        teq[teq == -np.inf] = np.min(teq[teq != -np.inf])
        teq = teq[~np.isnan(teq)]
        teq[
            teq > 18000.0
        ] = 18000.0  # limit maximum time equivalence plot value to 5 hours
        list_t_eq.append(teq / 60.0)
        list_weight.append(
            np.average(
                mcs_out[mcs_out["case_name"] == case_name]["probability_weight"].values
            )
        )
        list_n_simulation.append(
            np.average(
                mcs_out[mcs_out["case_name"] == case_name]["n_simulations"].values
            )
        )

    # Time equivalence samples -> x, y of cumulative density function
    xlim = (0, np.max([np.max(v) for v in list_t_eq]) + bin_width)
    x = np.arange(*xlim, bin_width)
    list_cdf_t_eq_y = [
        cdf_xy(t_eq, xlim, bin_width, weights=None)[1] for t_eq in list_t_eq
    ]

    # Combined time equivalence cdf
    cdf_t_eq_y_combined = np.array(
        [list_weight[i] * v for i, v in enumerate(list_cdf_t_eq_y)]
    )
    cdf_t_eq_y_combined = np.sum(cdf_t_eq_y_combined, axis=0)

    # Plot figure
    fig = go.Figure()

    # add combined time equivalence to the plot
    if len(list_case_name) > 1 and np.max(cdf_t_eq_y_combined) > 0.9:
        fig.add_trace(
            go.Scatter(x=x, y=cdf_t_eq_y_combined, mode="lines", name="Combined")
        )

    # add individual time equivalence to the plot
    for i, t_eq in enumerate(list_cdf_t_eq_y):
        fig.add_trace(go.Scatter(x=x, y=t_eq, mode="lines", name=list_case_name[i]))

    fig.update_layout(
        autosize=True,
        paper_bgcolor="White",
        plot_bgcolor="White",
        xaxis=dict(
            title="Equivalent of time exposure [minute]",
            dtick=30,
            range=fig_x_limit,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            visible=True,
            showgrid=True,
            gridcolor="Black",
            gridwidth=1,
            ticks="outside",
            zeroline=False,
        ),
        yaxis=dict(
            title="CDF",
            dtick=0.2,
            range=fig_y_limit,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            visible=True,
            showgrid=True,
            gridcolor="Black",
            gridwidth=1,
            ticks="outside",
            zeroline=False,
        ),
        legend=dict(
            x=0.98,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            traceorder="normal",
            font=dict(family="sans-serif", color="black"),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1,
        ),
    )

    config = {
        "scrollZoom": False,
        "displayModeBar": True,
        "editable": True,
        "showLink": False,
        "displaylogo": False,
    }

    plotly.io.write_html(
        fig,
        file=os.path.join(os.path.dirname(fp_mcs0_out), "mcs.out.html"),
        auto_open=True,
        config=config,
    )


def main(fp_mcs_in: str, n_threads: int = None):

    import os
    import warnings
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen

    warnings.filterwarnings("ignore")

    fp_mcs_in = os.path.realpath(fp_mcs_in)

    mcs = MCS()
    mcs.define_problem(fp_mcs_in)
    mcs.define_stochastic_parameter_generator(gen)
    mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)

    try:
        if n_threads:
            mcs.config = (
                dict(n_threads=n_threads)
                if mcs.config
                else mcs.config.update(dict(n_threads=n_threads))
            )
    except KeyError:
        pass

    mcs.run_mcs()


if __name__ == "__main__":
    print("Use `sfeprapy` CLI, `sfeprapy.mcs0:__main__` is depreciated on 22 Oct 2019.")
