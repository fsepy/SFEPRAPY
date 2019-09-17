#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def save_figure(mcs_out, fp: str):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io

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
    bin_width = 0.1

    # Process time equivalence value, obtain `probability_weight` and `n_simulations`
    list_case_name = list(set(mcs_out['case_name'].values))
    list_t_eq = list()
    list_weight = list()
    list_n_simulation = list()
    for case_name in list_case_name:
        teq = np.asarray(mcs_out[mcs_out['case_name'] == case_name]['solver_time_equivalence_solved'].values, float)
        teq[teq == np.inf] = np.max(teq[teq != np.inf])
        teq[teq == -np.inf] = np.min(teq[teq != -np.inf])
        teq = teq[teq != np.nan]
        teq[teq > 18000.] = 18000.  # limit maximum time equivalence plot value to 5 hours
        list_t_eq.append(teq / 60.)
        list_weight.append(np.average(mcs_out[mcs_out['case_name'] == case_name]['probability_weight'].values))
        list_n_simulation.append(np.average(mcs_out[mcs_out['case_name'] == case_name]['n_simulations'].values))

    # Time equivalence samples -> x, y of cumulative density function
    xlim = (0, np.max([np.max(v) for v in list_t_eq]) + bin_width)
    x = np.arange(*xlim, bin_width)
    list_cdf_t_eq_y = [cdf_xy(t_eq, xlim, bin_width, weights=None)[1] for t_eq in list_t_eq]

    # Combined time equivalence cdf
    cdf_t_eq_y_combined = np.array([list_weight[i] * v for i, v in enumerate(list_cdf_t_eq_y)])
    cdf_t_eq_y_combined = np.sum(cdf_t_eq_y_combined, axis=0)

    # Plot figure
    fig = go.Figure()

    # add combined time equivalence to the plot
    if len(list_case_name) > 1:
        fig.add_trace(go.Scatter(x=x, y=cdf_t_eq_y_combined, mode='lines', name='Combined'))

    # add individual time equivalence to the plot
    for i, t_eq in enumerate(list_cdf_t_eq_y):
        fig.add_trace(go.Scatter(x=x, y=t_eq, mode='lines', name=list_case_name[i]))

    fig.update_layout(
        autosize=True,
        # width=fig_size[0] * 96,
        # height=fig_size[1] * 96,
        # margin=dict(
        #     l=20,
        #     r=20,
        #     b=20,
        #     t=20,
        #     pad=0
        # ),
        paper_bgcolor='White',
        plot_bgcolor='White',
        xaxis=dict(
            title="Equivalent of time exposure [minute]",
            dtick=30,
            range=fig_x_limit,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            visible=True,
            showgrid=True,
            gridcolor='Black',
            gridwidth=1,
            ticks='outside',
            zeroline=False,
        ),
        yaxis=dict(
            title='CDF',
            dtick=0.2,
            range=fig_y_limit,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            visible=True,
            showgrid=True,
            gridcolor='Black',
            gridwidth=1,
            ticks='outside',
            zeroline=False,
        ),
        legend=dict(
            x=0.98,
            xanchor='right',
            y=0.02,
            yanchor='bottom',
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=9,
                color="black"
            ),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1,
        ),
    )
    plotly.io.write_html(fig, file=fp, auto_open=True)


if __name__ == '__main__':
    import os
    import sys
    import warnings
    from sfeprapy.func.mcs_obj import MCS
    from sfeprapy.mcs0.mcs0_calc import teq_main, teq_main_wrapper, mcs_out_post
    from sfeprapy.func.mcs_gen import main as gen

    warnings.filterwarnings('ignore')

    mcs_problem_definition = None
    n_threads = None
    fig = False
    fig_only = False
    if len(sys.argv) > 1:
        mcs_problem_definition = os.path.realpath(sys.argv[1])
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if 'mp' in arg:
                n_threads = int(str(arg).replace('mp', ''))
            if 'fig' in arg:
                fig = True
            if 'figonly' in arg:
                fig_only = True

    if fig_only:
        import pandas as pd
        mcs_out = pd.read_csv(mcs_problem_definition)
        save_figure(mcs_out=mcs_out, fp=os.path.join(os.path.dirname(mcs_problem_definition), 'mcs.out.html'))
    else:
        mcs = MCS()
        mcs.define_problem(mcs_problem_definition)
        mcs.define_stochastic_parameter_generator(gen)
        mcs.define_calculation_routine(teq_main, teq_main_wrapper, mcs_out_post)

        try:
            if n_threads:
                mcs.config = dict(n_threads=n_threads) if mcs.config else mcs.config.update(dict(n_threads=n_threads))
        except KeyError:
            pass

        mcs.run_mcs()

        if fig:
            save_figure(mcs_out=mcs.mcs_out, fp=os.path.join(os.path.dirname(mcs_problem_definition), 'mcs.out.html'))
