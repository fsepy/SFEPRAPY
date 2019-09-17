#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def save_figure(mcs_out, fp: str):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io

    def cdf_xy(v, xlim, bin_width=0.1, weights=None):
        edges = np.arange(*xlim, bin_width)
        x = (edges[1:] + edges[:-1]) / 2
        y_pdf = np.histogram(v, bins=edges, weights=weights)[0] / len(v)
        y_cdf = np.cumsum(y_pdf)
        return x, y_cdf

    # fig_size = np.array([3.5, 3.5]) * 1  # in inch
    fig_x_limit = (0, 180)
    fig_y_limit = (0, 1)
    bin_width = 0.1

    dict_teq = dict()
    teq_all_case = list()
    pweight_all_case = list()
    for case_name in set(mcs_out['case_name'].values):
        teq = np.asarray(mcs_out[mcs_out['case_name'] == case_name]['solver_time_equivalence_solved'].values, float)
        teq[teq == np.inf] = np.max(teq[teq != np.inf])
        teq[teq == -np.inf] = np.min(teq[teq != -np.inf])
        pweight = np.asarray(mcs_out[mcs_out['case_name'] == case_name]['probability_weight'].values, float)
        pweight = np.array(pweight[teq != np.nan] / 1., dtype=np.float64)
        teq = teq[teq != np.nan]
        teq[teq > 18000.] = 18000.  # limit maximum time equivalence plot value to 5 hours
        dict_teq[case_name] = teq / 60.  # unit conversion: seconds -> minute

        teq_all_case.append(teq / 60.)
        pweight_all_case.append(pweight)
    teq_all_case = np.concatenate(teq_all_case, axis=0)
    pweight_all_case = np.concatenate(pweight_all_case, axis=0)

    xlim = (0, np.max([np.max(v) for k, v in dict_teq.items()]) + bin_width)
    x = None
    y_cdf = dict()
    for k, v in dict_teq.items():
        x, y_cdf[k] = cdf_xy(v, xlim)

    # Create random data with numpy

    y_cdf_combined = np.histogram(teq_all_case, bins=np.arange(*xlim, bin_width))[0]
    y_cdf_combined = np.cumsum(y_cdf_combined)
    print(np.max(y_cdf_combined))
    y_cdf_combined = np.divide(y_cdf_combined, np.max(y_cdf_combined))

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=x, y=y_cdf_combined, mode='lines', name='Combined'))
    for case_name, y in y_cdf.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=case_name))

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
    plotly.io.write_html(fig, file=fp, auto_open=False)


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
