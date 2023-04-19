""" plot module """

import plotly.graph_objects as go
import plotly.subplots as ps

import hiplot as hip

import numpy as np
import math
import itertools


COLORS = ['red', 'green', 'blue', 'black', 'pink', 'purple', 'yellow', 'grey', 'brown'] * 100


def subplots(n, m=None, **kwargs):
    """ Create n subplots
    :param int n: number of subplots
    :param None, int m: if int, maximum number of columns
    :param kwargs: keyword arguments passed to plotly.subplots.make_subplots

    Example
    -------
    >>> subplots(3)[1]
    [(1, 1), (2, 1), (3, 1)]
    >>> subplots(9, 2)[1]
    [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1)]"""

    nb_cols = int(np.sqrt(n))
    if isinstance(m, int) and nb_cols > m:
        nb_cols = m
    nb_rows = int(math.ceil(n / nb_cols))
    positions = list(itertools.product(range(1, nb_rows + 1), range(1, nb_cols + 1)))[:n]
    return ps.make_subplots(rows=nb_rows, cols=nb_cols, **kwargs), positions


def plot_fit(xs_data, ys_data, ys_data_fit=None, labels=None):
    """ Plot raw data and fit data
    :param list, np.ndarray xs_data: list of np.ndarray associated with the x-axis
    :param list, np.ndarray ys_data: list of np.ndarray associated with the y-axis (raw data)
    :param list, np.ndarray ys_data_fit: list of np.ndarray associated with the y-axis (fit)
    :param list labels: list of labels or floats """

    updatemenus = list([
        dict(active=1,
             buttons=list([
                 dict(label='Log Scale',
                      method='update',
                      args=[{'visible': [True, True]},
                            {'yaxis': {'type': 'log', 'tickformat': '%.2e', 'dtick': 1}}]),
                 dict(label='Linear Scale',
                      method='update',
                      args=[{'visible': [True, True]},
                            {'yaxis': {'type': 'linear'}}])
             ]),
             x=0.7, y=1.2
             )
    ])

    figure = go.Figure(layout=dict(updatemenus=updatemenus))

    for i in range(len(xs_data)):
        figure.add_trace(go.Scatter(x=xs_data[i], y=ys_data[i], name=labels[i], line=dict(color=COLORS[i])))
        if ys_data_fit is not None:
            figure.add_trace(go.Scatter(x=xs_data[i], y=ys_data_fit[i], name=labels[i] + ' (fit)',
                                        line=dict(color=COLORS[i], dash='dash')))
    figure.update_xaxes(title_text='Time (ns)', tickformat=',')
    figure.update_yaxes(title_text='Intensity (a.u.)')
    return figure


def plot_carrier_concentrations(xs_data, ys_data, N_0s, titles, xlabel, model):
    """ Plot all the charge carrier concentrations
    :param xs_data: x axis data
    :param ys_data: list of list of concentration arrays for each initial carrier concentration
    :param N_0s: list of initial carrier concentration
    :param titles: list of titles of each subplot
    :param xlabel: x axis label
    :param model: module used"""

    figure, positions = subplots(len(N_0s), 2, subplot_titles=titles)

    for i, N_0, position, x_data, y_data in zip(range(len(N_0s)), N_0s, positions, xs_data, ys_data):

        for key in y_data:
            showlegend = True if i == 0 else False
            figure.add_trace(go.Scatter(x=x_data, y=y_data[key] / N_0, name=model.n_labels_html[key], showlegend=showlegend,
                                        line=dict(color=model.n_colors[key])), row=position[0], col=position[1])

        figure.update_xaxes(title_text=xlabel, row=position[0], col=position[1], tickformat=',')
        figure.update_yaxes(title_text='Concentration (N<sub>0</sub>)', row=position[0], col=position[1])
        figure.update_layout(height=900)

    return figure


def parallel_plot(data, notdisp, order):
    """ Plot data in a parallel plot
    :param list data: list of dicts
    :param list notdisp: list of keys not displayed
    :param list order: list of keys in display order """

    data_ = []
    for i, d in enumerate(data):
        d['ID'] = i + 1
        data_.append(d)
    data = data_

    xp = hip.Experiment.from_iterable(data)
    for key in data[0]:
        xp.parameters_definition[key].label_html = key
    xp.display_data(hip.Displays.PARALLEL_PLOT).update({'hide': ['uid'] + notdisp, 'order': ['ID'] + order})
    xp.display_data(hip.Displays.TABLE).update({'hide': ['from_uid', 'uid']})
    return xp


if __name__ == '__main__':
    import doctest
    doctest.testmod()
