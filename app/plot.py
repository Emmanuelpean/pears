"""plot module"""

import itertools
import math

import hiplot
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps

COLORS = ["red", "green", "blue", "black", "pink", "purple", "yellow", "grey", "brown"] * 100


def subplots(
    n: int,
    m: int | None = None,
    **kwargs,
) -> tuple[go.Figure, list[tuple[int, int]]]:
    """Create n subplots
    :param n: number of subplots
    :param m: if int, maximum number of columns
    :param kwargs: keyword arguments passed to plotly.subplots.make_subplots

    Examples
    --------
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


def plot_decays(
    xs_data: list[np.ndarray],
    ys_data: list[np.ndarray],
    quantity: str,
    ys_data_fit: list[np.ndarray] | None = None,
    labels: list[str] | None = None,
) -> go.Figure:
    """Plot decays
    :param xs_data: list of np.ndarray associated with the x-axis
    :param ys_data: list of np.ndarray associated with the y-axis (raw data)
    :param quantity: quantity fitted: 'TRPL' or 'TRMC'
    :param ys_data_fit: list of np.ndarray associated with the y-axis (fit)
    :param labels: list of labels or floats"""

    figure = go.Figure()
    for i in range(len(xs_data)):
        figure.add_trace(
            go.Scatter(
                x=xs_data[i],
                y=ys_data[i],
                name=labels[i],
                line=dict(color=COLORS[i]),
            )
        )
        if ys_data_fit is not None:
            figure.add_trace(
                go.Scatter(
                    x=xs_data[i],
                    y=ys_data_fit[i],
                    name=labels[i] + " (fit)",
                    line=dict(color=COLORS[i], dash="dash"),
                )
            )
    figure.update_xaxes(title_text="Time (ns)", tickformat=",")
    if quantity == "TRPL":
        figure.update_yaxes(title_text="Intensity (a.u.)")
    elif quantity == "TRMC":
        figure.update_yaxes(title_text="Intensity (cm<sup>2</sup>/(Vs))")
    return figure


def plot_carrier_concentrations(
    xs_data: list[np.ndarray],
    ys_data: list[dict[str, np.ndarray]],
    N_0s: list[float],
    titles: list[str],
    xlabel: str,
    model,
) -> go.Figure:
    """Plot all the charge carrier concentrations
    :param xs_data: x-axis data
    :param ys_data: list of list of concentration arrays for each initial carrier concentration
    :param N_0s: list of initial carrier concentration
    :param titles: list of titles of each subplot
    :param xlabel: x-axis label
    :param model: module used"""

    figure, positions = subplots(len(N_0s), 2, subplot_titles=titles)

    for i, N_0, position, x_data, y_data in zip(range(len(N_0s)), N_0s, positions, xs_data, ys_data):

        for key in y_data:
            showlegend = True if i == 0 else False
            figure.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data[key] / N_0,
                    name=model.CONC_LABELS_HTML[key],
                    showlegend=showlegend,
                    line=dict(color=model.CONC_COLORS[key]),
                ),
                row=position[0],
                col=position[1],
            )

        figure.update_xaxes(title_text=xlabel, row=position[0], col=position[1], tickformat=",")
        figure.update_yaxes(title_text="Concentration (N<sub>0</sub>)", row=position[0], col=position[1])
        figure.update_layout(height=900)

    return figure


def parallel_plot(
    data: list[dict],
    notdisp: list[str],
    order: list[str],
) -> hiplot.Experiment:
    """Plot data in a parallel plot
    :param data: list of dicts
    :param notdisp: list of keys not displayed
    :param order: list of keys in display order"""

    data_ = []
    for i, d in enumerate(data):
        d["ID"] = i + 1
        data_.append(d)
    data = data_

    xp = hiplot.Experiment.from_iterable(data)
    for key in data[0]:
        xp.parameters_definition[key].label_html = key
    xp.display_data(hiplot.Displays.PARALLEL_PLOT).update({"hide": ["uid"] + notdisp, "order": ["ID"] + order})
    xp.display_data(hiplot.Displays.TABLE).update({"hide": ["from_uid", "uid"]})
    return xp


if __name__ == "__main__":
    import doctest

    doctest.testmod()
