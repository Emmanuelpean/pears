"""plot module"""

import importlib.util
import itertools
import math
import typing as tp
from pathlib import Path

import hiplot
import hiplot.streamlit_helpers
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps
from streamlit import runtime

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
    if labels is None:
        labels = [f"Decay {i + 1}" for i in range(len(xs_data))]

    for i in range(len(xs_data)):
        scatter = go.Scatter(
            x=xs_data[i],
            y=ys_data[i],
            name=labels[i],
            line=dict(color=COLORS[i]),
        )
        figure.add_trace(scatter)
        if ys_data_fit is not None:
            scatter = go.Scatter(
                x=xs_data[i],
                y=ys_data_fit[i],
                name=labels[i] + " (fit)",
                line=dict(color=COLORS[i], dash="dash"),
            )
            figure.add_trace(scatter)

    # Axes
    font = dict(size=16, color="black")
    x_min = min(np.min(x_data) for x_data in xs_data)
    x_max = max(np.max(x_data) for x_data in xs_data)
    xrange = [x_min - 0.01 * (x_max - x_min), x_max + 0.01 * (x_max - x_min)]
    figure.update_xaxes(
        title_text="Time (ns)",
        tickformat=",",
        title_font=font,
        tickfont=font,
        showgrid=True,
        gridcolor="lightgray",
        range=xrange,
    )
    if quantity == "TRPL":
        ylabel = "Intensity (a.u.)"
    else:
        ylabel = "Intensity (cm<sup>2</sup>/(Vs))"

    figure.update_yaxes(
        title_text=ylabel,
        title_font=font,
        tickfont=font,
        showgrid=True,
        gridcolor="lightgray",
    )

    figure.update_layout(
        margin=dict(l=10, r=10, t=40, b=10, pad=0),
        plot_bgcolor="#f0f4f8",
    )

    return figure


def plot_carrier_concentrations(
    xs_data: list[np.ndarray],
    ys_data: list[dict[str, np.ndarray]],
    N0s: list[float],
    titles: list[str],
    xlabel: str,
    model,
) -> go.Figure:
    """Plot all the charge carrier concentrations
    :param xs_data: x-axis data
    :param ys_data: list of dicts of concentration arrays for each initial carrier concentration
    :param N0s: list of initial carrier concentration
    :param titles: list of titles of each subplot
    :param xlabel: x-axis label
    :param model: module used"""

    figure, positions = subplots(len(N0s), 2, subplot_titles=titles)
    font = dict(size=16, color="black")

    for i, N0, position, x_data, y_data in zip(range(len(N0s)), N0s, positions, xs_data, ys_data):

        for key in y_data:
            showlegend = True if i == 0 else False
            scatter = go.Scatter(
                x=x_data,
                y=y_data[key] / N0,
                name=model.CONC_LABELS_HTML[key],
                showlegend=showlegend,
                line=dict(color=model.CONC_COLORS[key]),
            )
            figure.add_trace(scatter, row=position[0], col=position[1])

        figure.update_xaxes(
            title_text=xlabel,
            row=position[0],
            col=position[1],
            tickformat=",",
            title_font=font,
            tickfont=font,
            showgrid=True,
            gridcolor="lightgray",
            range=[min(x_data) - 0.01 * (max(x_data) - min(x_data)), max(x_data) + 0.01 * (max(x_data) - min(x_data))],
        )
        figure.update_yaxes(
            title_text="Concentration (N<sub>0</sub>)",
            row=position[0],
            col=position[1],
            title_font=font,
            tickfont=font,
            showgrid=True,
            gridcolor="lightgray",
        )

    figure.update_layout(
        height=900,
        margin=dict(l=0, r=0, t=40, b=0, pad=0),
        plot_bgcolor="#f0f4f8",
        annotations=[dict(font=font, y=anno["y"] + 0.01) for anno in figure.layout.annotations],
    )
    return figure


def parallel_plot(
    popts: list[dict[str, np.ndarray | float]],
    hidden_keys: list[str],
) -> hiplot.Experiment:
    """Plot data in a parallel plot
    :param popts: list of dicts
    :param hidden_keys: list of keys not displayed"""

    # Determine the key order
    order = ["ID"] + [key for key in popts[0] if key not in hidden_keys]

    # Add ID to the data
    data = []
    for i, popt in enumerate(popts):
        popt["ID"] = i + 1
        data.append(popt)
    popts = data

    # Plot the data
    xp = hiplot.Experiment.from_iterable(popts)
    for key in popts[0]:
        xp.parameters_definition[key].label_html = key
    xp.display_data(hiplot.Displays.PARALLEL_PLOT).update({"hide": hidden_keys + ["uid"], "order": order[::-1]})
    xp.display_data(hiplot.Displays.TABLE).update({"hide": ["from_uid", "uid"]})
    return xp


class _StreamlitHelpers:  # pragma: no cover
    component: tp.Optional[tp.Callable[..., tp.Any]] = None

    @staticmethod
    def is_running_within_streamlit() -> bool:
        try:
            import streamlit as st
        except:  # pylint: disable=bare-except
            return False
        return bool(runtime.exists())

    @classmethod
    def create_component(cls) -> tp.Optional[tp.Callable[..., tp.Any]]:
        if cls.component is not None:
            return cls.component
        import streamlit as st

        try:
            import streamlit.components.v1 as components
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"""Your streamlit version ({st.__version__}) is too old and does not support components.
Please update streamlit with `pip install -U streamlit`"""
            ) from e
        assert runtime.exists()

        # Locate HiPlot module and resolve the path to its static build
        spec = importlib.util.find_spec("hiplot")
        if spec is None or not spec.origin:
            raise RuntimeError("HiPlot module could not be found. Ensure it is installed.")
        hiplot_path = Path(spec.origin).parent
        built_path = (hiplot_path / "static" / "built" / "streamlit_component").resolve()

        assert (
            built_path / "index.html"
        ).is_file(), f"""HiPlot component does not appear to exist in {built_path}
If you did not install HiPlot using official channels (pip, conda...), maybe you forgot to build the JavaScript files?
See https://facebookresearch.github.io/hiplot/contributing.html#building-javascript-bundle"""

        cls.component = components.declare_component("hiplot", path=str(built_path))
        return cls.component


# Fix hiplot compatibility issues
hiplot.streamlit_helpers._StreamlitHelpers = _StreamlitHelpers
