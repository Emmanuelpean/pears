import hiplot
import numpy as np
import plotly.graph_objects as go
import pytest

from app.plot import parallel_plot, plot_carrier_concentrations, plot_decays, subplots


class TestSubplots:
    def test_simple_subplot_creation(self):
        fig, positions = subplots(3)
        assert isinstance(fig, go.Figure)
        assert positions == [(1, 1), (2, 1), (3, 1)]

    def test_subplot_with_max_columns(self):
        fig, positions = subplots(9, 2)
        assert isinstance(fig, go.Figure)
        assert positions == [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1)]

    def test_subplot_with_kwargs(self):
        fig, positions = subplots(4, subplot_titles=["A", "B", "C", "D"])
        assert isinstance(fig, go.Figure)
        assert positions == [(1, 1), (1, 2), (2, 1), (2, 2)]


class TestPlotDecays:

    @pytest.fixture
    def decay_data(self):
        return {
            "xs_data": [np.array([1, 2, 3]), np.array([1, 2, 3])],
            "ys_data": [np.array([10, 5, 2]), np.array([8, 4, 1])],
            "ys_data_fit": [np.array([9, 5, 2.5]), np.array([8.5, 4.2, 1.1])],
            "labels": ["Sample A", "Sample B"],
        }

    def test_plot_decays_trpl(self, decay_data):
        fig = plot_decays(
            decay_data["xs_data"],
            decay_data["ys_data"],
            "TRPL",
            labels=decay_data["labels"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two traces for two datasets
        assert fig.data[0].name == "Sample A"
        assert fig.data[1].name == "Sample B"
        assert "Time (ns)" in fig.layout.xaxis.title.text
        assert "Intensity (a.u.)" in fig.layout.yaxis.title.text

    def test_plot_decays_trmc(self, decay_data):
        fig = plot_decays(
            decay_data["xs_data"],
            decay_data["ys_data"],
            "TRMC",
            labels=decay_data["labels"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert "Time (ns)" in fig.layout.xaxis.title.text
        assert "Intensity (cm<sup>2</sup>/(Vs))" in fig.layout.yaxis.title.text

    def test_plot_decays_with_fit(self, decay_data):
        fig = plot_decays(
            decay_data["xs_data"],
            decay_data["ys_data"],
            "TRPL",
            ys_data_fit=decay_data["ys_data_fit"],
            labels=decay_data["labels"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # Four traces: two for data and two for fit
        assert fig.data[0].name == "Sample A"
        assert fig.data[1].name == "Sample A (fit)"
        assert fig.data[2].name == "Sample B"
        assert fig.data[3].name == "Sample B (fit)"
        assert "dash" in fig.data[1].line
        assert "dash" in fig.data[3].line


class SimpleCarrierModel:
    def __init__(self):
        self.CONC_LABELS_HTML = {"e": "Electrons", "h": "Holes"}
        self.CONC_COLORS = {"e": "blue", "h": "red"}


class TestPlotCarrierConcentrations:

    @pytest.fixture
    def carrier_data(self):

        model = SimpleCarrierModel()

        # Test data
        xs_data = [np.array([1, 2, 3]), np.array([1, 2, 3])]
        ys_data = [
            {"e": np.array([10, 5, 2]), "h": np.array([10, 6, 3])},
            {"e": np.array([8, 4, 1]), "h": np.array([8, 5, 2])},
        ]
        N_0s = [1e15, 1e16]
        titles = ["Sample A", "Sample B"]
        xlabel = "Time (ns)"

        return {
            "model": model,
            "xs_data": xs_data,
            "ys_data": ys_data,
            "N_0s": N_0s,
            "titles": titles,
            "xlabel": xlabel,
        }

    def test_plot_carrier_concentrations(self, carrier_data):
        fig = plot_carrier_concentrations(
            carrier_data["xs_data"],
            carrier_data["ys_data"],
            carrier_data["N_0s"],
            carrier_data["titles"],
            carrier_data["xlabel"],
            carrier_data["model"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # 2 samples x 2 carrier types = 4 traces

        # Check that the traces are correctly named and colored
        electron_traces = [trace for trace in fig.data if trace.name == "Electrons"]
        hole_traces = [trace for trace in fig.data if trace.name == "Holes"]

        assert len(electron_traces) == 2
        assert len(hole_traces) == 2

        # Check that colors are correctly applied
        for trace in electron_traces:
            assert trace.line.color == "blue"

        for trace in hole_traces:
            assert trace.line.color == "red"

        # Check that showlegend is only true for the first occurrence of each type
        assert electron_traces[0].showlegend is True
        assert electron_traces[1].showlegend is False
        assert hole_traces[0].showlegend is True
        assert hole_traces[1].showlegend is False


class TestParallelPlot:
    def test_basic_plot(self):
        data = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        notdisp = ["b"]
        order = ["a"]

        xp = parallel_plot(data, notdisp, order)

        assert isinstance(xp, hiplot.Experiment)
        assert len(xp.datapoints) == 2
        assert xp.datapoints[0].values["ID"] == 1
        assert xp.datapoints[1].values["ID"] == 2

    def test_hide_and_order(self):
        data = [{"x": 10, "y": 20, "z": 30}]
        notdisp = ["y"]
        order = ["x", "z"]

        xp = parallel_plot(data, notdisp, order)
        display_data = xp.display_data(hiplot.Displays.PARALLEL_PLOT)

        assert "y" in display_data["hide"]
        assert display_data["order"] == ["ID", "x", "z"]
