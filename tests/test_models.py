import numpy as np
import pytest

import utils
from app import resources
from app.models import BTDModel, BTDModelTRMC, BTDModelTRPL, BTModel, BTModelTRMC, BTModelTRPL, Model

BT_KWARGS = dict(k_B=50e-20, k_T=10e-3, k_A=1e-40)
BTD_KWARGS = dict(k_B=50e-20, k_T=12000e-20, N_T=60e12, p_0=65e12, k_D=80e-20)
T = np.linspace(0, 100, 100)


class TestModel:

    @pytest.fixture
    def model(self):

        param_ids = ["k_B", "k_T", "k_A"]
        units = {"k_B": "cm^3/ns", "k_T": "1/ns", "k_A": "cm^6/ns"}
        units_html = {"k_B": "cm<sup>3</sup>/ns", "k_T": "1/ns", "k_A": "cm<sup>6</sup>/ns"}
        factors = {"k_B": 1e-20, "k_T": 1e-3, "k_A": 1e-40}
        fvalues = {"k_B": 1e-18}
        gvalues = {"k_B": 1e-19, "k_T": 1e-3, "k_A": 1e-40}
        gvalues_range = {"k_B": [1e-20, 1e-19], "k_T": [1e-3, 1e-2], "k_A": [1e-40]}
        n_keys = ["n"]
        conc_ca_ids = ["n"]
        param_filters = []

        def n_init(N_0):
            return {"n": N_0}

        return Model(
            param_ids,
            units,
            units_html,
            factors,
            fvalues,
            gvalues,
            gvalues_range,
            n_keys,
            n_init,
            conc_ca_ids,
            param_filters,
        )

    def test_initialisation(self, model):

        assert model.param_ids == ["k_B", "k_T", "k_A"]
        assert model.units["k_B"] == "cm^3/ns"
        assert model.n_keys == ["n"]
        assert model.fvalues["k_B"] == 1e-18

    def test_get_parameter_label(self, model):

        assert model.get_parameter_label("k_B", "cm^3/ns") == "k_B (cm^3/ns)"
        assert model.get_parameter_label("k_A", "") == "k_A"

    def test_fixed_values(self, model):

        assert model.fixed_values == {"k_B": 1e-18, "y_0": 0.0, "I": 1.0}

    def test_eq(self, model):

        model2 = model
        assert model == model2

        model3 = Model(
            model.param_ids,
            model.units,
            model.units_html,
            model.factors,
            model.fvalues,
            model.gvalues,
            model.gvalues_range,
            model.n_keys,
            model.n_init,
            model.conc_ca_ids,
            model.param_filters,
        )
        assert model == model3

    def test_get_rec_string(self, model):

        expected = "This fit predicts low Auger. The values associated with this process may be inaccurate."
        assert model.get_rec_string("Auger") == expected
        assert "higher excitation fluence" in model.get_rec_string("Auger", "higher")


class TestBTModel:

    def setup_method(self):

        self.model = BTModel(["k_T", "k_B", "k_A", "mu", "y_0"])

    def test_initialization(self):

        assert self.model.param_ids == ["k_T", "k_B", "k_A", "mu", "y_0"]
        assert self.model.units["k_B"] == "cm3/ns"
        assert self.model.factors["k_T"] == 1e-3

    def test_rate_equations(self):

        rates = self.model.rate_equations(n=1e17, **BT_KWARGS)
        expected_rate = -BT_KWARGS["k_T"] * 1e17 - BT_KWARGS["k_B"] * 1e17**2 - BT_KWARGS["k_A"] * 1e17**3
        assert np.isclose(rates["n"], expected_rate)

    def test_calculate_concentrations(self):

        output = self.model.calculate_concentrations(np.linspace(0, 100), 1e17, **BT_KWARGS)
        assert np.allclose(output["n"][0][:3], np.array([1.00000000e17, 8.89910152e16, 8.00020301e16]))
        output2 = self.model.calculate_concentrations(np.linspace(0, 50), 1e17, **BT_KWARGS, p=1000)
        assert np.allclose(output2["n"][-1][:3], np.array([1.21749365e17, 1.13498818e17, 1.06226387e17]))
        assert len(output2["n"]) == 6

    def test_get_carrier_accumulation(self):

        popts = [{"y_0": 0.0, "I": 1.0, "N_0": 1e17, **BT_KWARGS}]
        output = self.model.get_carrier_concentrations([T], popts, 100)
        assert np.allclose(output[2][0]["n"][:3], np.array([1.00000000e17, 9.94032709e16, 9.88130378e16]))
        output2 = self.model.get_carrier_concentrations([T], popts, "")
        assert np.allclose(output2[2][0]["n"][:3], np.array([1.00000000e17, 9.42583326e16, 8.90910901e16]))

    def test_get_recommendations(self):

        contributions = {"Trapping": np.array([5.0]), "Bimolecular": np.array([15.0])}
        recs = self.model.get_recommendations(contributions)
        assert "low trapping" in recs[0]
        assert "higher bimolecular" not in recs[0]


class TestBTModelTRPL:

    def test_calculate_fit_quantity(self):

        result = BTModelTRPL().calculate_fit_quantity(T, N_0=1e17, I=1, y_0=0, **BT_KWARGS)
        assert np.allclose(result[:3], np.array([1.0, 0.88846333, 0.79372223]))

    def test_calculate_contributions(self):

        concentrations = BTModelTRPL().calculate_concentrations(T, 1e16, **BT_KWARGS)
        concentrations = {key: value[0] for key, value in concentrations.items()}
        contributions = BTModelTRPL().calculate_contributions(T, **concentrations, I=1, **BT_KWARGS)
        expected = {
            "T": np.float64(74.27568305105386),
            "B": np.float64(25.72427835628209),
            "A": np.float64(3.8592664051396627e-05),
        }
        assert contributions == expected


class TestBTModelTRMC:

    def test_calculate_fit_quantity(self):

        result = BTModelTRMC().calculate_fit_quantity(T, N_0=1e17, mu=10, y_0=0, **BT_KWARGS)
        assert np.allclose(result[:3], np.array([20.0, 18.85166652, 17.81821802]))

    def test_calculate_contributions(self):

        concentrations = BTModelTRMC().calculate_concentrations(T, 1e16, **BT_KWARGS)
        concentrations = {key: value[0] for key, value in concentrations.items()}
        contributions = BTModelTRMC().calculate_contributions(T, **concentrations, mu=10, **BT_KWARGS)
        expected = {
            "T": np.float64(76.23898193855194),
            "B": np.float64(23.760985144392535),
            "A": np.float64(3.291705552966516e-05),
        }
        assert contributions == expected


class TestBTDModel:

    def setup_method(self):
        self.model = BTDModel(["k_B", "k_T", "k_D", "N_T", "p_0", "mu_e", "mu_h"])

    def test_initialization(self):

        assert self.model.units["k_B"] == "cm3/ns"
        assert self.model.factors["k_B"] == 1e-20
        assert self.model.n_keys == ["n_e", "n_t", "n_h"]

    def test_rate_equations(self):

        result = self.model.rate_equations(n_e=1e17, n_t=1e12, n_h=1e17, **BTD_KWARGS)
        assert result == {"n_e": -5711250000000000.0, "n_t": 707919948000000.0, "n_h": -5003330052000000.0}

    def test_calculate_concentrations(self):

        output = self.model.calculate_concentrations(T, 1e15, **BTD_KWARGS)
        assert np.allclose(output["n_e"][0][:3], np.array([1.00000000e15, 9.92639108e14, 9.86097594e14]))
        output2 = self.model.calculate_concentrations(T, 1e15, **BTD_KWARGS, p=1000)
        assert np.allclose(output2["n_e"][-1][:3], np.array([4.89003948e15, 4.87744397e15, 4.86491226e15]))
        assert len(output2["n_e"]) == 30

    def test_get_carrier_accumulation(self):

        popts = [{"y_0": 0.0, "I": 1.0, "N_0": 1e17, **BTD_KWARGS}]
        output = self.model.get_carrier_concentrations([T], popts, 100)
        assert np.allclose(output[2][0]["n_e"][:3], np.array([1.0000000e17, 9.9460356e16, 9.8954829e16]))
        output2 = self.model.get_carrier_concentrations([T], popts, "")
        assert np.allclose(output2[2][0]["n_t"][:3], np.array([0.00000000e00, 5.96017283e13, 5.96021100e13]))

    def test_get_recommendations(self):

        contributions = {"Bimolecular": np.array([5]), "Trapping": np.array([15]), "Detrapping": np.array([8])}
        recs = self.model.get_recommendations(contributions)
        assert "low" in recs[0] and "low" in recs[1]


class TestBTDModelTRPL:

    def test_calculate_fit_quantity(self):

        result = BTDModelTRPL().calculate_fit_quantity(T, N_0=1e15, I=1, y_0=0, **BTD_KWARGS)
        assert np.allclose(result[:3], np.array([1.0, 0.99213699, 0.98509858]))

    def test_calculate_contributions(self):

        concentrations = BTDModelTRPL().calculate_concentrations(T, 1e15, **BTD_KWARGS)
        concentrations = {key: value[0] for key, value in concentrations.items()}
        contributions = BTDModelTRPL().calculate_contributions(T, **concentrations, **BTD_KWARGS)
        expected = {
            "T": np.float64(41.01610233796725),
            "B": np.float64(56.489242785175534),
            "D": np.float64(2.4946548768572145),
        }
        assert contributions == expected

    def test_fit(self):

        x_data, *ys_data = resources.BTD_TRPL_DATA
        xs_data = [x_data] * len(ys_data)
        N0s = [0.51e14, 1.61e14, 4.75e14, 16.1e14, 43.8e14]
        fit = BTDModelTRPL().fit(xs_data, ys_data, N0s)
        expected = [
            {
                "k_B": np.float64(2.7875838571088765e-19),
                "k_T": np.float64(1.1683973351593177e-16),
                "k_D": np.float64(8.05963830549266e-19),
                "N_T": np.float64(58469666808355.99),
                "p_0": np.float64(66463801712067.75),
                "mu_e": np.float64(10.0),
                "mu_h": np.float64(10.0),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 51000000000000.0,
            },
            {
                "k_B": np.float64(2.7875838571088765e-19),
                "k_T": np.float64(1.1683973351593177e-16),
                "k_D": np.float64(8.05963830549266e-19),
                "N_T": np.float64(58469666808355.99),
                "p_0": np.float64(66463801712067.75),
                "mu_e": np.float64(10.0),
                "mu_h": np.float64(10.0),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 161000000000000.0,
            },
            {
                "k_B": np.float64(2.7875838571088765e-19),
                "k_T": np.float64(1.1683973351593177e-16),
                "k_D": np.float64(8.05963830549266e-19),
                "N_T": np.float64(58469666808355.99),
                "p_0": np.float64(66463801712067.75),
                "mu_e": np.float64(10.0),
                "mu_h": np.float64(10.0),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 475000000000000.0,
            },
            {
                "k_B": np.float64(2.7875838571088765e-19),
                "k_T": np.float64(1.1683973351593177e-16),
                "k_D": np.float64(8.05963830549266e-19),
                "N_T": np.float64(58469666808355.99),
                "p_0": np.float64(66463801712067.75),
                "mu_e": np.float64(10.0),
                "mu_h": np.float64(10.0),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 1610000000000000.0,
            },
            {
                "k_B": np.float64(2.7875838571088765e-19),
                "k_T": np.float64(1.1683973351593177e-16),
                "k_D": np.float64(8.05963830549266e-19),
                "N_T": np.float64(58469666808355.99),
                "p_0": np.float64(66463801712067.75),
                "mu_e": np.float64(10.0),
                "mu_h": np.float64(10.0),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 4380000000000000.0,
            },
        ]
        assert fit["popts"] == expected

    def test_grid_fitting(self):

        x_data, *ys_data = resources.BT_TRPL_DATA
        xs_data = [x_data] * len(ys_data)
        xs_data, ys_data = utils.process_data(xs_data, ys_data, True)
        N0s1 = [1e15, 1e16, 1e17]
        analysis = BTModelTRPL().grid_fitting(None, N0s1, xs_data=xs_data, ys_data=ys_data)
        expected = {"k_A": 0.0, "k_B": np.float64(1.0989031524999562e-19), "k_T": np.float64(0.004680673970587158)}
        assert analysis[0]["popt"] == expected


class TestBTDModelTRMC:

    def setup_method(self):

        self.model = BTDModelTRMC()

    def test_calculate_fit_quantity(self):

        result = self.model.calculate_fit_quantity(T, N_0=1e17, mu_e=10, mu_h=20, y_0=0, **BTD_KWARGS)
        assert np.allclose(result[:3], np.array([30.0, 28.55035409, 27.23913686]))

    def test_calculate_contributions(self):

        concentrations = self.model.calculate_concentrations(T, 1e15, **BTD_KWARGS)
        concentrations = {key: value[0] for key, value in concentrations.items()}
        contributions = self.model.calculate_contributions(T, mu_e=10, mu_h=20, **concentrations, **BTD_KWARGS)
        expected = {
            "T": np.float64(29.64161869985319),
            "B": np.float64(66.18612689655187),
            "D": np.float64(4.172254403594931),
        }
        assert contributions == expected

    def test_fit(self):

        x_data, *ys_data = resources.BTD_TRMC_DATA
        xs_data = [x_data] * len(ys_data)
        N0s = [0.51e14, 1.61e14, 4.75e14, 16.1e14, 43.8e14]
        fit = BTDModelTRMC().fit(xs_data, ys_data, N0s)
        expected = [
            {
                "k_B": np.float64(7.595299999991923e-20),
                "k_T": np.float64(1.1999999998667952e-16),
                "k_D": np.float64(3.199999998957932e-19),
                "N_T": np.float64(54999999999343.74),
                "p_0": np.float64(20000000023555.55),
                "mu_e": np.float64(29.00000000020227),
                "mu_h": np.float64(57.99999999926121),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 51000000000000.0,
            },
            {
                "k_B": np.float64(7.595299999991923e-20),
                "k_T": np.float64(1.1999999998667952e-16),
                "k_D": np.float64(3.199999998957932e-19),
                "N_T": np.float64(54999999999343.74),
                "p_0": np.float64(20000000023555.55),
                "mu_e": np.float64(29.00000000020227),
                "mu_h": np.float64(57.99999999926121),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 161000000000000.0,
            },
            {
                "k_B": np.float64(7.595299999991923e-20),
                "k_T": np.float64(1.1999999998667952e-16),
                "k_D": np.float64(3.199999998957932e-19),
                "N_T": np.float64(54999999999343.74),
                "p_0": np.float64(20000000023555.55),
                "mu_e": np.float64(29.00000000020227),
                "mu_h": np.float64(57.99999999926121),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 475000000000000.0,
            },
            {
                "k_B": np.float64(7.595299999991923e-20),
                "k_T": np.float64(1.1999999998667952e-16),
                "k_D": np.float64(3.199999998957932e-19),
                "N_T": np.float64(54999999999343.74),
                "p_0": np.float64(20000000023555.55),
                "mu_e": np.float64(29.00000000020227),
                "mu_h": np.float64(57.99999999926121),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 1610000000000000.0,
            },
            {
                "k_B": np.float64(7.595299999991923e-20),
                "k_T": np.float64(1.1999999998667952e-16),
                "k_D": np.float64(3.199999998957932e-19),
                "N_T": np.float64(54999999999343.74),
                "p_0": np.float64(20000000023555.55),
                "mu_e": np.float64(29.00000000020227),
                "mu_h": np.float64(57.99999999926121),
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 4380000000000000.0,
            },
        ]
        assert fit["popts"] == expected
