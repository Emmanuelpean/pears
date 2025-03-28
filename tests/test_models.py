import numpy as np
import pytest

from app.models import (
    BTDModel,
    BTDModelTRMC,
    BTDModelTRPL,
    BTD_KWARGS,
    BTModel,
    BTModelTRMC,
    BTModelTRPL,
    BT_KWARGS,
    Model,
)
from app.utility.data import are_close

# Time array
T = np.linspace(0, 100, 101)


def assert_fit(
    fit: dict,
    popt_expected: dict,
    contribution_expected: dict,
    cod_expected: float,
) -> None:
    """Check the result of a fit.
    :param fit: fit result
    :param popt_expected: expected optimised fit popt
    :param contribution_expected: expected contributions
    :param cod_expected: expected cod"""

    assert are_close(fit["popts"][0], popt_expected)
    assert are_close(fit["contributions"], contribution_expected)
    assert are_close(fit["cod"], cod_expected)


class TestModel:

    @pytest.fixture
    def model(self):

        param_ids = ["k_B", "k_T", "k_A", "y_0"]
        units = {"k_B": "cm^3/ns", "k_T": "1/ns", "k_A": "cm^6/ns"}
        units_html = {"k_B": "cm<sup>3</sup>/ns", "k_T": "1/ns", "k_A": "cm<sup>6</sup>/ns"}
        factors = {"k_B": 1e-20, "k_T": 1e-3, "k_A": 1e-40}
        fvalues = {"k_B": 1e-18, "k_T": None, "k_A": None}
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

        assert model.param_ids == ["k_B", "k_T", "k_A", "y_0"]
        assert model.units["k_B"] == "cm^3/ns"
        assert model.n_keys == ["n"]
        assert model.fvalues["k_B"] == 1e-18

    def test_get_parameter_label(self, model):

        assert model.get_parameter_label("k_B") == "k_B (cm^3/ns)"
        assert model.get_parameter_label("y_0") == "y_0"

    def test_fixed_values(self, model):

        assert model.fixed_values == {"k_B": 1e-18, "y_0": 0.0}

    def test_eq(self, model):

        model2 = model
        assert model == model2

        model3 = Model(
            model.param_ids,
            model.units,
            model.units_html,
            model.factors,
            {"k_B": 1},
            model.gvalues,
            model.gvalues_range,
            model.n_keys,
            model.n_init,
            model.conc_ca_ids,
            model.param_filters,
        )
        assert model != model3

    def test_get_rec_string(self, model):

        expected = "This fit predicts low Auger. The values associated with this process may be inaccurate."
        expected2 = (
            "This fit predicts low Auger. The values associated with this process may be inaccurate.\nIt is "
            "recommended to measure your sample under higher excitation fluence for this process to become significant"
        )
        assert model.get_contribution_recommendation("Auger") == expected
        assert model.get_contribution_recommendation("Auger", "higher") == expected2


class TestBTModel:

    @pytest.fixture
    def model(self):

        return BTModel(["k_T", "k_B", "k_A", "mu", "y_0"])

    def test_initialization(self, model):

        assert model.param_ids == ["k_T", "k_B", "k_A", "mu", "y_0"]
        assert model.units["k_B"] == "cm3/ns"
        assert model.factors["k_T"] == 1e-3

    def test_rate_equations(self, model):

        rates = model._rate_equations(n=1e17, **BT_KWARGS)
        assert np.isclose(rates["n"], -6000100000000000.0)

    def test_calculate_concentrations(self, model):

        # Single pulse
        output = model._calculate_concentrations(T, 1e17, **BT_KWARGS)
        assert np.allclose(output["n"][0][:3], np.array([1.00000000e17, 9.43127550e16, 8.91893621e16]))

        # Multiple pulses
        output2 = model._calculate_concentrations(T, 1e17, **BT_KWARGS, p=1000)
        assert np.allclose(output2["n"][-1][:3], np.array([1.09021346e17, 1.02383279e17, 9.64515430e16]))
        assert len(output2["n"]) == 5

        # Additional parameters
        output = model._calculate_concentrations(T, 1e17, **BT_KWARGS)
        assert np.allclose(output["n"][0][:3], np.array([1.00000000e17, 9.43127550e16, 8.91893621e16]))

    def test_get_carrier_concentrations(self, model):

        popts = [{"I": 1.0, "N_0": 1e17, **BT_KWARGS}]

        # Period provided
        output = model.get_carrier_concentrations([T], popts, 100)
        assert np.allclose(output[2][0]["n"][:3], np.array([1.00000000e17, 9.94032709e16, 9.88130378e16]))
        assert np.allclose(output[0][0][:3], np.array([0.00000000e00, 9.99010979e-04, 1.99802196e-03]))

        # Period not provided
        output2 = model.get_carrier_concentrations([T], popts, 0)
        assert np.allclose(output2[2][0]["n"][:3], np.array([1.00000000e17, 9.43127550e16, 8.91893621e16]))
        assert np.allclose(output2[0][0][:3], np.array([0.0, 1.0, 2.0]))

    def test_get_contribution_recommendations(self, model):

        contributions = {"T": np.array([5.0]), "B": np.array([15.0])}
        recs = model.get_contribution_recommendations(contributions)
        expected = [
            "This fit predicts low trapping. The values associated with this process may be inaccurate.\nIt is "
            "recommended to measure your sample under lower excitation fluence for this process to become significant"
        ]
        assert recs == expected

    def test_calculate_trpl(self, model):

        result = model.calculate_trpl(T, N_0=1e17, **BT_KWARGS)
        assert np.allclose(result[:3], np.array([1.0, 0.88948958, 0.79547423]))

    def test_calculate_trmc(self, model):

        result = model.calculate_trmc(T, N_0=1e17, **BT_KWARGS)
        assert np.allclose(result[:3], np.array([20.0, 18.86255101, 17.83787242]))


class TestBTModelTRPL:

    def test_calculate_fit_quantity(self):

        result = BTModelTRPL().calculate_fit_quantity(T, N_0=1e17, **BT_KWARGS)
        assert np.allclose(result[:3], np.array([1.0, 0.88948958, 0.79547423]))

    def test_calculate_contributions(self):

        concentrations = BTModelTRPL()._calculate_concentrations(T, 1e16, **BT_KWARGS)
        concentrations = {key: value[0] for key, value in concentrations.items()}
        contributions = BTModelTRPL().calculate_contributions(T, **concentrations, **BT_KWARGS)
        expected = {
            "T": np.float64(74.27571672613342),
            "B": np.float64(25.724244681350395),
            "A": np.float64(3.859251618686695e-05),
        }
        assert contributions == expected

    def test_get_carrier_accumulation(self):

        N0s = [1e17, 1e18]
        popts = [{"N_0": n, "I": 1.0, "y_0": 0.0, **BT_KWARGS} for n in N0s]

        # 100 ns period
        output = BTModelTRPL().get_carrier_accumulation(popts, 100)
        expected = [np.float64(2.1474605461112906), np.float64(0.3260927825203319)]
        assert are_close(output, expected)

        # 50 ns period
        output = BTModelTRPL().get_carrier_accumulation(popts, 50)
        expected = [np.float64(4.931657664085842), np.float64(0.8414419278757801)]
        assert are_close(output, expected)

    def test_generate_decays(self):

        # Without noise
        xs_data, ys_data, N0s = BTModelTRPL().generate_decays()
        assert are_close(ys_data[0][:3], [1.0, 0.99476408, 0.98955622])
        assert are_close(ys_data[-1][:3], [1.0, 0.9706254, 0.94245754])

        # With noise
        xs_data, ys_data, N0s = BTModelTRPL().generate_decays(noise=0.02)
        assert are_close(ys_data[0][:3], [0.96980343, 1.0, 0.98891807])
        assert are_close(ys_data[-1][:3], [1.0, 0.95023039, 0.95215302])

    def test_fit(self):

        # -------------------------------------------------- NO NOISE --------------------------------------------------

        test_data = BTModelTRPL().generate_decays()
        fit = BTModelTRPL().fit(*test_data)
        popt_expected = {
            "k_T": np.float64(0.009999986902280364),
            "k_B": np.float64(5.000059540779067e-19),
            "k_A": 0.0,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([96.76851866, 75.55601245, 25.64917742]),
            "B": np.array([3.23148134, 24.44398755, 74.35082258]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        assert_fit(fit, popt_expected, contribution_expected, 0.9999999999984733)

        # ---------------------------------------------------- NOISY ---------------------------------------------------

        test_data = BTModelTRPL().generate_decays(0.05)
        fit = BTModelTRPL().fit(*test_data)
        popt_expected = {
            "k_T": np.float64(0.010968908554455715),
            "k_B": np.float64(5.169626461951155e-19),
            "k_A": 0.0,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([96.95417309, 76.62826407, 26.73269065]),
            "B": np.array([3.04582691, 23.37173593, 73.26730935]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        assert_fit(fit, popt_expected, contribution_expected, 0.9419815552329127)

        # --------------------------------------------- NOISY / NON-FIXED I --------------------------------------------

        test_data = BTModelTRPL().generate_decays(0.05)
        model = BTModelTRPL()
        model.fvalues["I"] = None
        fit = model.fit(*test_data)

        popt_expected = {
            "k_T": np.float64(0.010076397709080696),
            "k_B": np.float64(4.901882357446167e-19),
            "I": np.float64(0.9320239638447994),
            "k_A": 0.0,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([96.85353317, 76.04748899, 26.1462359]),
            "B": np.array([3.14646683, 23.95251101, 73.8537641]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        assert_fit(fit, popt_expected, contribution_expected, 0.9464096079674329)

        # ------------------------------------------- NO NOISE / AUGER GUESS -------------------------------------------

        test_data = BTModelTRPL().generate_decays()
        model = BTModelTRPL()
        model.fvalues["k_A"] = None
        fit = model.fit(*test_data)
        popt_expected = {
            "k_T": np.float64(0.010000000000022824),
            "k_B": np.float64(4.999999999757379e-19),
            "k_A": np.float64(1.0000045458996652e-40),
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([96.76855947, 75.55622408, 25.64918984]),
            "B": np.array([3.23144005, 24.44373996, 74.34977381]),
            "A": np.array([4.83765434e-07, 3.59589010e-05, 1.03635229e-03]),
        }
        assert_fit(fit, popt_expected, contribution_expected, 1.0)

    def test_grid_fitting(self):

        # -------------------------------------------------- NO NOISE --------------------------------------------------

        xs_data, ys_data, N0s = BTModelTRPL().generate_decays()
        analysis = BTModelTRPL().grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": np.float64(5.000059541427541e-19),
            "k_T": np.float64(0.009999986899992526),
            "k_A": 0.0,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 1000000000000000.0,
        }
        contributions_expected = {
            "T": np.array([96.76851866, 75.55601244, 25.64917742]),
            "B": np.array([3.23148134, 24.44398756, 74.35082258]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        assert_fit(analysis[0], popt_expected, contributions_expected, 0.9999999999984733)
        assert_fit(analysis[-1], popt_expected, contributions_expected, 0.9999999999984733)

        # ---------------------------------------------------- NOISY ---------------------------------------------------

        xs_data, ys_data, N0s = BTModelTRPL().generate_decays(0.05)
        analysis = BTModelTRPL().grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": np.float64(5.169646462205936e-19),
            "k_T": np.float64(0.010968900882386862),
            "k_A": 0.0,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 1000000000000000.0,
        }
        contributions_expected = {
            "T": np.array([96.95415961, 76.62818434, 26.73260624]),
            "B": np.array([3.04584039, 23.37181566, 73.26739376]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        assert_fit(analysis[0], popt_expected, contributions_expected, 0.9419815552335047)
        assert_fit(analysis[-1], popt_expected, contributions_expected, 0.9419815552345455)

        # -------------------------------------------- NO NOISE/ NON-FIXED I -------------------------------------------

        xs_data, ys_data, N0s = BTModelTRPL().generate_decays(0.05)
        model = BTModelTRPL()
        model.fvalues["I"] = None
        analysis = model.grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": np.float64(4.901881281164643e-19),
            "k_T": np.float64(0.0100763981602312),
            "I": np.float64(0.9320239815477664),
            "k_A": 0.0,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contributions_expected = {
            "T": np.array([96.85353398, 76.04749368, 26.14624072]),
            "B": np.array([3.14646602, 23.95250632, 73.85375928]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        assert_fit(analysis[0], popt_expected, contributions_expected, 0.9464096079674144)
        assert_fit(analysis[-1], popt_expected, contributions_expected, 0.9464096079674144)

        # ------------------------------------------- NO NOISE / AUGER GUESS -------------------------------------------

        xs_data, ys_data, N0s = BTModelTRPL().generate_decays()
        model = BTModelTRPL()
        model.fvalues["k_A"] = None
        analysis = model.grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": np.float64(4.999999999517231e-19),
            "k_T": np.float64(0.010000000000056396),
            "k_A": np.float64(9.99999839030493e-41),
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 1000000000000000.0,
        }
        popt_expected_2 = {
            "k_B": np.float64(3.661606705835427e-19),
            "k_T": np.float64(0.010497516613971638),
            "k_A": np.float64(1.7842548483988976e-36),
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 1000000000000000.0,
        }
        contributions_expected = {
            "T": np.array([96.76855947, 75.55622408, 25.64918984]),
            "B": np.array([3.23144005, 24.44373996, 74.34977381]),
            "A": np.array([4.83763157e-07, 3.59587318e-05, 1.03634741e-03]),
        }
        contribution_expected_2 = {
            "T": np.array([97.7152958, 80.94098053, 27.43533307]),
            "B": np.array([2.27639629, 18.39690108, 54.37465037]),
            "A": np.array([8.30790588e-03, 6.62118386e-01, 1.81900166e01]),
        }
        assert_fit(analysis[0], popt_expected, contributions_expected, 1.0)
        assert_fit(analysis[-1], popt_expected_2, contribution_expected_2, 0.9992469395390701)


class TestBTModelTRMC:

    def test_calculate_fit_quantity(self):

        result = BTModelTRMC().calculate_trmc(T, N_0=1e17, **BT_KWARGS)
        assert np.allclose(result[:3], np.array([20.0, 18.86255101, 17.83787242]))

    def test_calculate_contributions(self):

        concentrations = BTModelTRMC()._calculate_concentrations(T, 1e16, **BT_KWARGS)
        concentrations = {key: value[0] for key, value in concentrations.items()}
        contributions = BTModelTRMC().calculate_contributions(T, **concentrations, **BT_KWARGS)
        expected = {
            "T": np.float64(76.23900060688833),
            "B": np.float64(23.760966476140023),
            "A": np.float64(3.291697165313421e-05),
        }
        assert are_close(contributions, expected)

    def test_get_carrier_accumulation(self):

        N0s = [1e17, 1e18]
        popts = [{"N_0": n, "I": 1.0, "y_0": 0.0, "mu": 10, **BT_KWARGS} for n in N0s]

        # 100 ns period
        output = BTModelTRMC().get_carrier_accumulation(popts, 100)
        expected = [np.float64(1.8119795431245034), np.float64(0.2751318097141131)]
        assert are_close(output, expected)

        # 50 ns period
        output = BTModelTRMC().get_carrier_accumulation(popts, 50)
        expected = [np.float64(4.161872397435568), np.float64(0.7099465316118103)]
        assert are_close(output, expected)

    def test_generate_decays(self):

        # Without noise
        xs_data, ys_data, N0s = BTModelTRMC().generate_decays()
        assert are_close(ys_data[0][:3], [20.0, 19.79115011, 19.58458361])
        assert are_close(ys_data[-1][:3], [20.0, 18.86255101, 17.83787242])

        # With noise
        xs_data, ys_data, N0s = BTModelTRMC().generate_decays(noise=0.02)
        assert are_close(ys_data[0][:3], [19.57021361, 20.07543598, 19.74939803])
        assert are_close(ys_data[-1][:3], [20.14851545, 18.5957746, 18.1731914])

    def test_fit(self):

        # -------------------------------------------------- NO NOISE --------------------------------------------------

        test_data = BTModelTRMC().generate_decays()
        fit = BTModelTRMC().fit(*test_data)
        popt_expected = {
            "k_T": 0.009999990976444148,
            "k_B": 5.000049109826438e-19,
            "mu": 9.999999708991059,
            "k_A": 0.0,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.58013435402015, 81.09151713962103, 35.81980709285807]),
            "B": np.array([2.419865645979866, 18.90848286037897, 64.18019290714193]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        expected_cod = 0.9999999999993268
        assert_fit(fit, popt_expected, contribution_expected, expected_cod)

        # ---------------------------------------------------- NOISY ---------------------------------------------------

        test_data = BTModelTRMC().generate_decays(0.05)
        fit = BTModelTRMC().fit(*test_data)
        popt_expected = {
            "k_T": 0.010116290480286557,
            "k_B": 4.802023013869001e-19,
            "mu": 10.05269417362719,
            "k_A": 0.0,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.69896938291939, 81.83087367000402, 36.82379525515391]),
            "B": np.array([2.3010306170806087, 18.16912632999597, 63.17620474484609]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        expected_cod = 0.9184744027161614
        assert_fit(fit, popt_expected, contribution_expected, expected_cod)

        # ------------------------------------------- NO NOISE / AUGER GUESS  ------------------------------------------

        test_data = BTModelTRMC().generate_decays()
        model = BTModelTRMC()
        model.fvalues["k_A"] = None
        fit = model.fit(*test_data)
        popt_expected = {
            "k_T": 0.009999999999996585,
            "k_B": 5.000000000015119e-19,
            "k_A": 9.99999711161973e-41,
            "mu": 9.999999999997167,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.58015916566838, 81.09165394240388, 35.819832181412764]),
            "B": np.array([2.4198405129812923, 18.908321681647912, 64.1794233913077]),
            "A": np.array([3.2135031959224633e-07, 2.4375948198990817e-05, 0.0007444272795414486]),
        }
        expected_cod = 1.0
        assert_fit(fit, popt_expected, contribution_expected, expected_cod)

    def test_grid_fitting(self):

        # -------------------------------------------------- NO NOISE --------------------------------------------------

        xs_data, ys_data, N0s = BTModelTRMC().generate_decays()
        analysis = BTModelTRMC().grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": 5.000049117261053e-19,
            "k_T": 0.009999990976318993,
            "mu": 9.999999709727632,
            "k_A": 0.0,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.58013435051565, 81.09151711799227, 35.81980706408841]),
            "B": np.array([2.4198656494843624, 18.908482882007725, 64.18019293591158]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        expected_cod = 0.9999999999993268
        assert_fit(analysis[0], popt_expected, contribution_expected, expected_cod)

        popt_expected = {
            "k_B": 5.000049117289082e-19,
            "k_T": 0.00999999097631548,
            "mu": 9.999999709742937,
            "k_A": 0.0,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.5801343504991, 81.09151711790633, 35.81980706397423]),
            "B": np.array([2.419865649500915, 18.90848288209365, 64.18019293602576]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        expected_cod = 0.9999999999993268
        assert_fit(analysis[-1], popt_expected, contribution_expected, expected_cod)

        # ---------------------------------------------------- NOISY ---------------------------------------------------

        xs_data, ys_data, N0s = BTModelTRMC().generate_decays(0.05)
        analysis = BTModelTRMC().grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": 4.802021953300244e-19,
            "k_T": 0.010116289289924205,
            "mu": 10.052693492580108,
            "k_A": 0.0,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.6989696130472, 81.83087511653866, 36.82379726558527]),
            "B": np.array([2.3010303869527995, 18.169124883461336, 63.17620273441473]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        expected_cod = 0.9184744027161531
        assert_fit(analysis[0], popt_expected, contribution_expected, expected_cod)

        popt_expected = {
            "k_B": 4.802026294974155e-19,
            "k_T": 0.010116289206037831,
            "mu": 10.052694327271345,
            "k_A": 0.0,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.69896757743622, 81.83086232245338, 36.82377951378882]),
            "B": np.array([2.301032422563783, 18.169137677546612, 63.176220486211186]),
            "A": np.array([0.0, 0.0, 0.0]),
        }
        expected_cod = 0.9184744027161734
        assert_fit(analysis[-1], popt_expected, contribution_expected, expected_cod)

        # ------------------------------------------- NO NOISE / AUGER GUESS -------------------------------------------

        xs_data, ys_data, N0s = BTModelTRMC().generate_decays()
        model = BTModelTRMC()
        model.fvalues["k_A"] = None
        analysis = model.grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": 4.999999999957749e-19,
            "k_T": 0.010000000000002705,
            "k_A": 9.999973258931717e-41,
            "mu": 9.999999999984738,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.58015916568964, 81.09165394262986, 35.819832182075714]),
            "B": np.array([2.4198405129608025, 18.908321681480082, 64.17942339242039]),
            "A": np.array([3.213495530872035e-07, 2.4375890055846062e-05, 0.0007444255039074176]),
        }
        expected_cod = 1.0
        assert_fit(analysis[0], popt_expected, contribution_expected, expected_cod)

        popt_expected = {
            "k_B": 3.898375046878196e-19,
            "k_T": 0.010363531958922218,
            "k_A": 2.3293175945341933e-36,
            "mu": 10.106211747446103,
            "y_0": 0.0,
            "N_0": 1000000000000000.0,
        }
        contribution_expected = {
            "T": np.array([98.15786554403202, 84.42608368385423, 36.67047807117695]),
            "B": np.array([1.8348477970950738, 14.993304359125895, 47.42593643224026]),
            "A": np.array([0.007286658872892999, 0.5806119570198793, 15.903585496582778]),
        }
        expected_cod = 0.9995910035912059
        assert_fit(analysis[-1], popt_expected, contribution_expected, expected_cod)


# ------------------------------------------------------ BTD MODEL -----------------------------------------------------


class TestBTDModel:

    @pytest.fixture
    def model(self):

        return BTDModel(["k_B", "k_T", "k_D", "N_T", "p_0", "mu_e", "mu_h"])

    def test_initialization(self, model):

        assert model.units["k_B"] == "cm3/ns"
        assert model.factors["k_B"] == 1e-20
        assert model.n_keys == ["n_e", "n_t", "n_h"]

    def test_rate_equations(self, model):

        result = model._rate_equations(n_e=1e17, n_t=1e12, n_h=1e17, **BTD_KWARGS)
        assert result == {"n_e": -5711250000000000.0, "n_t": 707919948000000.0, "n_h": -5003330052000000.0}

    def test_calculate_concentrations(self, model):

        # Single pulse
        output = model._calculate_concentrations(T, 1e17, **BTD_KWARGS)
        assert np.allclose(output["n_e"][0][:3], np.array([1.00000000e17, 9.51739441e16, 9.08408676e16]))
        assert np.allclose(output["n_t"][0][:3], np.array([0.00000000e00, 5.96016786e13, 5.96021105e13]))
        assert np.allclose(output["n_h"][0][:3], np.array([1.00000000e17, 9.52335458e16, 9.09004697e16]))
        assert np.allclose(output["n_e"][0][:3] + output["n_t"][0][:3], output["n_h"][0][:3])

        # Multiple pulses
        output = model._calculate_concentrations(T, 1e17, **BTD_KWARGS, p=1000)
        assert np.allclose(output["n_e"][-1][:3], np.array([1.16972060e17, 1.10497012e17, 1.04700620e17]))
        assert np.allclose(output["n_t"][-1][:3], np.array([5.95997634e13, 5.96022059e13, 5.96021815e13]))
        assert np.allclose(output["n_h"][-1][:3], np.array([1.17031660e17, 1.10556615e17, 1.04760222e17]))
        assert np.allclose(output["n_e"][-1][:3] + output["n_t"][-1][:3], output["n_h"][-1][:3])
        assert len(output["n_e"]) == 5

        # Single pulse (with additional argument)
        output = model._calculate_concentrations(T, 1e17, **BTD_KWARGS)
        assert np.allclose(output["n_e"][0][:3], np.array([1.00000000e17, 9.51739441e16, 9.08408676e16]))
        assert np.allclose(output["n_t"][0][:3], np.array([0.00000000e00, 5.96016786e13, 5.96021105e13]))
        assert np.allclose(output["n_h"][0][:3], np.array([1.00000000e17, 9.52335458e16, 9.09004697e16]))
        assert np.allclose(output["n_e"][0][:3] + output["n_t"][0][:3], output["n_h"][0][:3])

    def test_get_carrier_concentrations(self, model):

        popts = [{"I0": 1.0, "N_0": 1e17, **BTD_KWARGS}]

        # Period provided
        output = model.get_carrier_concentrations([T], popts, 100)
        assert np.allclose(output[2][0]["n_e"][:3], np.array([1.0000000e17, 9.9460356e16, 9.8954829e16]))
        assert np.allclose(output[2][0]["n_t"][:3], np.array([0.00000000e00, 4.17344709e13, 5.42120112e13]))
        assert np.allclose(output[2][0]["n_h"][:3], np.array([1.00000000e17, 9.95020905e16, 9.90090410e16]))
        assert np.allclose(output[0][0][:3], np.array([0.0, 0.00099901, 0.00199802]))

        # Period not provided
        output2 = model.get_carrier_concentrations([T], popts, 0)
        assert np.allclose(output2[2][0]["n_e"][:3], np.array([1.00000000e17, 9.51739441e16, 9.08408676e16]))
        assert np.allclose(output2[2][0]["n_t"][:3], np.array([0.00000000e00, 5.96016786e13, 5.96021105e13]))
        assert np.allclose(output2[2][0]["n_h"][:3], np.array([1.00000000e17, 9.52335458e16, 9.09004697e16]))
        assert np.allclose(output2[0][0][:3], np.array([0.0, 1.0, 2.0]))

    def test_get_contribution_recommendations(self, model):

        contributions = {"B": np.array([5]), "T": np.array([15]), "D": np.array([8])}
        recs = model.get_contribution_recommendations(contributions)
        expected = [
            "This fit predicts low bimolecular. The values associated with this process may be inaccurate.\nIt is "
            "recommended to measure your sample under higher excitation fluence for this process to become significant",
            "This fit predicts low detrapping. The values associated with this process may be inaccurate.",
            "Note: For the bimolecular-trapping-detrapping model, although a low contribution suggests that the parameter "
            "associated with the process are not be accurate, a non-negligible contribution does not automatically "
            "indicate that the parameters retrieved are accurate due to the complex nature of the model. It is recommended "
            "to perform a grid fitting analysis with this model.",
        ]
        assert recs == expected

    def test_calculate_trpl(self, model):

        result = model.calculate_trpl(T, N_0=1e15, **BTD_KWARGS)
        assert np.allclose(result[:3], np.array([1.0, 0.99221134, 0.98523199]))

    def test_calculate_trmc(self, model):

        result = model.calculate_trmc(T, N_0=1e17, **BTD_KWARGS)
        assert np.allclose(result[:3], np.array([50.0, 47.60485258, 45.43831441]))


class TestBTDModelTRPL:

    def test_calculate_fit_quantity(self):

        result = BTDModelTRPL().calculate_fit_quantity(T, N_0=1e15, **BTD_KWARGS)
        assert np.allclose(result[:3], np.array([1.0, 0.99221134, 0.98523199]))

    def test_calculate_contributions(self):

        concentrations = BTDModelTRPL()._calculate_concentrations(T, 1e15, **BTD_KWARGS)
        concentrations = {key: value[0] for key, value in concentrations.items()}
        contributions = BTDModelTRPL().calculate_contributions(T, **concentrations, **BTD_KWARGS)
        expected = {
            "T": np.float64(41.01551259497901),
            "B": np.float64(56.48980101942542),
            "D": np.float64(2.494686385595578),
        }
        assert contributions == expected

    def test_get_carrier_accumulation(self):

        N0s = [1e17, 1e18]
        popts = [{"N_0": n, "I": 1.0, "y_0": 0.0, **BTD_KWARGS} for n in N0s]

        # 100 ns period
        output = BTDModelTRPL().get_carrier_accumulation(popts, 100)
        expected = [np.float64(4.624820971652416), np.float64(0.5713873355827237)]
        assert are_close(output, expected)

        # 50 ns period
        output = BTDModelTRPL().get_carrier_accumulation(popts, 50)
        expected = [np.float64(7.852802160625521), np.float64(1.1155396319428357)]
        assert are_close(output, expected)

    def test_generate_decays(self):

        # Without noise
        xs_data, ys_data, N0s = BTDModelTRPL().generate_decays()
        assert are_close(ys_data[0][:3], [1.0, 0.93171234, 0.87145003])
        assert are_close(ys_data[-1][:3], [1.0, 0.9437801, 0.90364939])

        # With noise
        xs_data, ys_data, N0s = BTDModelTRPL().generate_decays(noise=0.02)
        assert are_close(ys_data[0][:3], [1.0, 0.96670037, 0.89900986])
        assert are_close(ys_data[-1][:3], [1.0, 0.98174779, 0.93649283])

    def test_fit(self):

        # -------------------------------------------------- NO NOISE --------------------------------------------------

        test_data = BTDModelTRPL().generate_decays()
        fit = BTDModelTRPL().fit(*test_data)
        popt_expected = {
            "k_B": 5.000000052524678e-19,
            "k_T": 1.2000000020860115e-16,
            "k_D": 7.99999965517838e-19,
            "N_T": 59999999908205.234,
            "p_0": 65000005482068.65,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.55966, 67.41144, 28.74199, 10.17702, 5.60928]),
            "B": np.array([1.80154, 23.92938, 61.86576, 85.42388, 92.5306]),
            "D": np.array([0.6388, 8.65918, 9.39225, 4.3991, 1.86012]),
        }
        expected_cod = 1.0
        assert_fit(fit, popt_expected, contribution_expected, expected_cod)

        # ---------------------------------------------------- NOISY ---------------------------------------------------

        test_data = BTDModelTRPL().generate_decays(0.05)
        fit = BTDModelTRPL().fit(*test_data)
        popt_expected = {
            "k_B": 5.281696267903158e-19,
            "k_T": 1.2005725792145259e-16,
            "k_D": 5.907688024451682e-19,
            "N_T": 58111233208509.96,
            "p_0": 108786247871822.42,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([96.85071, 63.90628, 25.56837, 8.67334, 4.91477]),
            "B": np.array([2.65371, 29.80912, 67.7983, 88.26982, 93.80589]),
            "D": np.array([0.49558, 6.2846, 6.63334, 3.05684, 1.27934]),
        }
        expected_cod = 0.9219066553006975
        assert_fit(fit, popt_expected, contribution_expected, expected_cod)

        # --------------------------------------------- NOISY / NON-FIXED I --------------------------------------------

        test_data = BTDModelTRPL().generate_decays(0.05)
        model = BTDModelTRPL()
        model.fvalues["I"] = None
        fit = model.fit(*test_data)
        popt_expected = {
            "k_B": 5.231708686847702e-19,
            "k_T": 1.187069255529963e-16,
            "k_D": 6.120441223117572e-19,
            "N_T": 60333138581717.695,
            "p_0": 97515750573609.22,
            "I": 1.0431146652007544,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.19901, 65.60995, 26.60699, 9.08448, 5.11529]),
            "B": np.array([2.31978, 27.78817, 66.25586, 87.61144, 93.49959]),
            "D": np.array([0.48121, 6.60188, 7.13715, 3.30408, 1.38512]),
        }
        expected_cod = 0.9221545665271867
        assert_fit(fit, popt_expected, contribution_expected, expected_cod)

    def test_grid_fitting(self):

        model = BTDModelTRPL()
        model.gvalues_range["k_B"] = [1e-20]
        model.gvalues_range["k_T"] = [1e-16]
        model.gvalues_range["p_0"] = [1e14]

        # -------------------------------------------------- NO NOISE --------------------------------------------------

        xs_data, ys_data, N0s = model.generate_decays()
        analysis = model.grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": 5.044092519858126e-19,
            "k_T": 1.221507144388288e-16,
            "k_D": 3.2627489522773756e-19,
            "p_0": 201606458996472.16,
            "N_T": 58686486048362.56,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([96.06959, 59.97392, 22.63214, 7.46053, 4.46979]),
            "B": np.array([3.67138, 36.55816, 73.6027, 90.75852, 94.78018]),
            "D": np.array([0.25903, 3.46791, 3.76516, 1.78095, 0.75003]),
        }
        expected_cod = 0.9999689786967758
        assert_fit(analysis[0], popt_expected, contribution_expected, expected_cod)
        popt_expected = {
            "k_B": 5.0000000437758285e-19,
            "k_T": 1.200000008803637e-16,
            "k_D": 8.000000823783076e-19,
            "p_0": 64999983771959.49,
            "N_T": 59999999918573.16,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.55966, 67.41144, 28.74199, 10.17702, 5.60928]),
            "B": np.array([1.80154, 23.92937, 61.86576, 85.42388, 92.5306]),
            "D": np.array([0.6388, 8.65918, 9.39225, 4.3991, 1.86012]),
        }
        expected_cod = 0.9999999999999998
        assert_fit(analysis[-1], popt_expected, contribution_expected, expected_cod)

        # ---------------------------------------------------- NOISY ---------------------------------------------------

        xs_data, ys_data, N0s = model.generate_decays(0.05)
        analysis = model.grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": 5.281625280355903e-19,
            "k_T": 1.2005402935900586e-16,
            "k_D": 5.911623649351193e-19,
            "p_0": 108688274641093.72,
            "N_T": 58112578119929.21,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([96.85187, 63.91184, 25.5731, 8.67546, 4.91564]),
            "B": np.array([2.65221, 29.79925, 67.78904, 88.26562, 93.80414]),
            "D": np.array([0.49592, 6.28891, 6.63786, 3.05892, 1.28022]),
        }
        expected_cod = 0.9219066554530871
        assert_fit(analysis[0], popt_expected, contribution_expected, expected_cod)
        popt_expected = {
            "k_B": 5.281014612329833e-19,
            "k_T": 1.2004301649017149e-16,
            "k_D": 5.952924043139263e-19,
            "p_0": 107673071106750.62,
            "N_T": 58122054681198.03,
            "y_0": 0.0,
            "I": 1.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([96.86386, 63.9668, 25.62092, 8.69737, 4.92498]),
            "B": np.array([2.63673, 29.69927, 67.6943, 88.22221, 93.78573]),
            "D": np.array([0.49942, 6.33393, 6.68478, 3.08042, 1.28929]),
        }
        expected_cod = 0.9219066470404309
        assert_fit(analysis[-1], popt_expected, contribution_expected, expected_cod)

        # -------------------------------------------- NO NOISE/ NON-FIXED I -------------------------------------------

        xs_data, ys_data, N0s = model.generate_decays(0.05)
        model.fvalues["I"] = None
        model.gvalues_range["k_D"] = [1e-18]
        analysis = model.grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": 5.233892964974858e-19,
            "k_T": 1.1867745425973867e-16,
            "k_D": 6.055792534709746e-19,
            "p_0": 98983350486299.22,
            "N_T": 60330106371418.1,
            "I": 1.0430840799173018,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.18283, 65.53522, 26.53384, 9.04918, 5.09949]),
            "B": np.array([2.34114, 27.93346, 66.40415, 87.68164, 93.53021]),
            "D": np.array([0.47603, 6.53132, 7.06201, 3.26918, 1.3703]),
        }
        expected_cod = 0.9221545717141318
        assert_fit(analysis[0], popt_expected, contribution_expected, expected_cod)
        popt_expected = {
            "k_B": 5.231318664511353e-19,
            "k_T": 1.1868317014121706e-16,
            "k_D": 6.1288804671095715e-19,
            "p_0": 97302985162596.94,
            "N_T": 60340004786445.39,
            "I": 1.0431027851890577,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([97.20144, 65.62315, 26.61848, 9.08953, 5.11716]),
            "B": np.array([2.31668, 27.76541, 66.23378, 87.60143, 93.49561]),
            "D": np.array([0.48187, 6.61144, 7.14774, 3.30904, 1.38723]),
        }
        expected_cod = 0.9221545624359501
        assert_fit(analysis[-1], popt_expected, contribution_expected, expected_cod)


class TestBTDModelTRMC:

    def test_calculate_fit_quantity(self):

        result = BTDModelTRMC().calculate_fit_quantity(T, N_0=1e17, **BTD_KWARGS)
        assert np.allclose(result[:3], np.array([50.0, 47.60485258, 45.43831441]))

    def test_calculate_contributions(self):

        concentrations = BTDModelTRMC()._calculate_concentrations(T, 1e15, **BTD_KWARGS)
        concentrations = {key: value[0] for key, value in concentrations.items()}
        contributions = BTDModelTRMC().calculate_contributions(T, **concentrations, **BTD_KWARGS)
        expected = {
            "T": np.float64(33.7114972857804),
            "B": np.float64(62.7295706296424),
            "D": np.float64(3.558932084577181),
        }
        assert contributions == expected

    def test_get_carrier_accumulation(self):

        N0s = [1e17, 1e18]
        popts = [{"N_0": n, **BTD_KWARGS} for n in N0s]

        # 100 ns period
        output = BTDModelTRMC().get_carrier_accumulation(popts, 100)
        expected = [np.float64(3.912967640970455), np.float64(0.48314189311882694)]
        assert are_close(output, expected)

        # 50 ns period
        output = BTDModelTRMC().get_carrier_accumulation(popts, 50)
        expected = [np.float64(6.63910507746735), np.float64(0.9422743536986633)]
        assert are_close(output, expected)

    def test_generate_decays(self):

        # Without noise
        xs_data, ys_data, N0s = BTDModelTRMC().generate_decays()
        assert are_close(ys_data[0][:3], [50.0, 44.46133618, 41.08877032])
        assert are_close(ys_data[-1][:3], [50.0, 44.68626895, 40.55753199])

        # With noise
        xs_data, ys_data, N0s = BTDModelTRMC().generate_decays(noise=0.02)
        assert are_close(ys_data[0][:3], [48.92553402, 45.17205085, 41.50080638])
        assert are_close(ys_data[-1][:3], [49.25755246, 45.85575692, 41.50440705])

    def test_fit(self):

        # -------------------------------------------------- NO NOISE --------------------------------------------------

        test_data = BTDModelTRMC().generate_decays()
        fit = BTDModelTRMC().fit(*test_data)
        popt_expected = {
            "k_B": 5.000000164708902e-19,
            "k_T": 1.200000031835309e-16,
            "k_D": 7.999999808965121e-19,
            "N_T": 59999999376176.54,
            "p_0": 65000001818720.9,
            "mu_e": 20.000000281407857,
            "mu_h": 29.99999994350886,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([40.90276, 32.92583, 20.94199, 13.13185, 9.81368]),
            "B": np.array([1.59628, 21.39947, 52.96585, 75.52489, 84.90275]),
            "D": np.array([57.50096, 45.6747, 26.09216, 11.34326, 5.28357]),
        }
        expected_cod = 1.0
        assert_fit(fit, popt_expected, contribution_expected, expected_cod)

        # ---------------------------------------------------- NOISY ---------------------------------------------------

        test_data = BTDModelTRMC().generate_decays(0.05)
        fit = BTDModelTRMC().fit(*test_data)
        popt_expected = {
            "k_B": 5.170253809634543e-19,
            "k_T": 1.1017602528332628e-16,
            "k_D": 8.109404996553888e-19,
            "N_T": 57522615670991.11,
            "p_0": 63326777368258.42,
            "mu_e": 20.819525615904908,
            "mu_h": 29.662688836182774,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([41.93316, 33.03053, 20.50887, 12.50261, 9.1934]),
            "B": np.array([1.91827, 23.1565, 54.71674, 76.79712, 85.84359]),
            "D": np.array([56.14857, 43.81296, 24.7744, 10.70027, 4.96301]),
        }
        expected_cod = 0.9072435319931186
        assert_fit(fit, popt_expected, contribution_expected, expected_cod)

    def test_grid_fitting(self):

        model = BTDModelTRMC()
        model.gvalues_range["k_B"] = [1e-20]
        model.gvalues_range["k_T"] = [1e-16]
        model.gvalues_range["k_D"] = [1e-18]
        model.gvalues_range["p_0"] = [1e14]

        # -------------------------------------------------- NO NOISE --------------------------------------------------

        xs_data, ys_data, N0s = model.generate_decays()
        analysis = model.grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": 5.000000263245968e-19,
            "k_T": 1.200000041725577e-16,
            "k_D": 7.999999617651796e-19,
            "p_0": 65000003889440.46,
            "N_T": 59999999181832.375,
            "mu_e": 20.000000542628477,
            "mu_h": 29.9999999067198,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([40.90276, 32.92583, 20.94199, 13.13185, 9.81368]),
            "B": np.array([1.59628, 21.39947, 52.96585, 75.52489, 84.90275]),
            "D": np.array([57.50096, 45.6747, 26.09216, 11.34326, 5.28357]),
        }
        expected_cod = 0.9999999999999999
        assert_fit(analysis[0], popt_expected, contribution_expected, expected_cod)

        popt_expected = {
            "k_B": 5.000000344817324e-19,
            "k_T": 1.2000000694970784e-16,
            "k_D": 7.999999442799611e-19,
            "p_0": 65000005667179.016,
            "N_T": 59999998470043.97,
            "mu_e": 20.00000063847375,
            "mu_h": 29.999999859906556,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([40.90276, 32.92583, 20.94199, 13.13185, 9.81368]),
            "B": np.array([1.59628, 21.39947, 52.96585, 75.52489, 84.90275]),
            "D": np.array([57.50096, 45.6747, 26.09216, 11.34326, 5.28357]),
        }
        expected_cod = 0.9999999999999994
        assert_fit(analysis[-1], popt_expected, contribution_expected, expected_cod)

        # ---------------------------------------------------- NOISY ---------------------------------------------------

        xs_data, ys_data, N0s = model.generate_decays(0.05)
        analysis = model.grid_fitting(None, N0s, xs_data=xs_data, ys_data=ys_data)
        popt_expected = {
            "k_B": 5.169968590421277e-19,
            "k_T": 1.1017306106751546e-16,
            "k_D": 8.120500700336253e-19,
            "p_0": 63209041226835.805,
            "N_T": 57509469769832.9,
            "mu_e": 20.816413593836856,
            "mu_h": 29.663528347986045,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([41.92978, 33.03175, 20.51353, 12.50502, 9.19378]),
            "B": np.array([1.91722, 23.14519, 54.69963, 76.78697, 85.83906]),
            "D": np.array([56.153, 43.82306, 24.78684, 10.70801, 4.96716]),
        }
        expected_cod = 0.9072435361184141
        assert_fit(analysis[0], popt_expected, contribution_expected, expected_cod)
        popt_expected = {
            "k_B": 5.170343293253923e-19,
            "k_T": 1.1018194467749615e-16,
            "k_D": 8.121236402177761e-19,
            "p_0": 63200380749480.586,
            "N_T": 57501357284825.3,
            "mu_e": 20.817089249169957,
            "mu_h": 29.662940880799475,
            "y_0": 0.0,
            "N_0": 51000000000000.0,
        }
        contribution_expected = {
            "T": np.array([41.93091, 33.03105, 20.51274, 12.50452, 9.19342]),
            "B": np.array([1.91766, 23.14931, 54.70297, 76.78871, 85.84002]),
            "D": np.array([56.15143, 43.81964, 24.78429, 10.70677, 4.96656]),
        }
        expected_cod = 0.9072435358895351
        assert_fit(analysis[-1], popt_expected, contribution_expected, expected_cod)
