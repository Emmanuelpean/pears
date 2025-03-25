import numpy as np
import pytest

from app import models
from app import resources
from app import utils
from app.fitting import Fit, run_grid_fit


@pytest.fixture
def gaussian_data():
    """Gaussian data"""

    def gaussian(x, a, x0, c):
        return a * np.exp(-((x - x0) ** 2) / (2 * c))

    xs_data = [np.linspace(-2, 5, 101)] * 3
    ys_data = [
        gaussian(xs_data[0], 1e8, 1, 0.5),
        gaussian(xs_data[0], 3e8, 1, 0.25),
        gaussian(xs_data[0], 5e8, 1, 0.25),
    ]
    p0 = dict(a=1e8, x0=0, c=0.3)
    detached_parameters = ["a"]
    fixed_parameters = [dict(c=0.5), dict(c=0.25), dict(c=0.25)]
    return xs_data, ys_data, gaussian, p0, detached_parameters, fixed_parameters


@pytest.fixture
def trpl_bt_data():
    """TRPL data"""

    x_data, *ys_data = resources.BT_TRPL_DATA
    xs_data = [x_data] * len(ys_data)
    xs_data, ys_data = utils.process_data(xs_data, ys_data, True)
    function = models.BTModelTRPL().calculate_fit_quantity
    p0 = dict(k_T=1e-3, k_B=1e-20, k_A=1e-40, y_0=0.01, I=1)
    detached_parameters = ["y_0", "I"]
    fixed_parameters = [dict(N_0=1e15), dict(N_0=1e16), dict(N_0=1e17)]
    return xs_data, ys_data, function, p0, detached_parameters, fixed_parameters


@pytest.fixture
def trpl_btd_data():
    """TRPL BTD data"""

    xdata, *ys_data = resources.BTD_TRPL_DATA
    xs_data = [xdata] * len(ys_data)
    function = models.BTDModelTRPL().calculate_fit_quantity
    fixed_parameters = [dict(N_0=n) for n in (55e12, 164e12, 511e12, 1720e12, 4750e12)]
    p0 = dict(k_B=10e-20, k_T=1000e-20, k_D=100e-20, p_0=1e12, N_T=1e12, y_0=0.0, I=1.0)
    detached_parameters = ["y_0", "I"]
    return xs_data, ys_data, function, p0, detached_parameters, fixed_parameters


@pytest.fixture
def trmc_btd_data():
    """TRMC data"""

    x_data, *ys_data = resources.BTD_TRMC_DATA
    xs_data = [x_data] * len(ys_data)
    function = models.BTDModelTRMC().calculate_fit_quantity
    p0 = dict(k_T=1e-16, k_B=25e-20, k_D=80e-20, N_T=1e14, p_0=1e14, y_0=0.01, mu_e=10.0, mu_h=10.0)
    detached_parameters = []
    fixed_parameters = [dict(N_0=n, y_0=0) for n in (55e12, 164e12, 511e12, 1720e12, 4750e12)]
    return xs_data, ys_data, function, p0, detached_parameters, fixed_parameters


@pytest.fixture
def trmc_btd_data2():
    """TRMC data 2"""

    xs_data, ys_data = resources.BTD_TRMC_DATA_2[::2], resources.BTD_TRMC_DATA_2[1::2]
    function = models.BTDModelTRMC().calculate_fit_quantity
    p0 = dict(k_T=1e-16, k_B=25e-20, k_D=80e-20, N_T=1e14, p_0=1e14, y_0=0.01, mu_e=10.0, mu_h=10.0)
    detached_parameters = []
    fixed_parameters = [dict(N_0=n, y_0=0) for n in (55e12, 164e12, 511e12, 1720e12, 4750e12)]
    return xs_data, ys_data, function, p0, detached_parameters, fixed_parameters


class TestFit:

    @pytest.fixture
    def gaussian_fit(self, gaussian_data):
        """Gaussian fit object"""

        xs_data, ys_data, function, p0, detached_parameters, fixed_parameters = gaussian_data
        return Fit(xs_data, ys_data, function, p0, detached_parameters, fixed_parameters)

    @pytest.fixture
    def trpl_bt_fit(self, trpl_bt_data):
        """TRPL fit object"""

        xs_data, ys_data, function, p0, detached_parameters, fixed_parameters = trpl_bt_data
        return Fit(xs_data, ys_data, function, p0, detached_parameters, fixed_parameters)

    @pytest.fixture
    def trpl_btd_fit(self, trpl_btd_data):
        """TRPL fit object"""

        xs_data, ys_data, function, p0, detached_parameters, fixed_parameters = trpl_btd_data
        return Fit(xs_data, ys_data, function, p0, detached_parameters, fixed_parameters)

    @pytest.fixture
    def trmc_btd_fit2(self, trmc_btd_data2):
        """TRMC fit object"""

        xs_data, ys_data, function, p0, detached_parameters, fixed_parameters = trmc_btd_data2
        return Fit(xs_data, ys_data, function, p0, detached_parameters, fixed_parameters)

    @pytest.fixture
    def trmc_btd_fit(self, trmc_btd_data):
        """TRMC fit object"""

        xs_data, ys_data, function, p0, detached_parameters, fixed_parameters = trmc_btd_data
        return Fit(xs_data, ys_data, function, p0, detached_parameters, fixed_parameters)

    def test_gaussian_creation(self, gaussian_fit):

        assert gaussian_fit.p0_mantissa == {"a": 1, "x0": 0.0}
        assert gaussian_fit.p0_factors == {"a": 8, "x0": 0}
        assert gaussian_fit.p0_list == [1.0, 0.0, 1.0, 1.0]
        assert gaussian_fit.bounds == {"a": [0, np.inf], "x0": [0, np.inf]}
        assert gaussian_fit.bounds_list == [[0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf]]

    def test_gaussian_incorrect_fixed_parameters(self, gaussian_data):

        xs_data, ys_data, gaussian, p0, detached_parameters = gaussian_data[:-1]
        fixed_parameters = [dict(c=0.5), dict()]
        with pytest.raises(AssertionError):
            Fit(xs_data, ys_data, gaussian, p0, detached_parameters, fixed_parameters)

    # -------------------------------------------------- list_to_dicts -------------------------------------------------

    def test_gaussian_list_to_dicts(self, gaussian_fit):

        expected = [
            {"a": 100000000.0, "x0": 0.0},
            {"a": np.float64(100000000.0), "x0": 0.0},
            {"a": np.float64(100000000.0), "x0": 0.0},
        ]
        assert gaussian_fit.list_to_dicts(gaussian_fit.p0_list) == expected

    def test_trpl_list_to_dicts(self, trpl_bt_fit):

        expected = [
            {
                "k_T": 0.001,
                "k_B": 1.0000000000000005e-20,
                "k_A": 1.0000000000000005e-40,
                "y_0": 0.010000000000000002,
                "I": 1.0,
            },
            {
                "y_0": np.float64(0.010000000000000002),
                "k_T": 0.001,
                "k_B": 1.0000000000000005e-20,
                "k_A": 1.0000000000000005e-40,
                "I": 1.0,
            },
            {
                "y_0": np.float64(0.010000000000000002),
                "k_T": 0.001,
                "k_B": 1.0000000000000005e-20,
                "k_A": 1.0000000000000005e-40,
                "I": 1.0,
            },
        ]
        assert trpl_bt_fit.list_to_dicts(trpl_bt_fit.p0_list) == expected

    def test_trmc_list_to_dicts(self, trmc_btd_fit):

        expected = [
            {
                "N_T": 100000000000000.0,
                "k_B": 2.5000000000000016e-19,
                "k_D": 8.000000000000006e-19,
                "k_T": 1.0000000000000007e-16,
                "mu_e": 10.0,
                "mu_h": 10.0,
                "p_0": 100000000000000.0,
            },
            {
                "N_T": 100000000000000.0,
                "k_B": 2.5000000000000016e-19,
                "k_D": 8.000000000000006e-19,
                "k_T": 1.0000000000000007e-16,
                "mu_e": 10.0,
                "mu_h": 10.0,
                "p_0": 100000000000000.0,
            },
            {
                "N_T": 100000000000000.0,
                "k_B": 2.5000000000000016e-19,
                "k_D": 8.000000000000006e-19,
                "k_T": 1.0000000000000007e-16,
                "mu_e": 10.0,
                "mu_h": 10.0,
                "p_0": 100000000000000.0,
            },
            {
                "N_T": 100000000000000.0,
                "k_B": 2.5000000000000016e-19,
                "k_D": 8.000000000000006e-19,
                "k_T": 1.0000000000000007e-16,
                "mu_e": 10.0,
                "mu_h": 10.0,
                "p_0": 100000000000000.0,
            },
            {
                "N_T": 100000000000000.0,
                "k_B": 2.5000000000000016e-19,
                "k_D": 8.000000000000006e-19,
                "k_T": 1.0000000000000007e-16,
                "mu_e": 10.0,
                "mu_h": 10.0,
                "p_0": 100000000000000.0,
            },
        ]

        assert trmc_btd_fit.list_to_dicts(trmc_btd_fit.p0_list) == expected

    # ------------------------------------------------- error_function -------------------------------------------------

    def test_gaussian_error_function(self, gaussian_fit):

        expected = np.array([1819222.90846475, 2392860.49712516])
        assert np.allclose(gaussian_fit.error_function(gaussian_fit.p0_list)[:2], expected)

    def test_bt_trpl_error_function(self, trpl_bt_fit):

        expected = np.array([0.01, 0.06214057])
        assert np.allclose(trpl_bt_fit.error_function(trpl_bt_fit.p0_list)[:2], expected)

    def test_btd_trpl_error_function(self, trpl_btd_fit):

        expected = np.array([-0.01150735, 0.12307265])
        assert np.allclose(trpl_btd_fit.error_function(trpl_btd_fit.p0_list)[:2], expected)

    def test_btd_trmc_error_function(self, trmc_btd_fit):

        expected = np.array([-67.0, -66.99998181])
        assert np.allclose(trmc_btd_fit.error_function(trmc_btd_fit.p0_list)[:2], expected)

    def test_btd_trmc2_error_function(self, trmc_btd_fit2):

        expected = np.array([-40.40590897, -48.61018384])
        assert np.allclose(trmc_btd_fit2.error_function(trmc_btd_fit2.p0_list)[:2], expected)

    # ------------------------------------------------------- fit ------------------------------------------------------

    def test_gaussian_fit(self, gaussian_fit):

        expected = [
            {"a": 100000000.0, "x0": 1.0, "c": 0.5},
            {"a": 300000000.0, "x0": 1.0, "c": 0.25},
            {"a": 500000000.0, "x0": 1.0, "c": 0.25},
        ]
        assert gaussian_fit.fit() == expected

    def test_bt_trpl_fit(self, trpl_bt_fit):

        expected = [
            {
                "k_T": np.float64(0.004584339786301158),
                "k_B": np.float64(1.1131745473770256e-19),
                "k_A": np.float64(5.69878025416592e-50),
                "y_0": np.float64(0.03612514047499745),
                "I": np.float64(0.8625791894434233),
                "N_0": 1000000000000000.0,
            },
            {
                "y_0": np.float64(0.014801532068305062),
                "I": np.float64(0.8833857929049156),
                "k_T": np.float64(0.004584339786301158),
                "k_B": np.float64(1.1131745473770256e-19),
                "k_A": np.float64(5.69878025416592e-50),
                "N_0": 1e16,
            },
            {
                "y_0": np.float64(0.006609966715367191),
                "I": np.float64(1.0065354059671912),
                "k_T": np.float64(0.004584339786301158),
                "k_B": np.float64(1.1131745473770256e-19),
                "k_A": np.float64(5.69878025416592e-50),
                "N_0": 1e17,
            },
        ]
        assert trpl_bt_fit.fit() == expected

    def test_btd_trpl_fit(self, trpl_btd_fit):

        expected = [
            {
                "I": np.float64(1.0128495783208284),
                "N_0": 55000000000000.0,
                "N_T": np.float64(59841335760098.56),
                "k_B": np.float64(2.594539213598768e-19),
                "k_D": np.float64(8.355779666124636e-19),
                "k_T": np.float64(1.2283178241222857e-16),
                "p_0": np.float64(59878037988656.27),
                "y_0": np.float64(5.21644014106968e-23),
            },
            {
                "I": np.float64(0.9987039532459023),
                "N_0": 164000000000000.0,
                "N_T": np.float64(59841335760098.56),
                "k_B": np.float64(2.594539213598768e-19),
                "k_D": np.float64(8.355779666124636e-19),
                "k_T": np.float64(1.2283178241222857e-16),
                "p_0": np.float64(59878037988656.27),
                "y_0": np.float64(0.00033500850639488135),
            },
            {
                "I": np.float64(1.0003554561891748),
                "N_0": 511000000000000.0,
                "N_T": np.float64(59841335760098.56),
                "k_B": np.float64(2.594539213598768e-19),
                "k_D": np.float64(8.355779666124636e-19),
                "k_T": np.float64(1.2283178241222857e-16),
                "p_0": np.float64(59878037988656.27),
                "y_0": np.float64(0.00019022451799703384),
            },
            {
                "I": np.float64(1.0036584819460788),
                "N_0": 1720000000000000.0,
                "N_T": np.float64(59841335760098.56),
                "k_B": np.float64(2.594539213598768e-19),
                "k_D": np.float64(8.355779666124636e-19),
                "k_T": np.float64(1.2283178241222857e-16),
                "p_0": np.float64(59878037988656.27),
                "y_0": np.float64(9.452358202348823e-05),
            },
            {
                "I": np.float64(0.9942773617468111),
                "N_0": 4750000000000000.0,
                "N_T": np.float64(59841335760098.56),
                "k_B": np.float64(2.594539213598768e-19),
                "k_D": np.float64(8.355779666124636e-19),
                "k_T": np.float64(1.2283178241222857e-16),
                "p_0": np.float64(59878037988656.27),
                "y_0": np.float64(0.0001634993553182116),
            },
        ]
        assert trpl_btd_fit.fit() == expected

    def test_trmc_fit(self, trmc_btd_fit):

        expected = [
            {
                "N_0": 55000000000000.0,
                "N_T": np.float64(56846624270121.38),
                "k_B": np.float64(7.021105074240901e-20),
                "k_D": np.float64(3.244976806038882e-19),
                "k_T": np.float64(1.1738541743820234e-16),
                "mu_e": np.float64(29.167501568118773),
                "mu_h": np.float64(57.83322022771485),
                "p_0": np.float64(15958840562808.432),
                "y_0": 0,
            },
            {
                "N_0": 164000000000000.0,
                "N_T": np.float64(56846624270121.38),
                "k_B": np.float64(7.021105074240901e-20),
                "k_D": np.float64(3.244976806038882e-19),
                "k_T": np.float64(1.1738541743820234e-16),
                "mu_e": np.float64(29.167501568118773),
                "mu_h": np.float64(57.83322022771485),
                "p_0": np.float64(15958840562808.432),
                "y_0": 0,
            },
            {
                "N_0": 511000000000000.0,
                "N_T": np.float64(56846624270121.38),
                "k_B": np.float64(7.021105074240901e-20),
                "k_D": np.float64(3.244976806038882e-19),
                "k_T": np.float64(1.1738541743820234e-16),
                "mu_e": np.float64(29.167501568118773),
                "mu_h": np.float64(57.83322022771485),
                "p_0": np.float64(15958840562808.432),
                "y_0": 0,
            },
            {
                "N_0": 1720000000000000.0,
                "N_T": np.float64(56846624270121.38),
                "k_B": np.float64(7.021105074240901e-20),
                "k_D": np.float64(3.244976806038882e-19),
                "k_T": np.float64(1.1738541743820234e-16),
                "mu_e": np.float64(29.167501568118773),
                "mu_h": np.float64(57.83322022771485),
                "p_0": np.float64(15958840562808.432),
                "y_0": 0,
            },
            {
                "N_0": 4750000000000000.0,
                "N_T": np.float64(56846624270121.38),
                "k_B": np.float64(7.021105074240901e-20),
                "k_D": np.float64(3.244976806038882e-19),
                "k_T": np.float64(1.1738541743820234e-16),
                "mu_e": np.float64(29.167501568118773),
                "mu_h": np.float64(57.83322022771485),
                "p_0": np.float64(15958840562808.432),
                "y_0": 0,
            },
        ]
        assert trmc_btd_fit.fit() == expected

    def test_trmc_fit2(self, trmc_btd_fit2):

        expected = [
            {
                "N_0": 55000000000000.0,
                "N_T": np.float64(43678812277957.82),
                "k_B": np.float64(6.887110004398958e-20),
                "k_D": np.float64(4.031066231308146e-19),
                "k_T": np.float64(1.3828518095278036e-13),
                "mu_e": np.float64(25.769913731468623),
                "mu_h": np.float64(57.322719765917114),
                "p_0": np.float64(13041527621252.596),
                "y_0": 0,
            },
            {
                "N_0": 164000000000000.0,
                "N_T": np.float64(43678812277957.82),
                "k_B": np.float64(6.887110004398958e-20),
                "k_D": np.float64(4.031066231308146e-19),
                "k_T": np.float64(1.3828518095278036e-13),
                "mu_e": np.float64(25.769913731468623),
                "mu_h": np.float64(57.322719765917114),
                "p_0": np.float64(13041527621252.596),
                "y_0": 0,
            },
            {
                "N_0": 511000000000000.0,
                "N_T": np.float64(43678812277957.82),
                "k_B": np.float64(6.887110004398958e-20),
                "k_D": np.float64(4.031066231308146e-19),
                "k_T": np.float64(1.3828518095278036e-13),
                "mu_e": np.float64(25.769913731468623),
                "mu_h": np.float64(57.322719765917114),
                "p_0": np.float64(13041527621252.596),
                "y_0": 0,
            },
            {
                "N_0": 1720000000000000.0,
                "N_T": np.float64(43678812277957.82),
                "k_B": np.float64(6.887110004398958e-20),
                "k_D": np.float64(4.031066231308146e-19),
                "k_T": np.float64(1.3828518095278036e-13),
                "mu_e": np.float64(25.769913731468623),
                "mu_h": np.float64(57.322719765917114),
                "p_0": np.float64(13041527621252.596),
                "y_0": 0,
            },
            {
                "N_0": 4750000000000000.0,
                "N_T": np.float64(43678812277957.82),
                "k_B": np.float64(6.887110004398958e-20),
                "k_D": np.float64(4.031066231308146e-19),
                "k_T": np.float64(1.3828518095278036e-13),
                "mu_e": np.float64(25.769913731468623),
                "mu_h": np.float64(57.322719765917114),
                "p_0": np.float64(13041527621252.596),
                "y_0": 0,
            },
        ]
        assert trmc_btd_fit2.fit() == expected

    # ------------------------------------------------- calculate_fits -------------------------------------------------

    def test_gaussian_calculate_fits(self, gaussian_fit):

        expected = np.array([12340.98040867, 18690.68861775])
        assert np.allclose(gaussian_fit.calculate_fits(gaussian_fit.fit())[0][:2], expected)

    def test_bt_trpl_calculate_fits(self, trpl_bt_fit):

        expected = np.array([0.89870433, 0.86764074])
        assert np.allclose(trpl_bt_fit.calculate_fits(trpl_bt_fit.fit())[0][:2], expected)

    def test_btd_trpl_calculate_fits(self, trpl_btd_fit):

        expected = np.array([1.01284958, 0.88153934])
        assert np.allclose(trpl_btd_fit.calculate_fits(trpl_btd_fit.fit())[0][:2], expected)

    def test_trmc_calculate_fits(self, trmc_btd_fit):

        expected = np.array([87.00027161, 87.0002521])
        assert np.allclose(trmc_btd_fit.calculate_fits(trmc_btd_fit.fit())[0][:2], expected)

    def test_trmc_calculate_fits2(self, trmc_btd_fit2):

        expected = np.array([83.0926335, 61.80049834])
        assert np.allclose(trmc_btd_fit2.calculate_fits(trmc_btd_fit2.fit())[0][:2], expected)

    # -------------------------------------------------- calculate_rss -------------------------------------------------

    def test_gaussian_calculate_rss(self, gaussian_fit):

        fits = gaussian_fit.calculate_fits(gaussian_fit.fit())
        assert gaussian_fit.calculate_rss(fits) == np.float64(1.0)

    def test_bt_trpl_calculate_rss(self, trpl_bt_fit):

        fits = trpl_bt_fit.calculate_fits(trpl_bt_fit.fit())
        assert trpl_bt_fit.calculate_rss(fits) == np.float64(0.9925465869198932)

    def test_btd_trpl_calculate_rss(self, trpl_btd_fit):

        fits = trpl_btd_fit.calculate_fits(trpl_btd_fit.fit())
        assert trpl_btd_fit.calculate_rss(fits) == np.float64(0.9959540173141508)

    def test_trmc_calculate_rss(self, trmc_btd_fit):

        fits = trmc_btd_fit.calculate_fits(trmc_btd_fit.fit())
        assert trmc_btd_fit.calculate_rss(fits) == np.float64(0.9999895781432716)

    def test_trmc_calculate_rss2(self, trmc_btd_fit2):

        fits = trmc_btd_fit2.calculate_fits(trmc_btd_fit2.fit())
        assert trmc_btd_fit2.calculate_rss(fits) == np.float64(0.9847535047208212)


class TestRunGridFit:

    def test_run_grid_fit_gaussian(self, gaussian_data):

        xs_data, ys_data, gaussian, p0, detached_parameters, fixed_parameters = gaussian_data
        p0s = dict(a=[1e7, 1e8], x0=[0, 10])
        output = run_grid_fit(
            p0s,
            fixed_parameters,
            None,
            None,
            xs_data=xs_data,
            ys_data=ys_data,
            function=gaussian,
            detached_parameters=detached_parameters,
        )
        expected = (
            {
                "a": np.array([1.00000000e08, 5.01064110e07, 1.00000000e08, 1.00134673e08]),
                "x0": np.array([1.0, 0.99497651, 1.0, 8.75007255]),
                "c": np.array([0.5, 0.5, 0.5, 0.5]),
            },
            {
                "a": np.array([1.0e07, 1.0e07, 1.0e08, 1.0e08]),
                "x0": np.array([0, 10, 0, 10]),
                "c": np.array([0.5, 0.5, 0.5, 0.5]),
            },
            np.array([1.0, 0.30705282, 1.0, -0.26443591]),
        )

        self.assert_grid_output(output, expected)

    def test_run_grid_fit_trpl_bt(self, trpl_bt_data):
        xs_data, ys_data, function, p0, detached_parameters, fixed_parameters = trpl_bt_data
        p0s = dict(k_B=[1e-20, 1e-19, 1e-18], k_T=[0.01, 0.001], I=[1.0], y_0=[0])
        fixed_parameters = [utils.merge_dicts(f, {"k_A": 0.0, "y_0": 0.0}) for f in fixed_parameters]
        output = run_grid_fit(
            p0s,
            fixed_parameters,
            None,
            xs_data=xs_data,
            ys_data=ys_data,
            function=function,
            detached_parameters=detached_parameters,
        )

        expected = (
            {
                "k_B": np.array(
                    [1.25796064e-19, 1.25797294e-19, 1.25795985e-19, 1.25797351e-19, 1.25798256e-19, 1.25797550e-19]
                ),
                "k_T": np.array([0.00381642, 0.0038164, 0.00381642, 0.0038164, 0.00381638, 0.00381639]),
                "I": np.array([0.86186475, 0.8618624, 0.8618649, 0.86186234, 0.86186092, 0.86186196]),
                "N_0": np.array([1.0e15, 1.0e15, 1.0e15, 1.0e15, 1.0e15, 1.0e15]),
                "k_A": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "y_0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            {
                "k_B": np.array([1.0e-20, 1.0e-20, 1.0e-19, 1.0e-19, 1.0e-18, 1.0e-18]),
                "k_T": np.array([0.01, 0.001, 0.01, 0.001, 0.01, 0.001]),
                "I": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                "N_0": np.array([1.0e15, 1.0e15, 1.0e15, 1.0e15, 1.0e15, 1.0e15]),
                "k_A": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "y_0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            np.array([0.98353857, 0.98353857, 0.98353857, 0.98353857, 0.98353857, 0.98353857]),
        )

        self.assert_grid_output(output, expected)

    @staticmethod
    def assert_grid_output(output, expected):

        for ex, out in zip(expected, output):
            if isinstance(ex, dict) and isinstance(out, dict):
                for key, value in ex.items():
                    assert key in out
                    assert np.allclose(value, out[key], atol=1e-8)  # You can adjust the tolerance if needed
