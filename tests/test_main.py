"""Test module for the functions in the `main.py` module.

This module contains unit tests for the functions implemented in the `main.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import copy
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from streamlit.testing.v1 import AppTest

from app.resources import APP_MODES, BTD_TRMC_DATA, BTD_TRPL_DATA, BT_TRMC_DATA, BT_TRPL_DATA
from app.utility.data import are_close


class TestApp:

    main_path = "app/main.py"

    def teardown_method(self) -> None:
        """Teardown method that runs after each test."""

        # Make sure that no exception happened
        assert len(self.at.exception) == 0

    def test_default(self) -> None:

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()
        assert len(self.at.error) == 0
        assert self.at.expander[-1].label == "License & Disclaimer"

    def get_widget_by_key(
        self,
        widget: str,
        key: str,
        verbose: bool = False,
    ):
        """Get a widget given its key
        :param widget: widget type
        :param key: key
        :param verbose: if True, print the index of the widget"""

        keys = [wid.key for wid in getattr(self.at, widget)]
        if verbose:
            print(keys)  # pragma: no cover
        index = keys.index(key)
        if verbose:
            print(index)  # pragma: no cover
        return getattr(self.at, widget)[index]

    def set_period(self, value: str):
        """Set the repetition period"""

        self.get_widget_by_key("text_input", "period_input_").set_value(value).run()

    def set_app_mode(self, value: str):
        """Set the app mode"""

        self.get_widget_by_key("selectbox", "fit_mode_").set_value(value).run()

    def set_quantity(self, value: str):
        """Set the fit quantity"""

        self.get_widget_by_key("radio", "quantity_input_").set_value(value).run()

    def set_model(self, value: str):
        """Set the model"""

        self.get_widget_by_key("selectbox", "model_name_").set_value(value).run()

    def run(self):
        """Press the run button"""

        self.get_widget_by_key("button", "run_button").click().run()

    def set_data_delimiter(self, value: str):
        """Set the data delimiter"""

        self.get_widget_by_key("radio", "data_delimiter").set_value(value).run()

    def set_data_format(self, value: str):
        """Set the data format"""

        self.get_widget_by_key("radio", "data_format_").set_value(value).run()

    def set_preprocess(self, value: bool):
        """Set the pre-processing option"""

        self.get_widget_by_key("checkbox", "preprocess_").set_value(value).run()

    def set_fixed_parameter(self, key: str, value: str, model: str = "BTA"):
        """Set the fixed parameter value"""

        self.get_widget_by_key("text_input", model + key + "fixed").set_value(value).run()

    def set_guess_parameter(self, key: str, value: str, model: str = "BTA"):
        """Set the guess parameter value"""

        self.get_widget_by_key("text_input", model + key + "guess").set_value(value).run()

    def set_guesses_parameter(self, key: str, value: str, model: str = "BTA"):
        """Set the guess value range"""

        self.get_widget_by_key("text_input", model + key + "guesses").set_value(value).run()

    def set_matching_input(self, value):
        """Set the matching input"""

        self.get_widget_by_key("text_input", "matching_input").set_value(value).run()

    def fill_N0s(self, N0s: list[float | str]) -> None:
        """Fill the photoexcited carrier concentrations inputs
        :param N0s: photoexcited carrier concentration values"""

        widgets = [self.get_widget_by_key("text_input", f"fluence_{i}") for i in range(len(N0s))]
        assert len(widgets) == len(N0s)

        for text_input, N0 in zip(widgets, N0s):
            text_input.input(str(N0))
        self.at.run()

    @staticmethod
    def create_mock_file(mock_file_uploader: MagicMock, data: np.ndarray) -> None:
        """Create a temporary CSV file with uneven columns and mock file upload.
        :param mock_file_uploader: MagicMock
        :param data: data to be uploaded"""

        # Save the data
        temp_path = "_temp.csv"
        np.savetxt(temp_path, data, fmt="%s", delimiter=",")

        # Load the data to the mock file
        with open(temp_path, "rb") as f:
            mock_file_uploader.return_value = BytesIO(f.read())

        # Remove the file
        os.remove(temp_path)

    # -------------------------------------------------- TEST FITTING --------------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def _test_fitting(
        self,
        dataset: tuple[list[np.ndarray], list[np.ndarray], list[float]],
        quantity: str,
        model: str,
        expected_output: dict,
        preprocess: bool,
        mock_file_uploader: MagicMock,
    ) -> None:

        # Load the data
        data = np.transpose([dataset[0][0]] + dataset[1])
        self.create_mock_file(mock_file_uploader, data)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Select the correct quantity
        self.set_quantity(quantity)

        # Pre-processing
        if preprocess:
            self.set_preprocess(True)

        # Check the number of fluence inputs
        self.fill_N0s(dataset[2])

        # Select the model
        self.set_model(model)

        # Click on the button and assert the fit results
        self.run()

        assert are_close(self.at.session_state["results"][0]["popts"][0], expected_output["popt"])
        assert are_close(self.at.session_state["results"][0]["contributions"], expected_output["contributions"])

        self.set_period("200")

        assert are_close(self.at.session_state["carrier_accumulation"]["CA"], expected_output["ca"])
        assert len(self.at.error) == 0

    def test_bt_trpl(self) -> None:
        expected = {
            "popt": {
                "I": 1.0,
                "N_0": 1000000000000000.0,
                "k_A": 0.0,
                "k_B": 5.036092379781616e-19,
                "k_T": 0.01013367471651538,
                "y_0": 0.0,
            },
            "contributions": {
                "A": [0.0, 0.0, 0.0],
                "B": [3.21140477, 24.33019951, 74.24002711],
                "T": [96.78859523, 75.66980049, 25.75997289],
            },
            "ca": [0.20075745903090358, 0.9503120802981269, 0.6080441775140444],
        }
        self._test_fitting(BT_TRPL_DATA, "TRPL", "BTA", expected, False)

    def test_btd_trpl(self) -> None:
        expected = {
            "popt": {
                "I": 1.0,
                "N_0": 51000000000000.0,
                "N_T": 59388399740318.6,
                "k_B": 5.07768333654429e-19,
                "k_D": 7.695781718031608e-19,
                "k_T": 1.1840584706593167e-16,
                "p_0": 71425163414191.94,
                "y_0": 0.0,
            },
            "contributions": {
                "B": [1.97464011, 25.01522655, 62.98962314, 86.000232, 92.82901167],
                "D": [0.63268051, 8.28295762, 8.90099192, 4.14960937, 1.75052228],
                "T": [97.39267938, 66.70181583, 28.10938494, 9.85015863, 5.42046605],
            },
            "ca": [47.390257057209226, 17.86713876214925, 14.169801097560752, 25.484481563736466, 20.172770374561566],
        }
        self._test_fitting(BTD_TRPL_DATA, "TRPL", "BTD", expected, False)

    def test_bt_trmc(self) -> None:
        expected = {
            "popt": {
                "N_0": 1000000000000000.0,
                "k_A": 0.0,
                "k_B": 4.9191697581358015e-19,
                "k_T": 0.010046921491709,
                "mu": 10.021005406705859,
                "y_0": 0.0,
            },
            "contributions": {
                "A": [0.0, 0.0, 0.0],
                "B": [2.37117424, 18.60734623, 63.77640104],
                "T": [97.62882576, 81.39265377, 36.22359896],
            },
            "ca": [0.17048546661642128, 0.8133504291988225, 0.5281220522039942],
        }
        self._test_fitting(BT_TRMC_DATA, "TRMC", "BTA", expected, False)

    def test_btd_trmc(self) -> None:
        expected = {
            "popt": {
                "N_0": 51000000000000.0,
                "N_T": 59033305468562.54,
                "k_B": 5.067910058596048e-19,
                "k_D": 8.038666341707501e-19,
                "k_T": 1.1646600592587665e-16,
                "mu_e": 20.316487864059788,
                "mu_h": 29.881798526675844,
                "p_0": 64418783311953.03,
                "y_0": 0.0,
            },
            "contributions": {
                "B": [1.71049246, 22.07876196, 53.65212212, 76.01675686, 85.26077901],
                "D": [56.99175709, 44.95497873, 25.57202925, 11.08623379, 5.1548549],
                "T": [41.29775044, 32.96625931, 20.77584864, 12.89700935, 9.58436609],
            },
            "ca": [17.754055972544826, 6.065924114804922, 9.763315562751995, 17.252574397074692, 16.546710504457472],
        }
        self._test_fitting(BTD_TRMC_DATA, "TRMC", "BTD", expected, False)

    def test_bt_trpl_preprocess(self) -> None:
        expected = {
            "popt": {
                "k_T": 0.01027692599435874,
                "k_B": 5.223353716145634e-19,
                "k_A": 0.0,
                "y_0": 0.0,
                "I": 1.0,
                "N_0": 1000000000000000.0,
            },
            "contributions": {
                "T": np.array([96.71939708, 75.26854715, 25.35779815]),
                "B": np.array([3.28060292, 24.73145285, 74.64220185]),
                "A": np.array([0.0, 0.0, 0.0]),
            },
            "ca": [0.19826769835378788, 0.9274715441877357, 0.5803950947538994],
        }
        data = list(copy.deepcopy(BT_TRPL_DATA))
        x0 = np.linspace(0, 50, 51)
        data[0] = [np.concatenate([x0, x + x0[-1]]) for x in data[0]]
        data[1] = [np.concatenate([np.zeros(len(x0)), x]) for x in data[1]]
        self._test_fitting(BT_TRPL_DATA, "TRPL", "BTA", expected, True)

    # -------------------------------------------------- TEST INVALID --------------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_invalid_carrier_concentrations(self, mock_file_uploader: MagicMock) -> None:

        # Load the data
        data = np.random.randn(10, 3)
        self.create_mock_file(mock_file_uploader, data)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Check the number of fluence inputs
        self.fill_N0s(["f", "3"])

        expected = "Uh-oh! The initial carrier concentrations input is not valid"
        assert self.at.error[0].value == expected

    @patch("streamlit.sidebar.file_uploader")
    def test_invalid_fixed_guess_value(self, mock_file_uploader: MagicMock) -> None:

        # Load the data
        data = np.transpose([BT_TRPL_DATA[0][0]] + BT_TRPL_DATA[1])
        self.create_mock_file(mock_file_uploader, data)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Check the number of fluence inputs
        self.fill_N0s(BT_TRPL_DATA[2])

        # Change the fixed value to an incorrect value
        self.set_fixed_parameter("k_T", "3")
        self.set_fixed_parameter("k_T", "f")
        assert self.at.session_state["models"]["BTA"]["TRPL"].fvalues["k_T"] == 3

        # Change the guess value to an incorrect value
        self.set_fixed_parameter("k_T", "")  # reset the fixed value
        self.set_guess_parameter("k_T", "3")  # set the guess value
        self.set_guess_parameter("k_T", "f")  # guess value with incorrect string
        assert self.at.session_state["models"]["BTA"]["TRPL"].gvalues["k_T"] == 3

        # Change the fixed value range to an incorrect value
        self.set_app_mode(APP_MODES[1])
        self.set_guesses_parameter("k_T", "2,5,6")
        self.set_guesses_parameter("k_T", "2,5,f")
        assert self.at.session_state["models"]["BTA"]["TRPL"].gvalues_range["k_T"] == [2.0, 5.0, 6.0]

    @patch("streamlit.sidebar.file_uploader")
    def test_bad_fitting(self, mock_file_uploader: MagicMock) -> None:

        # Load the data
        data = np.transpose([BT_TRPL_DATA[0][0]] + BT_TRPL_DATA[1])
        self.create_mock_file(mock_file_uploader, data)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Check the number of fluence inputs
        self.fill_N0s(BT_TRPL_DATA[2])

        # Change the fixed value to an incorrect value
        self.set_guess_parameter("k_T", "-1")

        # Click on the button and assert the fit results
        self.run()

        expected = "The data could not be fitted. Try changing the parameter guess or fixed values."
        assert self.at.error[0].value == expected

    @patch("streamlit.sidebar.file_uploader")
    def test_uneven_column_file(self, mock_file_uploader: MagicMock):

        x = np.linspace(0, 10, 50)
        y = np.cos(x[:-10])  # Make y shorter than x
        y_str = [str(_y) for _y in y] + [""] * (len(x) - len(y))
        temp_path = "_temp.csv"
        np.savetxt(temp_path, np.transpose([x, y_str]), fmt="%s", delimiter=",")
        with open("_temp.csv", "rb") as f:
            mock_file_uploader.return_value = BytesIO(f.read())
        os.remove(temp_path)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        expected = "Uh-oh! The data could not be loaded. Error: Mismatch at index 1: x and y columns must have the same length."
        assert self.at.error[0].value == expected
        assert len(self.at.error) == 1

    @patch("streamlit.sidebar.file_uploader")
    def test_uneven_column_file2(self, mock_file_uploader: MagicMock):

        # Create uneven data
        x = np.linspace(0, 10, 50)
        y = np.cos(x[:-10])
        y_str = [str(_y) for _y in y] + [""] * (len(x) - len(y))

        # Load the data
        self.create_mock_file(mock_file_uploader, np.transpose([x, y_str]))

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        expected = "Uh-oh! The data could not be loaded. Error: Mismatch at index 1: x and y columns must have the same length."
        assert self.at.error[0].value == expected
        assert len(self.at.error) == 1

    @patch("streamlit.sidebar.file_uploader")
    def test_column_file(self, mock_file_uploader: MagicMock):

        # Create uneven data
        x = [np.linspace(0, 10, 50)] * 3
        y = [np.cos(x[0])] * 2
        data = np.transpose([x[0], y[0], x[1], y[1], x[2]])

        # Load the data
        self.create_mock_file(mock_file_uploader, data)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Change the data format
        self.set_data_format("X1/Y1/X2/Y2...")

        expected = "Uh-oh! The data could not be loaded. Error: Mismatch: x data and y data must have the same number of columns."
        assert self.at.error[0].value == expected
        assert len(self.at.error) == 1

    @patch("streamlit.sidebar.file_uploader")
    def test_incorrect_delimiter(self, mock_file_uploader: MagicMock):

        # Load the data
        data = np.random.randn(10, 3)
        self.create_mock_file(mock_file_uploader, data)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Change the delimiter
        self.set_data_delimiter(";")

        expected = "Uh-oh! The data could not be loaded. Error: Unknown error, Check that the correct delimiter has been selected."
        assert self.at.error[0].value == expected
        assert len(self.at.error) == 1

    @patch("streamlit.sidebar.file_uploader")
    def test_failed_ca(self, mock_file_uploader: MagicMock) -> None:

        # Save and load the data
        self.create_mock_file(mock_file_uploader, np.transpose([BT_TRPL_DATA[0][0]] + BT_TRPL_DATA[1]))

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Check the number of fluence inputs
        self.fill_N0s(BT_TRPL_DATA[2])

        # Set the period
        self.set_period("0.0001")

        # Click on the button and assert the fit results
        self.run()

        assert self.at.session_state["carrier_accumulation"] == {}
        expected = "Carrier accumulation could not be calculated due to excessive computational requirements."
        assert self.at.warning[0].value == expected

        # Set the app mode
        self.set_app_mode(APP_MODES[1])

        # Click on the run button and assert the fit results
        self.run()

        assert self.at.session_state["carrier_accumulation"] == []

    @patch("streamlit.sidebar.file_uploader")
    def test_failed_matching(self, mock_file_uploader: MagicMock) -> None:

        # Save and load the data
        self.create_mock_file(mock_file_uploader, np.transpose([BT_TRPL_DATA[0][0]] + BT_TRPL_DATA[1]))

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Check the number of fluence inputs
        self.fill_N0s(BT_TRPL_DATA[2])

        # Click on the button and assert the fit results
        self.run()

        self.set_matching_input("f")

        assert self.at.warning[0].value == "Please input correct values."

    @patch("streamlit.sidebar.file_uploader")
    def test_bad_grid_fitting(self, mock_file_uploader: MagicMock) -> None:

        # Load the data
        data = np.transpose([BT_TRPL_DATA[0][0]] + BT_TRPL_DATA[1])
        self.create_mock_file(mock_file_uploader, data)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        self.set_app_mode(APP_MODES[1])

        # Check the number of fluence inputs
        self.fill_N0s(BT_TRPL_DATA[2])

        # Change the fixed value to an incorrect value
        self.set_guesses_parameter("k_T", "-1, -2")

        # Click on the button and assert the fit results
        self.run()

        expected = "The data could not be fitted. Try changing the parameter guess or fixed values."
        assert self.at.error[0].value == expected

    # ----------------------------------------------------- OTHERS -----------------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_settings_changed(self, mock_file_uploader: MagicMock) -> None:

        # Load the data
        data = np.transpose([BT_TRPL_DATA[0][0]] + BT_TRPL_DATA[1])
        self.create_mock_file(mock_file_uploader, data)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Check the number of fluence inputs
        self.fill_N0s(BT_TRPL_DATA[2])

        # Click on the button and assert the fit results
        self.run()

        # Change settings
        self.set_guess_parameter("k_T", "1")

        expected = "You have changed some of the input settings. Press 'Run' to apply these changes."
        assert self.at.warning[0].value == expected

    @patch("streamlit.sidebar.file_uploader")
    def test_stored_ca(self, mock_file_uploader: MagicMock) -> None:

        # Load the data
        data = np.transpose([BT_TRPL_DATA[0][0]] + BT_TRPL_DATA[1])
        self.create_mock_file(mock_file_uploader, data)

        self.at = AppTest(self.main_path, default_timeout=100).run()  # start the app and run it
        self.fill_N0s(BT_TRPL_DATA[2])  # check the number of fluence inputs
        self.run()  # click on the run button
        self.set_period("100")  # set the period

        # Rerun
        self.at.run()
        assert self.at.session_state.carrier_accumulation is not None
        assert len(self.at.error) == 0

    # ------------------------------------------------ TEST GRID FITTING -----------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def _test_grid_fitting(
        self,
        dataset: tuple[list[np.ndarray], list[np.ndarray], list[float]],
        quantity: str,
        model: str,
        expected_output: dict,
        mock_file_uploader: MagicMock,
    ) -> None:

        # Save and load the data
        self.create_mock_file(mock_file_uploader, np.transpose([dataset[0][0]] + dataset[1]))

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Set the app mode
        self.set_app_mode(APP_MODES[1])

        # Select the correct quantity
        self.set_quantity(quantity)

        # Check the number of fluence inputs
        self.fill_N0s(dataset[2])

        # Select the model
        self.set_model(model)

        # Check that the run button is present
        assert len(self.at.sidebar.button) == 1
        assert self.at.sidebar.button[0].label == "Run"

        # Click on the button and assert the fit results
        self.run()

        assert self.at.markdown[1].value == "## Parallel plot"
        self.set_period("200")

        ca = [f["CA"] for f in self.at.session_state["carrier_accumulation"]]
        assert are_close(ca, expected_output)
        assert len(self.at.error) == 0

    def test_bt_trpl_grid(self) -> None:
        expected = [
            [0.20075751831579725, 0.9503122259985342, 0.6080441291965166],
            [0.20075752653634926, 0.9503122402142683, 0.6080441126882885],
            [0.2007576800148292, 0.9503126618979119, 0.608044067096708],
            [0.2007575570692477, 0.9503123227196708, 0.6080440988083968],
        ]
        self._test_grid_fitting(BT_TRPL_DATA, "TRPL", "BTA", expected)

    def test_bt_trmc_grid(self) -> None:
        expected = [
            [0.1704854700170455, 0.8133504278268588, 0.5281220312685408],
            [0.17048547467589104, 0.8133504527513102, 0.5281220530171438],
            [0.17048546359939576, 0.8133504112995238, 0.5281220345702942],
            [0.17048545828642347, 0.8133503975122747, 0.5281220391482655],
            [0.17048548691176446, 0.8133504687864224, 0.5281220151080401],
            [0.17048546618355087, 0.8133504175408257, 0.528122032816325],
            [0.1704854670983691, 0.8133504197771313, 0.5281220324267089],
            [0.17048547165501304, 0.8133504321135743, 0.5281220299483136],
        ]

        self._test_grid_fitting(BT_TRMC_DATA, "TRMC", "BTA", expected)
