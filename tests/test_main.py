import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from streamlit.testing.v1 import AppTest

from app.resources import APP_MODES, BTD_TRMC_DATA, BTD_TRPL_DATA, BT_TRMC_DATA, BT_TRPL_DATA
from app.utility.data import are_close


BT_TRPL_expected = {
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

BTD_TRPL_expected = {
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
BT_TRMC_expected = {
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

BTD_TRMC_expected = {
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

BT_TRPL_GRID_expected = [
    [0.20075751831579725, 0.9503122259985342, 0.6080441291965166],
    [0.20075752653634926, 0.9503122402142683, 0.6080441126882885],
    [0.2007576800148292, 0.9503126618979119, 0.608044067096708],
    [0.2007575570692477, 0.9503123227196708, 0.6080440988083968],
]

BT_TRMC_GRID_expected = [
    [0.1704854700170455, 0.8133504278268588, 0.5281220312685408],
    [0.17048547467589104, 0.8133504527513102, 0.5281220530171438],
    [0.17048546359939576, 0.8133504112995238, 0.5281220345702942],
    [0.17048545828642347, 0.8133503975122747, 0.5281220391482655],
    [0.17048548691176446, 0.8133504687864224, 0.5281220151080401],
    [0.17048546618355087, 0.8133504175408257, 0.528122032816325],
    [0.1704854670983691, 0.8133504197771313, 0.5281220324267089],
    [0.17048547165501304, 0.8133504321135743, 0.5281220299483136],
]


class TestApp:

    @patch("streamlit.sidebar.file_uploader")
    def _test_fitting(
        self,
        dataset: tuple[list[np.ndarray], list[np.ndarray], list[float]],
        quantity: str,
        model: str,
        expected_output: dict,
        mock_file_uploader: MagicMock,
    ) -> None:

        # Save and load the data
        temp_path = "_temp.csv"
        np.savetxt(temp_path, np.transpose([dataset[0][0]] + dataset[1]), delimiter=",")
        with open("_temp.csv", "rb") as f:
            mock_file_uploader.return_value = BytesIO(f.read())
        os.remove(temp_path)

        # Start the app and run it
        at = AppTest(main_path, default_timeout=100)
        at.run()

        # Select the correct quantity
        at.sidebar.radio[2].set_value(quantity)

        # Check the number of fluence inputs
        assert len(at.text_input) == len(dataset[2])
        for text_input, N0 in zip(at.sidebar.text_input, dataset[2]):
            text_input.input(str(N0))
        at.run()

        # Select the model
        at.sidebar.selectbox[1].set_value(model)

        # Check that the run button is present
        assert len(at.sidebar.button) == 1
        assert at.sidebar.button[0].label == "Run"

        # Click on the button and assert the fit results
        at.sidebar.button[0].click()
        at.run()

        assert are_close(at.session_state["results"][0]["popts"][0], expected_output["popt"])
        assert are_close(at.session_state["results"][0]["contributions"], expected_output["contributions"])

        at.sidebar.text_input[-1].set_value("200")
        at.run()

        assert are_close(at.session_state["carrier_accumulation"], expected_output["ca"])

    def test_bt_trpl(self):
        self._test_fitting(BT_TRPL_DATA, "TRPL", "BTA", BT_TRPL_expected)

    def test_btd_trpl(self):
        self._test_fitting(BTD_TRPL_DATA, "TRPL", "BTD", BTD_TRPL_expected)

    def test_bt_trmc(self):
        self._test_fitting(BT_TRMC_DATA, "TRMC", "BTA", BT_TRMC_expected)

    def test_btd_trmc(self):
        self._test_fitting(BTD_TRMC_DATA, "TRMC", "BTD", BTD_TRMC_expected)

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
        temp_path = "_temp.csv"
        np.savetxt(temp_path, np.transpose([dataset[0][0]] + dataset[1]), delimiter=",")
        with open("_temp.csv", "rb") as f:
            mock_file_uploader.return_value = BytesIO(f.read())
        os.remove(temp_path)

        # Start the app and run it
        at = AppTest(main_path, default_timeout=100)
        at.run()

        # Set the app mode
        at.sidebar.selectbox[0].set_value(APP_MODES[1])

        # Select the correct quantity
        at.sidebar.radio[2].set_value(quantity)

        # Check the number of fluence inputs
        assert len(at.text_input) == len(dataset[2])
        for text_input, N0 in zip(at.sidebar.text_input, dataset[2]):
            text_input.input(str(N0))
        at.run()

        # Select the model
        at.sidebar.selectbox[1].set_value(model)

        # Check that the run button is present
        assert len(at.sidebar.button) == 1
        assert at.sidebar.button[0].label == "Run"

        # Click on the button and assert the fit results
        at.sidebar.button[0].click()
        at.run()

        assert at.markdown[1].value == "#### Fitting results"
        at.sidebar.text_input[-1].set_value("200")
        at.run()

        assert are_close(at.session_state["carrier_accumulation"], expected_output)

    def test_bt_trpl_grid(self):
        self._test_grid_fitting(BT_TRPL_DATA, "TRPL", "BTA", BT_TRPL_GRID_expected)

    def test_bt_trmc_grid(self):
        self._test_grid_fitting(BT_TRMC_DATA, "TRMC", "BTA", BT_TRMC_GRID_expected)


if __name__ != "__main__":
    main_path = "../app/main.py"
else:
    main_path = "pears/app/main.py"
