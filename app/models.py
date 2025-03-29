"""Models for simulating and fitting time-resolved photoluminescence and microwave conductivity data using the
bimolecular-trapping and bimolecular-trapping-detrapping models

The classes are:
- Model: defines common methods for all sub-models such as solving the rate equations, fitting data and calculating the
  contributions. It also contains constants for all models such as parameter symbols. The rate_equations method defined
  in subclasses need to have a **kwarg to allow parameters from the fitting optimisation to be used.
    - BTModel: is the general class for the Bimolecular-Trapping model. It defines the rate_equations for this model as
      well as calculate_trpl and calculate_trmc methods.
        - BTModelTRPL: is the specific class for fitting TRPL using the BT model. it defines calculate_fit_quantity and
          calculate_contributions which has a **kwarg to allow parameters from the fitting optimisation to be used.
        - BTModelTRMC: is the specific class for fitting TRMC using the BT model. it defines calculate_fit_quantity and
          calculate_contributions which has a **kwarg to allow parameters from the fitting optimisation to be used.
    - BTDModel: is the general class for the Bimolecular-Trapping-Detrapping model. It defines the rate_equations for
      this model as well as calculate_trpl and calculate_trmc.
        - BTDModelTRPL: is the specific class for fitting TRPL using the BTD model. it defines calculate_fit_quantity
          and calculate_contributions which has a **kwarg to allow parameters from the fitting optimisation to be used.
        - BTDModelTRMC: is the specific class for fitting TRMC using the BTD model. it defines calculate_fit_quantity
          and calculate_contributions which has a **kwarg to allow parameters from the fitting optimisation to be used.
"""

import itertools

import numpy as np
import scipy.integrate as sci
import streamlit

from fitting import Fit
from utility.dict import filter_dicts, list_to_dict, merge_dicts
from utility.numbers import get_power_html, get_power_labels

# Example parameters for the BT and BTD models
BT_KWARGS = dict(k_B=50e-20, k_T=10e-3, k_A=1e-40, I=1.0, y_0=0, mu=10.0)
BTD_KWARGS = dict(k_B=50e-20, k_T=12000e-20, N_T=60e12, p_0=65e12, k_D=80e-20, I=1.0, y_0=0.0, mu_e=20.0, mu_h=30.0)

# Example photoexcited carrier concentrations
BT_N0s = [1e15, 1e16, 1e17]
BTD_N0s = [0.51e14, 1.61e14, 4.75e14, 16.1e14, 43.8e14]


class Model(object):
    """Base class for charge carrier recombination models.

    This class serves as the parent class for various charge carrier recombination models,
    encapsulating key parameters and methods used to define, manage, and manipulate
    model-specific data."""

    CBT_LABELS = {"A": "Auger (%)", "B": "Bimolecular (%)", "T": "Trapping (%)", "D": "Detrapping (%)"}
    CONC_LABELS_HTML = {"n_e": "n<sub>e</sub>", "n_t": "n<sub>t</sub>", "n_h": "n<sub>h</sub>", "n": "n"}
    CONC_COLORS = {"n_e": "red", "n_t": "green", "n_h": "blue", "n": "black"}

    # Quantities display
    PARAM_FULLNAME = {
        "k_B": "Bimolecular recombination rate constant",
        "k_T": "Trapping rate constant",
        "k_D": "Detrapping rate constant",
        "k_A": "Auger recombination rate constant",
        "N_T": "Trap state concentration",
        "p_0": "Doping concentration",
        "y_0": "Intensity offset",
        "I": "Intensity factor",
        "mu": "Carrier mobility",
        "mu_e": "Electron mobility",
        "mu_h": "Hole mobility",
    }

    # Symbols display
    PARAM_SYMBOLS = {
        "k_B": "k_B",
        "k_T": "k_T",
        "k_D": "k_D",
        "k_A": "k_A",
        "N_T": "N_T",
        "p_0": "p_0",
        "y_0": "y_0",
        "I": "I_0",
        "mu": "mu",
        "mu_e": "mu_e",
        "mu_h": "mu_h",
    }
    PARAM_SYMBOLS_HTML = {
        "N_T": "N<sub>T</sub>",
        "p_0": "p<sub>0</sub>",
        "k_B": "k<sub>B</sub>",
        "k_T": "k<sub>T</sub>",
        "k_D": "k<sub>D</sub>",
        "k_A": "k<sub>A</sub>",
        "y_0": "y<sub>0</sub>",
        "I": "I<sub>0</sub>",
        "mu": "&mu;",
        "mu_e": "&mu;<sub>e</sub>",
        "mu_h": "&mu;<sub>h</sub>",
    }

    def __init__(
        self,
        param_ids: list[str],
        units: dict[str, str],
        units_html: dict[str, str],
        factors: dict[str, float],
        fvalues: dict[str, float | None],
        gvalues: dict[str, float],
        gvalues_range: dict[str, list[float]],
        n_keys: list[str],
        n_init: callable,
        conc_ca_ids: list[str],
        param_filters: list[str],
    ):
        """Initializes the charge carrier recombination model with specified parameters.
        :param param_ids: model parameter ids required to calculate the fit quantity.
        :param units: dictionary mapping parameter ids to their units.
        :param units_html: dictionary mapping parameter ids to their HTML-formatted units.
        :param factors: dictionary mapping parameter ids to their factor for display purposes.
        :param fvalues: dictionary mapping parameter ids to their fixed value. Use `None` to indicate the parameter is
                        not fixed.
        :param gvalues: dictionary mapping parameter ids to their guess values.
        :param gvalues_range: dictionary mapping parameter ids to their list of guess values.
        :param n_keys: model carrier concentration keys.
        :param n_init: function used to calculate the initial carrier concentration.
        :param conc_ca_ids: list of carrier concentration keys used to determine stabilisation.
        :param param_filters: parameter relations used to filter the guess parameters combinations."""

        # Store the input arguments
        self.param_ids = param_ids
        self.n_keys = n_keys
        self.n_init = n_init
        self.conc_ca_ids = conc_ca_ids
        self.param_filters = param_filters

        # Add the y_0 and I parameter units, factors, and default fixed values, guess values and guess value ranges
        # and add them to the detached parameters
        self.units = merge_dicts(units, {"y_0": "", "I": ""})
        self.units_html = merge_dicts(units_html, {"y_0": "", "I": ""})
        self.factors = merge_dicts(factors, {"y_0": 1.0, "I": 1.0})
        self.fvalues = merge_dicts(fvalues, {"y_0": 0.0, "I": 1.0})
        self.gvalues = merge_dicts(gvalues, {"y_0": 0.0, "I": 1.0})
        self.gvalues_range = merge_dicts(gvalues_range, {"y_0": [0.0], "I": [1.0]})
        self.detached_parameters = [k for k in ["y_0", "I"] if k in self.param_ids]

        # Only keep the data for the param ids
        for d in ("units", "units_html", "factors", "fvalues", "gvalues", "gvalues_range"):
            setattr(self, d, {key: value for key, value in getattr(self, d).items() if key in self.param_ids})

        self.factors_html = {key: get_power_html(self.factors[key], None) for key in self.factors}

    def __eq__(self, other: any) -> bool:
        """Used to check if this object is the same as another object"""

        condition = (
            self.fvalues == other.fvalues  # same fixed values
            and self.gvalues == other.gvalues  # same guess values
            and self.gvalues_range == other.gvalues_range  # same guess value ranges
            and self.__class__ == other.__class__  # same class name
        )

        return condition

    def get_parameter_label(self, key: str) -> str:
        """Get the parameter label
        :param key: parameter id"""

        if self.units[key] == "":
            return self.PARAM_SYMBOLS[key]
        else:
            return self.PARAM_SYMBOLS[key] + " (" + self.units[key] + ")"

    @property
    def fixed_values(self) -> dict[str, float]:
        """return the dict of fixed values"""

        return {key: value for key, value in self.fvalues.items() if value is not None}

    # -------------------------------------------------- CORE METHODS --------------------------------------------------

    def _rate_equations(self, *args, **kwargs) -> dict:
        """Rate equation method"""

        return {}

    def _calculate_concentrations(
        self,
        t: np.ndarray,
        N_0: float,
        p: int = 1,
        threshold: float = 0.001,
        **kwargs,
    ) -> dict[str, list]:
        """Calculate the carrier concentrations
        :param t: time (ns)
        :param N_0: initial carrier concentration (cm-3)
        :param p: number of pulses
        :param threshold: threshold
        :param kwargs: keyword arguments passed to the rate equations"""

        var = {key: np.zeros(len(t)) for key in self.n_keys}
        variables = {key: [] for key in self.n_keys}

        def rate_equation(x, _t):
            """Rate equation wrapper"""

            dndt = self._rate_equations(*x, **kwargs)
            return [dndt[key] for key in self.n_keys]

        for i in range(p):
            init = self.n_init(N_0)
            var = np.transpose(sci.odeint(rate_equation, [init[key] + var[key][-1] for key in self.n_keys], t))
            var = dict(zip(self.n_keys, var))

            for key in var:
                variables[key].append(var[key])

            if threshold and i > 1:
                ca = np.array([np.max(np.abs(variables[key][-2] - var[key]) / N_0) for key in self.conc_ca_ids])
                if all(ca < threshold / 100):
                    break
                elif i == p - 1:
                    raise AssertionError("Attention: threshold condition never reached")

        return variables

    def calculate_fit_quantity(self, *args, **kwargs) -> np.ndarray:
        """Calculate the TRPL"""

        return np.array([0])

    def calculate_contributions(self, *args, **kwargs) -> dict:
        """Calculate the contributions"""

        return dict()

    def get_carrier_accumulation(
        self,
        params: list[dict[str, float]],
        period: float,
    ) -> list[float]:
        """Calculate the carrier accumulation effect on the TRPL.
        :param params: list of arguments passed to calculate_fit_quantity.
        :param period: excitation repetition period in ns."""

        nca = []

        # For each decay fitted
        for param in params:

            # Create a new log timescale
            x = np.insert(np.logspace(-4, np.log10(period), 10001), 0, 0)

            # Calculate the fit quantity after 1 pulse and until stabilisation
            pulse1 = self.calculate_fit_quantity(x, **param)
            pulse2 = self.calculate_fit_quantity(x, p=10000, **param)

            # Calculate the normalised carrier accumulation in %
            nca.append(np.max(np.abs(pulse1 / pulse1[0] - pulse2 / pulse2[0])) * 100)

        return nca

    def get_carrier_concentrations(
        self,
        xs_data: list[np.ndarray],
        params: list[dict[str, float]],
        period: float,
    ) -> tuple[list[np.ndarray], str, list[dict[str, np.ndarray]]]:
        """Calculate the carrier concentrations from the optimised parameters.
        :param xs_data: x-axis data.
        :param params: list of arguments passed to calculate_concentrations.
        :param period: excitation repetition period in ns."""

        # If no period is provided, use the x-axis data provided and set the number of pulses to 1
        if not period:
            xaxis_data = xs_data
            nb_pulses = 1
            xlabel = "Time (ns)"

        # If a period is provided, generate new x-axis data based on this period
        else:
            xs_data = [np.linspace(0, float(period), 1001)] * len(xs_data)
            nb_pulses = 100
            xaxis_data = [np.linspace(0, nb_pulses, len(x_data) * nb_pulses) for x_data in xs_data]
            xlabel = "Pulse"

        concentrations = []

        # For each x-axis data and parameter values, calculate the concentration after a certain number of pulses
        for x_data, param in zip(xs_data, params):
            concentration = self._calculate_concentrations(x_data, p=nb_pulses, **param)
            concentrations.append({key: np.concatenate(concentration[key]) for key in concentration})

        return xaxis_data, xlabel, concentrations

    # ----------------------------------------------------- FITTING ----------------------------------------------------

    def fit(
        self,
        xs_data: list[np.ndarray],
        ys_data: list[np.ndarray],
        N0s: list[float],
        p0: None | dict[str, float] = None,
    ) -> dict:
        """Fit the data using the model
        :param xs_data: list-like of x data
        :param ys_data: list-like of y data
        :param N0s: list of initial carrier concentrations (cm-3)
        :param p0: guess values. If None, use the gvalues dict"""

        # Add the initial carrier concentration to the fixed parameters and get the guess values
        fixed_parameters = [merge_dicts(self.fixed_values, dict(N_0=n0)) for n0 in N0s]
        if p0 is None:
            p0 = self.gvalues

        # Fitting
        fit = Fit(xs_data, ys_data, self.calculate_fit_quantity, p0, self.detached_parameters, fixed_parameters)

        # Popts, fitted data, R2
        popts = fit.fit()
        fit_ydata = fit.calculate_fits(popts)
        cod = fit.calculate_rss(fit_ydata)

        popts_labels = []
        for key in self.param_ids:

            # Generate the label for that parameter
            if key in self.detached_parameters:
                data = ["%.5f" % (popt[key] / self.factors[key]) for popt in popts]
                value_string = ", ".join(data)
            else:
                value_string = "%.5f" % (popts[0][key] / self.factors[key])
            label = (
                f"{self.PARAM_FULLNAME[key]} ({self.PARAM_SYMBOLS_HTML[key]}): {value_string} "
                f"{self.factors_html[key]} {self.units_html[key]}"
            )

            # Add (fixed) if the parameter was fixed
            if key in fixed_parameters[0]:
                label += " (fixed)"

            popts_labels.append(label)

        popts_labels.append(f"Coefficient of determination R<sup>2</sup>: {cod:.5f}")

        # Contributions
        contributions = []
        for x_data, popt in zip(xs_data, popts):
            concentration = {key: value[0] for key, value in self._calculate_concentrations(x_data, **popt).items()}
            kwargs = merge_dicts(popt, concentration)
            contributions.append(self.calculate_contributions(x_data, **kwargs))
        contributions = list_to_dict(contributions)
        contributions = {key: np.array(value) for key, value in contributions.items()}

        # All values together (guess, optimised, R2 and contributions)
        all_values = dict()  # all values
        hidden_keys = []  # keys not displayed in the parallel plot (fixed values and guess values)
        for key in self.param_ids:
            if key in self.detached_parameters:
                continue
            label = self.PARAM_SYMBOLS_HTML[key] + " (" + self.factors_html[key] + " " + self.units_html[key] + ")"
            if key in fixed_parameters[0]:
                all_values[label + " (fixed)"] = fixed_parameters[0][key] / self.factors[key]
                hidden_keys.append(label + " (fixed)")
            else:
                all_values[label + " (guess)"] = p0[key] / self.factors[key]
                all_values[label + " (opt.)"] = popts[0][key] / self.factors[key]
                hidden_keys.append(label + " (guess)")
        all_values["R<sub>2</sub>"] = cod
        for key in contributions:
            all_values["max. " + self.CBT_LABELS[key]] = np.max(contributions[key])

        return {
            "popts": popts,
            "cod": cod,
            "contributions": contributions,
            "fit_ydata": fit_ydata,
            "popt_labels": popts_labels,
            "N0s": N0s,  # keep a copy of the N0s used for the fits
            "N0s_labels": get_power_labels(N0s),
            "p0": p0,
            "all_values": all_values,
            "hidden_keys": hidden_keys,
            "xs_data": xs_data,
            "ys_data": ys_data,
        }

    def grid_fitting(
        self,
        progressbar: streamlit.progress,
        N0s: list[float],
        **kwargs,
    ) -> list:
        """Run a grid fitting analysis
        :param N0s: initial carrier concentration
        :param progressbar: progressbar
        :param kwargs: keyword arguments passed to the fit method"""

        # Filter out the fixed parameters
        p0s = {key: self.gvalues_range[key] for key in self.gvalues_range if key not in self.fixed_values}

        # Generate all the combination of guess values and filter them
        pkeys, pvalues = zip(*p0s.items())
        all_p0s = [dict(zip(pkeys, v)) for v in itertools.product(*pvalues)]
        all_p0s = filter_dicts(all_p0s, self.param_filters, self.fixed_values)

        # Run the fits
        fits = []
        for i, p0 in enumerate(all_p0s):

            # Update the progressbar if provided
            if progressbar is not None:
                progressbar.progress(i / float(len(all_p0s) - 1))

            # Fit the data
            try:
                fits.append(self.fit(p0=p0, N0s=N0s, **kwargs))
            except ValueError:
                pass

        return fits

    # -------------------------------------------------- CONTRIBUTIONS -------------------------------------------------

    @staticmethod
    def get_contribution_recommendation(
        process: str,
        val: str = "",
    ) -> str:
        """Get the recommendation string for the given contribution
        :param str process: name of the process
        :param str val: 'higher' or 'lower'"""

        string = f"This fit predicts low {process}. The values associated with this process may be inaccurate."
        if val:
            string += (
                f"\nIt is recommended to measure your sample under {val} excitation fluence for this process to become "
                "significant"
            )
        return string

    def get_contribution_recommendations(self, *args, **kwargs) -> list[str]:
        """Placeholder: get recommendations for the contributions"""

        return [""]

    # ----------------------------------------------------- OTHERS -----------------------------------------------------

    def _generate_decays(
        self,
        period: float,
        N0s: list[float],
        parameters: dict[str, float],
        noise: float = 0.0,
        normalise: bool = False,
        **kwargs,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        """Generate example TRMC decays
        :param period: period in ns
        :param N0s: photoexcited carrier concentrations
        :param parameters: parameters used to solve the differential equations
        :param noise: optional noise argument
        :param kwargs: keyword arguments passed to the calculate_fit_quantity method
        :param normalise: if True, normalise the data after adding the noise"""

        t = np.linspace(0, period, 1001)
        xs_data = [t] * len(N0s)
        ys_data = [self.calculate_fit_quantity(t, N_0=n, **merge_dicts(kwargs, parameters)) for n in N0s]
        if noise:
            np.random.seed(1537)
            for y in ys_data:
                noise_array = np.random.normal(loc=0.0, scale=noise * np.max(y), size=len(y))
                y += noise_array
                if normalise:
                    y /= np.max(y)

        return xs_data, ys_data, N0s


# ------------------------------------------------------ BT MODEL ------------------------------------------------------


class BTModel(Model):
    """Class for the Bimolecular-Trapping model"""

    def __init__(self, param_ids: list):
        """Object constructor"""

        units = {
            "k_B": "cm3/ns",
            "k_T": "ns-1",
            "k_A": "cm6/ns-1",
            "mu": "cm^2/(V s)",
        }
        units_html = {
            "k_B": "cm<sup>3</sup>/ns",
            "k_T": "ns<sup>-1</sup>",
            "k_A": "cm<sup>6</sup>/ns",
            "mu": "cm<sup>2</sup>/(V s)",
        }
        factors = {
            "k_B": 1e-20,
            "k_T": 1e-3,
            "k_A": 1e-40,
            "mu": 1,
        }
        fvalues = {
            "k_T": None,
            "k_B": None,
            "k_A": 0.0,
            "mu": None,
        }
        gvalues = {
            "k_T": 0.001,
            "k_B": 1e-20,
            "k_A": 1e-40,
            "mu": 10,
        }
        gvalues_range = {
            "k_B": [1e-20, 1e-18],
            "k_T": [1e-4, 1e-2],
            "k_A": [1e-32, 1e-30],
            "mu": [1, 10],
        }
        n_keys = ["n"]
        n_init = lambda N_0: {"n": N_0}
        conc_ca_ids = ["n"]

        Model.__init__(
            self,
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
            [],
        )

    @staticmethod
    def _rate_equations(
        n: float,
        k_T: float,
        k_B: float,
        k_A: float,
        **kwargs,
    ) -> dict[str, float]:
        """Rate equation of the BT model
        :param n: carrier concentration (cm-3)
        :param k_T: trapping rate (ns-1)
        :param k_B: bimolecular recombination rate (cm3/ns)
        :param k_A: Auger recombination rate (cm6/ns)
        :param kwargs: additional keyword arguments that are ignored"""

        return {"n": -k_T * n - k_B * n**2 - k_A * n**3}

    def calculate_trpl(
        self,
        t: np.ndarray,
        N_0: float,
        I: float,
        y_0: float,
        **kwargs,
    ) -> np.ndarray:
        """Calculate the normalised TRPL intensity
        :param t: time (ns)
        :param N_0: initial carrier concentration
        :param y_0: background intensity
        :param I: amplitude factor
        :param kwargs: keyword arguments passed to the calculate_concentrations function"""

        n = self._calculate_concentrations(t, N_0, **kwargs)["n"][-1]
        I_TRPL = n**2 / N_0
        return I * I_TRPL / I_TRPL[0] + y_0

    def calculate_trmc(
        self,
        t: np.ndarray,
        N_0: float,
        mu: float,
        y_0: float,
        **kwargs,
    ) -> np.ndarray:
        """Calculate the TRMC intensity
        :param t: time (ns)
        :param N_0: initial carrier concentration
        :param mu: carrier mobility (cm2/vs)
        :param y_0: background intensity
        :param kwargs: keyword arguments passed to the calculate_concentrations function"""

        n = self._calculate_concentrations(t, N_0, **kwargs)["n"][-1]
        return (2 * mu * n) / N_0 + y_0

    def get_contribution_recommendations(
        self,
        contributions: dict[str, np.ndarray],
        threshold: float = 10.0,
    ) -> list[str]:
        """Get recommendations for the contributions
        :param contributions: dictionary associating contribution key to their values
        :param threshold: threshold below which a warning is given"""

        recommendations = []
        for key in contributions:
            if np.max(contributions[key]) < threshold:
                if key == "T" and self.fvalues.get("k_T", 1) != 0:
                    recommendations.append(self.get_contribution_recommendation("trapping", "lower"))
                elif key == "B" and self.fvalues.get("k_B", 1) != 0:
                    recommendations.append(self.get_contribution_recommendation("bimolecular", "higher"))
                elif key == "A" and self.fvalues.get("k_A", 1) != 0:
                    recommendations.append(self.get_contribution_recommendation("Auger", "higher"))
        return recommendations


class BTModelTRPL(BTModel):

    def __init__(self):
        BTModel.__init__(self, ["k_T", "k_B", "k_A", "y_0", "I"])

    def calculate_fit_quantity(self, *args, **kwargs):

        return self.calculate_trpl(*args, **kwargs)

    @staticmethod
    def calculate_contributions(
        t: np.ndarray,
        k_T: float,
        k_B: float,
        k_A: float,
        n: np.ndarray,
        **kwargs,
    ) -> dict[str, float]:
        """Calculate the total contributions to the TRPL
        :param k_T: trapping rate constant (ns-1)
        :param k_B: bimolecular rate constant (cm3/ns)
        :param k_A: Auger rate constant (cm6/ns)
        :param n: carrier concentration (cm-3)
        :param t: time (ns)"""

        T = sci.trapezoid(k_T * n**2, t)
        B = sci.trapezoid(k_B * n**3, t)
        A = sci.trapezoid(k_A * n**4, t)
        S = T + B + A
        return {"T": T / S * 100, "B": B / S * 100, "A": A / S * 100}

    def generate_decays(self, *args, **kwargs):
        """Generate decays.
        :param args: arguments passed to _generate_decays.
        :param kwargs: keyword arguments passed to _generate_decays."""

        return self._generate_decays(250.0, BT_N0s, BT_KWARGS, normalise=True, *args, **kwargs)


class BTModelTRMC(BTModel):

    def __init__(self):
        BTModel.__init__(self, ["k_T", "k_B", "k_A", "mu", "y_0"])

    def calculate_fit_quantity(self, *args, **kwargs):

        return self.calculate_trmc(*args, **kwargs)

    @staticmethod
    def calculate_contributions(
        t: np.ndarray,
        k_T: float,
        k_B: float,
        k_A: float,
        mu: float,
        n: np.ndarray,
        **kwargs,
    ) -> dict[str, float]:
        """Calculate the total contributions to the TRPL
        :param k_T: trapping rate constant (ns-1)
        :param k_B: bimolecular rate constant (cm3/ns)
        :param k_A: Auger rate constant (cm6/ns)
        :param mu: carrier mobility (cm2/Vs)
        :param n: carrier concentration (cm-3)
        :param t: time (ns)"""

        T = sci.trapezoid(2 * mu * k_T * n, t)
        B = sci.trapezoid(2 * mu * k_B * n**2, t)
        A = sci.trapezoid(2 * mu * k_A * n**3, t)
        S = T + B + A
        return {"T": T / S * 100, "B": B / S * 100, "A": A / S * 100}

    def generate_decays(self, *args, **kwargs):
        """Generate decays.
        :param args: arguments passed to _generate_decays.
        :param kwargs: keyword arguments passed to _generate_decays."""

        return self._generate_decays(1000.0, BT_N0s, BT_KWARGS, *args, **kwargs)


# ------------------------------------------------------ BTD MODEL -----------------------------------------------------


class BTDModel(Model):
    """Class for the Bimolecular-Trapping-Detrapping model"""

    def __init__(self, param_ids):

        units = {
            "N_T": "cm-3",
            "p_0": "cm-3",
            "k_B": "cm3/ns",
            "k_T": "cm3/ns",
            "k_D": "cm3/ns",
            "mu_e": "cm^2/(V s)",
            "mu_h": "cm^2/(V s)",
        }
        units_html = {
            "N_T": "cm<sup>-3</sup>",
            "p_0": "cm<sup>-3</sup>",
            "k_B": "cm<sup>3</sup>/ns",
            "k_T": "cm<sup>3</sup>/ns",
            "k_D": "cm<sup>3</sup>/ns",
            "mu_e": "cm<sup>2</sup>/(V s)",
            "mu_h": "cm<sup>2</sup>/(V s)",
        }
        factors = {
            "k_B": 1e-20,
            "k_T": 1e-20,
            "k_D": 1e-20,
            "p_0": 1e12,
            "N_T": 1e12,
            "mu_e": 1,
            "mu_h": 1,
        }
        fvalues = {
            "k_T": None,
            "k_B": None,
            "k_D": None,
            "N_T": None,
            "p_0": None,
            "mu_e": None,
            "mu_h": None,
        }
        gvalues = {
            "k_B": 30e-20,
            "k_T": 12000e-20,
            "k_D": 80e-20,
            "N_T": 60e12,
            "p_0": 65e12,
            "mu_e": 10,
            "mu_h": 10,
        }
        gvalues_range = {
            "k_B": [1e-20, 1e-18],
            "k_T": [1e-18, 1e-16],
            "k_D": [1e-20, 1e-18],
            "p_0": [1e12, 1e14],
            "N_T": [1e12, 1e14],
            "mu_e": [10],
            "mu_h": [10],
        }
        n_keys = ["n_e", "n_t", "n_h"]  # need to be ordered same way as rate_equations input
        n_init = lambda N_0: {"n_e": N_0, "n_t": 0, "n_h": N_0}
        conc_ca_ids = ["n_e", "n_h"]

        Model.__init__(
            self,
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
            ["k_B < k_T", "k_D < k_T"],
        )

    @staticmethod
    def _rate_equations(
        n_e: float,
        n_t: float,
        n_h: float,
        k_B: float,
        k_T: float,
        k_D: float,
        p_0: float,
        N_T: float,
        **kwargs,
    ) -> dict[str, float]:
        """Rate equations of the BTD model
        :param n_e: electron concentration (cm-3)
        :param n_t: trapped electron concentration (cm-3)
        :param n_h: hole concentration (cm-3)
        :param k_B: bimolecular recombination rate constant (cm3/ns)
        :param k_T: trapping rate constant (cm3/ns)
        :param k_D: detrapping rate constant (cm3/ns)
        :param p_0: doping concentration (cm-3)
        :param N_T: trap states concentration (cm-3)"""

        B = k_B * n_e * (n_h + p_0)
        T = k_T * n_e * (N_T - n_t)
        D = k_D * n_t * (n_h + p_0)
        dne_dt = -B - T
        dnt_dt = T - D
        dnh_dt = -B - D
        return {"n_e": dne_dt, "n_t": dnt_dt, "n_h": dnh_dt}

    def calculate_trpl(
        self,
        t: np.ndarray,
        N_0: float,
        p_0: float,
        I: float,
        y_0: float,
        **kwargs,
    ) -> np.ndarray:
        """Calculate the normalised TRPL intensity
        :param t: time (ns)
        :param N_0: initial carrier concentration (cm-3)
        :param y_0: background intensity
        :param p_0: doping concentration (cm-3)
        :param I: amplitude factor
        :param kwargs: keyword arguments passed to the calculate_concentrations function"""

        n = self._calculate_concentrations(t, N_0, p_0=p_0, **kwargs)
        I_TRPL = n["n_e"][-1] * (n["n_h"][-1] + p_0) / N_0
        return I * I_TRPL / I_TRPL[0] + y_0

    def calculate_trmc(
        self,
        t: np.ndarray,
        N_0: float,
        p_0: float,
        mu_e: float,
        mu_h: float,
        y_0: float,
        **kwargs,
    ) -> np.ndarray:
        """Calculate the normalised TRPL intensity
        :param t: time (ns)
        :param N_0: initial carrier concentration (cm-3)
        :param y_0: background intensity
        :param p_0: doping concentration (cm-3)
        :param mu_e: electron mobility (cm2/Vs)
        :param mu_h: hole mobility (cm2/Vs)
        :param kwargs: keyword arguments passed to the calculate_concentrations function"""

        n = self._calculate_concentrations(t, N_0, p_0=p_0, **kwargs)
        return (mu_e * n["n_e"][-1] + mu_h * n["n_h"][-1]) / N_0 + y_0

    def get_contribution_recommendations(
        self,
        contributions: dict[str, np.ndarray],
        threshold: float = 10.0,
    ) -> list[str]:
        """Get recommendations for the contributions
        :param contributions: contribution dictionary
        :param threshold: threshold below which a warning is given"""

        recs = []
        for key in contributions:
            if np.max(contributions[key]) < threshold:
                if key == "B" and self.fvalues.get("k_B", 1) != 0:
                    recs.append(self.get_contribution_recommendation("bimolecular", "higher"))
                elif key == "T" and self.fvalues.get("k_T", 1) != 0 and self.fvalues.get("N_T", 1) != 0:
                    recs.append(self.get_contribution_recommendation("trapping", "lower"))
                elif key == "D" and self.fvalues.get("k_D", 1) != 0:
                    recs.append(self.get_contribution_recommendation("detrapping"))
        recs.append(
            "Note: For the bimolecular-trapping-detrapping model, although a low contribution suggests that the "
            "parameter associated with the process are not be accurate, a non-negligible contribution does not "
            "automatically indicate that the parameters retrieved are accurate due to the complex nature of the "
            "model. It is recommended to perform a grid fitting analysis with this model."
        )
        return recs


class BTDModelTRPL(BTDModel):

    def __init__(self):
        BTDModel.__init__(self, ["k_B", "k_T", "k_D", "N_T", "p_0", "y_0", "I"])

    def calculate_fit_quantity(self, *args, **kwargs):

        return self.calculate_trpl(*args, **kwargs)

    @staticmethod
    def calculate_contributions(
        t: np.ndarray,
        k_T: float,
        k_B: float,
        k_D: float,
        N_T: float,
        p_0: float,
        n_e: np.ndarray,
        n_t: np.ndarray,
        n_h: np.ndarray,
        **kwargs,
    ) -> dict[str, float]:
        """Calculate the total contributions to the TRPL
        :param k_T: trapping rate constant (cm3/ns)
        :param k_B: bimolecular rate constant (cm3/ns)
        :param k_D: Auger detrapping rate constant (cm3/ns)
        :param N_T: trap state concentration (cm-3)
        :param p_0: doping concentration (cm-3)
        :param n_e: electron concentration (cm-3)
        :param n_t: trapped electron concentration (cm-3)
        :param n_h: hole concentration (cm-3)
        :param t: time (ns)"""

        T = sci.trapezoid(k_T * n_e * (N_T - n_t) * (n_h + p_0), t)
        B = sci.trapezoid(k_B * n_e * (n_h + p_0) * (n_e + n_h + p_0), t)
        D = sci.trapezoid(k_D * n_t * (n_h + p_0) * n_e, t)
        S = T + B + D
        return {"T": T / S * 100, "B": B / S * 100, "D": D / S * 100}

    def generate_decays(self, *args, **kwargs):
        """Generate decays.
        :param args: arguments passed to _generate_decays.
        :param kwargs: keyword arguments passed to _generate_decays."""

        return self._generate_decays(10e3, BTD_N0s, BTD_KWARGS, normalise=True, *args, **kwargs)


class BTDModelTRMC(BTDModel):

    def __init__(self):
        BTDModel.__init__(self, ["k_B", "k_T", "k_D", "N_T", "p_0", "mu_e", "mu_h", "y_0"])

    def calculate_fit_quantity(self, *args, **kwargs):

        return self.calculate_trmc(*args, **kwargs)

    @staticmethod
    def calculate_contributions(
        t: np.ndarray,
        k_T: float,
        k_B: float,
        k_D: float,
        N_T: float,
        p_0: float,
        n_e: np.ndarray,
        n_t: np.ndarray,
        n_h: np.ndarray,
        mu_e: float,
        mu_h: float,
        **kwargs,
    ) -> dict[str, float]:
        """Calculate the total contributions to the TRPL
        :param k_T: trapping rate constant (cm3/ns)
        :param k_B: bimolecular rate constant (cm3/ns)
        :param k_D: Auger detrapping rate constant (cm3/ns)
        :param N_T: trap state concentration (cm-3)
        :param p_0: doping concentration (cm-3)
        :param n_e: electron concentration (cm-3)
        :param n_t: trapped electron concentration (cm-3)
        :param n_h: hole concentration (cm-3)
        :param mu_e: electron mobility (cm2/Vs)
        :param mu_h: hole mobility (cm2/Vs)
        :param t: time (ns)"""

        T = sci.trapezoid(k_T * n_e * (N_T - n_t) * mu_e, t)
        B = sci.trapezoid(k_B * n_e * (n_h + p_0) * (mu_e + mu_h), t)
        D = sci.trapezoid(k_D * n_t * (n_h + p_0) * mu_h, t)
        S = T + B + D
        return {"T": T / S * 100, "B": B / S * 100, "D": D / S * 100}

    def generate_decays(self, *args, **kwargs):
        """Generate decays.
        :param args: arguments passed to _generate_decays.
        :param kwargs: keyword arguments passed to _generate_decays."""

        return self._generate_decays(50e3, BTD_N0s, BTD_KWARGS, *args, **kwargs)


models = {
    "BTA": {"TRPL": BTModelTRPL(), "TRMC": BTModelTRMC()},
    "BTD": {"TRPL": BTDModelTRPL(), "TRMC": BTDModelTRMC()},
}
