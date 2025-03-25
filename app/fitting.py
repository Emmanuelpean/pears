"""fitting module"""

import itertools

import numpy as np
import scipy.optimize as sco

import utils


class Fit(object):
    """Fit class"""

    def __init__(
        self,
        xs_data: list[np.ndarray],
        ys_data: list[np.ndarray],
        function: callable,
        p0: dict,
        detached_parameters: list[str],
        fixed_parameters: list[dict],
        **kwargs,
    ):
        """Object constructor
        :param xs_data: array or list of arrays associated with the x data
        :param ys_data: array or list of arrays associated with the y data
        :param function: function used to fit the data
        :param p0: parameters initial guess. Need to contain any non-fixed parameters
        :param detached_parameters: list of parameters that are not shared between y data
        :param fixed_parameters: fixed parameters dict for each y data
        :param kwargs: keyword arguments passed to the scipy.optimise.least_square function"""

        # Store the input arguments
        self.xs_data = xs_data
        self.ys_data = ys_data
        self.function = function
        self.p0 = p0
        self.fixed_parameters = fixed_parameters
        self.kwargs = kwargs

        # Make sure that the fixed parameters are all the same and that the lengths match
        assert all(d.keys() == fixed_parameters[0].keys() for d in fixed_parameters)
        assert len(self.fixed_parameters) == len(self.xs_data) == len(self.ys_data)

        # Remove fixed parameters from the detached parameters
        self.detached_parameters = [f for f in detached_parameters if f not in self.fixed_parameters[0]]

        # Remove fixed parameters from initial guess
        self.p0 = {key: value for key, value in self.p0.items() if key not in self.fixed_parameters[0]}
        self.keys = list(self.p0.keys())  # all keys except for the ones fixed

        # Normalise the initial guess values to unit to facilitate the optimisation
        p0_split = {key: utils.normalize_to_unit(self.p0[key]) for key in self.p0}
        self.p0_mantissa, self.p0_factors = [{key: val[i] for key, val in p0_split.items()} for i in (0, 1)]

        # Generate the positive bounds
        self.bounds = {key: [0, np.inf] for key in self.keys}

        # Convert the guess values and bounds to list for the scipy curve fitting
        self.p0_list = [self.p0_mantissa[key] for key in self.keys]
        self.bounds_list = [self.bounds[key] for key in self.keys]

        # Add the detached parameters to the list
        self.n = len(self.xs_data)  # number of xs arrays
        self.p0_list += [self.p0_mantissa[key] for key in self.detached_parameters] * (self.n - 1)
        self.bounds_list += [self.bounds[key] for key in self.detached_parameters] * (self.n - 1)

    def list_to_dicts(self, alist: list) -> list[dict]:
        """Convert the list of guess values (including detached parameters) into a list of dictionaries
        :param alist: list of guess values associated with the keys"""

        # Create the base dict (with any potential first instance of the detached parameter)
        base_dict = dict(zip(self.keys, alist))

        # If detached parameters, extract them from the list and add them to the dictionaries
        if self.detached_parameters:
            # Determine the value of the detached parameters
            supp = np.array(alist[len(self.keys) :]).reshape((-1, len(self.detached_parameters)))
            # Add the detached values to each dictionary
            dicts = [utils.merge_dicts(dict(zip(self.detached_parameters, p)), base_dict) for p in supp]
            # Add the base dictionary to the list
            dicts = [base_dict] + dicts

        # If no detached parameters, just multiply the number of dictionaries
        else:
            dicts = [base_dict] * self.n

        # Return the values multiplied by their factor
        return [{k: ps[k] * 10 ** self.p0_factors[k] for k in ps} for ps in dicts]

    def error_function(self, alist: list) -> np.ndarray:
        """Calculate the difference between the output of the fit function and each array for the given list of
        parameter values.
        :param alist: list of parameter values"""

        params_list = self.list_to_dicts(alist)
        errors = []
        for params, x, y, fp in zip(params_list, self.xs_data, self.ys_data, self.fixed_parameters):
            errors.append((self.function(x, **utils.merge_dicts(params, fp)) - y))
        return np.concatenate(errors)

    def fit(self) -> list[dict]:
        """Fit the data"""

        popts = sco.least_squares(
            self.error_function,
            self.p0_list,
            bounds=np.transpose(self.bounds_list),
            jac="3-point",
            **self.kwargs,
        ).x
        popts_dicts = self.list_to_dicts(popts)
        return [utils.merge_dicts(i, j) for i, j in zip(popts_dicts, self.fixed_parameters)]

    def calculate_rss(self, y2: list[np.ndarray]) -> float:
        """Calculate the residual sum of squares"""

        y1 = np.concatenate(self.ys_data)
        y2 = np.concatenate(y2)
        return 1.0 - np.sum((y1 - y2) ** 2, axis=-1) / np.sum((y1 - np.mean(y1)) ** 2)

    def calculate_fits(self, popts: list[dict]) -> list[np.ndarray]:
        """Calculate the fits
        :param popts: list of optimised parameter dicts"""

        return [self.function(x, **popt) for x, popt in zip(self.xs_data, popts)]


def run_grid_fit(
    p0s: dict[str, list],
    fixed_parameters: list[dict],
    filters: list[str] | None,
    progressbar=None,
    **kwargs,
) -> tuple[dict, dict, np.ndarray]:
    """Run a grid of fits each with different initial guess values
    :param p0s: dictionary with each key associated with a list of floats. Need to contain all the parameters.
    :param fixed_parameters: argument of the least_square function
    :param filters: list of string filters
    :param progressbar: optional streamlit progressbar
    :param kwargs: keyword argument passed to the least_square function
    :return a dict of the optimised values, a dict of the guess values and an array of CODs"""

    # Calculate all the combinations of parameters
    p0s = {key: p0s[key] for key in p0s if key not in fixed_parameters[0]}  # filter out the fixed parameters
    pkeys, pvalues = zip(*p0s.items())
    all_p0s = [dict(zip(pkeys, v)) for v in itertools.product(*pvalues)]
    if filters is not None:
        all_p0s = utils.filter_dicts(all_p0s, filters, fixed_parameters[0])

    # Run the fits
    data_opt, data_init, cods = [], [], []
    for i, p0 in enumerate(all_p0s):

        # Update the progressbar if provided
        if progressbar is not None:
            progressbar.progress(i / float(len(all_p0s) - 1))

        try:
            fit = Fit(p0=p0, fixed_parameters=fixed_parameters, **kwargs)
            popts = fit.fit()
            ys_fit = fit.calculate_fits(popts)
            cod = fit.calculate_rss(ys_fit)
        except ValueError:
            popts = [{key: [float("nan")] for key in pkeys}]
            cod = float("nan")

        data_opt.append(popts[0])
        data_init.append(utils.merge_dicts(p0, fixed_parameters[0]))
        cods.append(cod)

    data_opt = {key: np.array([p[key] for p in data_opt]) for key in data_opt[0]}
    data_init = {key: np.array([p[key] for p in data_init]) for key in data_init[0]}

    return data_opt, data_init, np.array(cods)
