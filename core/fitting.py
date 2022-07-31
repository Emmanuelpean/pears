""" fitting module """

from core import utils

import numpy as np
import scipy.optimize as sco
import itertools


class Fit(object):
    """ Fit class """

    def __init__(self, xs_data, ys_data, function, p0, detached_parameters, fixed_parameters, **kwargs):
        """ Object constructor
        :param list, np.ndarray xs_data: array or list of arrays associated with the x data
        :param list, np.ndarray ys_data: array or list of arrays associated with the y data
        :param function function: function used to fit the data
        :param dict p0: parameters initial guess. Need to contain any non fixed parameters
        :param list detached_parameters: list of parameters that are not shared between y data
        :param list of dicts fixed_parameters: fixed parameters dict for each y data
        :param kwargs: keyword arguments passed to the scipy.optimise.least_square function

        Example
        -------
        >>> from core import models
        >>> from core import resources
        >>> x_data1 = resources.BT_TRPL_np[0]
        >>> ys_data1 = resources.BT_TRPL_np[1:]
        >>> xs_data1 = [x_data1] * len(ys_data1)
        >>> function1 = models.BTModel().calculate_fit_quantity
        >>> p01 = dict(k_T=1e-3, k_B=1e-20, k_A=1e-40, y_0=0.01, I=1)
        >>> detached_parameters1 = ['y_0']
        >>> fixed_parameters1 = [dict(N_0=1e15), dict(N_0=1e16), dict(N_0=1e17)]
        >>> a = Fit(xs_data1, ys_data1, function1, p01, detached_parameters1, fixed_parameters1)
        >>> a.list_to_dicts(a.p0_list)[-1]
        {'y_0': 0.010000000000000002, 'k_T': 0.001, 'k_B': 1.0000000000000005e-20, 'k_A': 1.0000000000000005e-40, 'I': 1}
        >>> popts1 = a.fit()
        >>> popts1[0]
        {'k_T': 0.00484319120069832, 'k_B': 8.381129570270533e-20, 'k_A': 2.3267038027561887e-52, 'y_0': 0.0374458235167957, 'I': 0.8939331193339572, 'N_0': 1000000000000000.0}
        >>> fits1 = a.calculate_fits(popts1)
        >>> a.calculate_rss(fits1)
        0.9907935203410436"""

        self.xs_data = xs_data
        self.ys_data = ys_data
        self.function = function
        self.p0 = p0
        self.detached_parameters = [f for f in detached_parameters if f not in fixed_parameters[0]]
        self.fixed_parameters = fixed_parameters
        self.kwargs = kwargs
        self.n = len(self.xs_data)

        # Remove fixed parameters from initial guess
        self.p0 = {key: value for key, value in self.p0.items() if key not in self.fixed_parameters[0]}
        self.keys = list(self.p0.keys())  # all keys except for the ones fixed

        # Split the initial guess values to facilitate the optimisation
        p0_split = {key: utils.normalise_number(self.p0[key]) for key in self.p0}
        self.factors = {key: p0_split[key][1] for key in p0_split}
        self.p0 = {key: p0_split[key][0] for key in p0_split}

        # Lists
        self.p0_list = [self.p0[key] for key in self.keys]
        self.bounds = {key: [0, np.inf] for key in self.keys}
        self.bounds_list = [self.bounds[key] for key in self.keys]

        # Global fitting
        self.p0_list += [self.p0[key] for key in self.detached_parameters] * (self.n - 1)  # add the detached parameters to the list
        self.bounds_list = np.transpose(self.bounds_list + [self.bounds[key] for key in self.detached_parameters] * (self.n - 1))

    def list_to_dicts(self, alist):
        """ Convert a list of numbers to a list of dicts given a list of keys, and repeated keys associated with numbers at
        the end of the list
        :param list, tuple alist: list of numbers associated with the keys """

        base_dict = dict(zip(self.keys, alist))  # create the base dict
        if self.detached_parameters:
            supplementary = np.array(alist[len(self.keys):]).reshape((-1, len(self.detached_parameters)))  # supplementary values
            dicts = [utils.merge_dicts(dict(zip(self.detached_parameters, p)), base_dict) for p in supplementary]  # supplementary dicts
            dicts = [base_dict] + dicts
        else:
            dicts = [base_dict] * self.n
        return [{k: ps[k] * 10 ** self.factors[k] for k in ps} for ps in dicts]

    def error_function(self, alist):
        """ Error function
        :param list alist: list of parameters values"""

        dicts = self.list_to_dicts(alist)
        errors = []
        for d, x, y, fp in zip(dicts, self.xs_data, self.ys_data, self.fixed_parameters):
            errors.append((self.function(x, **utils.merge_dicts(d, fp)) - y))
        return np.concatenate(errors)

    def fit(self):
        """ Fit the data """

        popts = sco.least_squares(self.error_function, self.p0_list, bounds=self.bounds_list, jac='3-point', **self.kwargs).x
        popts_dicts = self.list_to_dicts(popts)
        return [utils.merge_dicts(i, j) for i, j in zip(popts_dicts, self.fixed_parameters)]

    def calculate_rss(self, y2):
        """ Calculate the residual sum of squares """

        y1 = np.concatenate(self.ys_data)
        y2 = np.concatenate(y2)
        return 1. - np.sum((y1 - y2) ** 2, axis=-1) / np.sum((y1 - np.mean(y1)) ** 2)

    def calculate_fits(self, popts):
        """ Calculate the fits
        :param list popts: list of optimised parameter dicts """

        return [self.function(x, **popt) for x, popt in zip(self.xs_data, popts)]


def run_grid_fit(p0s, fixed_parameters, filters, progressbar=None, **kwargs):
    """ Run a grid of fits each with different initial guess values
    :param dict p0s: dictionary with each key associated with a list of floats. Need to contain all the parameters.
    :param fixed_parameters: argument of the least_square function
    :param filters: list of filters
    :param progressbar: streamlit progressbar
    :param kwargs: keyword argument passed to the least_square function
    :return a dict of the optimised values, a dict of the guess values and an array of CODs

    Example
    -------
    >>> from core import models
    >>> from core import resources
    >>> x_data1 = resources.BT_TRPL_np[0]
    >>> ys_data1 = resources.BT_TRPL_np[1:]
    >>> xs_data1 = [x_data1] * len(ys_data1)
    >>> function1 = models.BTModel().calculate_fit_quantity
    >>> detached_parameters1 = ['y_0']
    >>> fixed_parameters1 = [dict(N_0=1e15), dict(N_0=1e16), dict(N_0=1e17)]
    >>> p0s1 = dict(k_T=[1e-2, 1e-4], k_B=[1e-20, 1e-19], k_A=[1e-40, 1e-45], y_0=[0], I=[1])
    >>> a = run_grid_fit(p0s1, fixed_parameters1, None, None, xs_data=xs_data1, ys_data=ys_data1, function=function1, detached_parameters=detached_parameters1) """

    # Calculate all the combinations of parameters
    p0s = {key: p0s[key] for key in p0s if key not in fixed_parameters[0]}   # filter out the fixed parameters
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
            popts = [{key: [float('nan')] for key in pkeys}]
            cod = float('nan')

        data_opt.append(popts[0])
        data_init.append(utils.merge_dicts(p0, fixed_parameters[0]))
        cods.append(cod)

    data_opt = {key: np.array([p[key] for p in data_opt]) for key in data_opt[0]}
    data_init = {key: np.array([p[key] for p in data_init]) for key in data_init[0]}

    return data_opt, data_init, np.array(cods)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
