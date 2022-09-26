""" models module """

import numpy as np
import scipy.integrate as sci
import itertools

from core import utils
from core import fitting


class Model(object):
    """ Parent Class for the charge carrier recombination models """

    def __init__(self, ids, units, units_html, factors, fvalues, gvalues, gvalues_range, n_keys, n_init, conc_ca_ids,
                 param_filters):
        """ Object constructor
        :param list ids: parameter ids
        :param dict units: parameter units
        :param dict units_html: parameter units (html)
        :param dict factors: parameter factors
        :param dict fvalues: parameter fixed values
        :param dict gvalues: parameter guess values
        :param dict gvalues_range: parameter guess values range """

        self.s_display_keys = ids.copy()  # single value display keys
        self.m_display_keys = ['y_0', 'I']  # multiple values display keys
        self.ids = ids + ['y_0', 'I']
        self.units = utils.merge_dicts(units, {'y_0': '', 'I': ''})
        self.units_html = utils.merge_dicts(units_html, {'y_0': '', 'I': ''})
        self.factors = utils.merge_dicts(factors, {'y_0': 1., 'I': 1})
        self.fvalues = utils.merge_dicts(fvalues, {'y_0': 0., 'I': 1.})
        self.gvalues = utils.merge_dicts(gvalues, {'y_0': 0., 'I': 1.})
        self.gvalues_range = utils.merge_dicts(gvalues_range, {'y_0': [0.], 'I': [1.]})
        self.detached_parameters = ['y_0', 'I']
        self.n_keys = n_keys
        self.n_init = n_init
        self.conc_ca_ids = conc_ca_ids
        self.param_filters = param_filters

        # Quantities display
        self.quantities = {'k_B': 'Bimolecular recombination rate constant', 'k_T': 'Trapping rate constant',
                           'k_D': 'Detrapping rate constant', 'k_A': 'Auger recombination rate constant',
                           'N_T': 'Trap state concentration', 'p_0': 'Doping concentration', 'y_0': 'Intensity offset',
                           'I': 'Intensity factor'}

        # Symbols display
        self.symbols = {'k_B': 'k_B', 'k_T': 'k_T', 'k_D': 'k_D', 'k_A': 'k_A', 'N_T': 'N_T', 'p_0': 'p_0',
                        'y_0': 'y_0', 'I': 'I_0'}
        self.symbols_html = {'N_T': 'N<sub>T</sub>', 'p_0': 'p<sub>0</sub>', 'k_B': 'k<sub>B</sub>',
                             'k_T': 'k<sub>T</sub>',
                             'k_D': 'k<sub>D</sub>', 'k_A': 'k<sub>A</sub>', 'y_0': 'y<sub>0</sub>', 'I': 'I<sub>0</sub>'}

        self.labels = {pkey: self.get_parameter_label(self.symbols[pkey], self.units[pkey]) for pkey in
                       self.ids}  # symbol + unit
        self.labels_i = {value: key for key, value in self.labels.items()}  # inverted dict for param_labels

        self.factors_html = {key: utils.get_power_text(self.factors[key], None) for key in self.factors}

        # Contributions
        self.cbt_labels = {'A': 'Auger (%)', 'B': 'Bimolecular (%)', 'T': 'Trapping (%)', 'D': 'Detrapping (%)'}

        self.n_labels_html = {'n_e': 'n<sub>e</sub>', 'n_t': 'n<sub>t</sub>', 'n_h': 'n<sub>h</sub>', 'n': 'n'}
        self.n_colors = {'n_e': 'red', 'n_t': 'green', 'n_h': 'blue', 'n': 'black'}

    def __eq__(self, other):
        """ self == other """

        if self.fvalues == other.fvalues and self.gvalues == other.gvalues and self.gvalues_range == other.gvalues_range:
            return True
        else:
            return False

    @staticmethod
    def get_parameter_label(symbol, unit):
        """ Get the parameter label
        :param str symbol: symbol
        :param str unit: unit """

        if unit == '':
            return symbol
        else:
            return symbol + ' (' + unit + ')'

    @property
    def fixed_values(self):
        """ return the dict of fixed values """

        return {key: value for key, value in self.fvalues.items() if value is not None}

    # -------------------------------------------------- CORE METHODS --------------------------------------------------

    def rate_equations(self, *args, **kwargs):
        """ Rate equation method """

        return {}

    def calculate_concentrations(self, t, N_0, p=1, threshold=0.001, **kwargs):
        """ Calculate the carrier concentrations as per the BTD model
        :param t: time (ns)
        :param N_0: initial carrier concentration (cm-3)
        :param p: number of pulses
        :param threshold: threshold
        :param kwargs: keyword arguments passed to the rate equations

        Example
        -------
        >>> _ = BTModel().calculate_concentrations(np.linspace(0, 100), N_0=1e17, k_B=50e-20, k_T=1e-3, k_A=1e-40)
        >>> _ = BTModel().calculate_concentrations(np.linspace(0, 100), N_0=1e17, k_B=50e-20, k_T=1e-3, k_A=1e-40, p=1000)
        >>> len(_['n'])
        5"""

        var = {key: np.zeros(len(t)) for key in self.n_keys}
        variables = {key: [] for key in self.n_keys}

        def rate_equation(x, _t):
            """ Rate equation wrapper """

            dndt = self.rate_equations(*x, **kwargs)
            return [dndt[key] for key in self.n_keys]

        for i in range(p):
            init = self.n_init(N_0)
            var = np.transpose(sci.odeint(rate_equation, [init[key] + var[key][-1] for key in self.n_keys], t))
            var = dict(zip(self.n_keys, var))

            for key in var:
                variables[key].append(var[key])

            if threshold > 0. and i > 1:
                ca = np.array([np.max(np.abs(variables[key][-2] - var[key]) / N_0) for key in self.conc_ca_ids])
                if all(ca < threshold / 100):
                    break
                elif i == p:
                    raise AssertionError('Attention: threshold condition never reached')

        return variables

    def calculate_trpl(self, *args, **kwargs):
        """ Calculate the TRPL """

        return np.array([0])

    def calculate_contributions(self, *args, **kwargs):
        """ Calculate the contributions """

        return dict()

    # ------------------------------------------------- FITTING METHODS ------------------------------------------------

    def fit(self, xs_data, ys_data, N0s, p0=None):
        """ Fit the data using the model
        :param list, np.ndarray xs_data: list-like of x data
        :param list, np.ndarray ys_data: list-like of y data
        :param list N0s: list of initial carrier concentrations (cm-3)
        :param None, dict p0: guess values. If None, use the gvalues dict

        Example
        -------
        >>> from core import models
        >>> from core import resources
        >>> x_data1 = resources.test_file1[0]
        >>> ys_data1 = resources.test_file1[1:]
        >>> xs_data1 = [x_data1] * len(ys_data1)
        >>> N0s1 = [1e15, 1e16, 1e17]
        >>> fit1 = BTModel().fit(xs_data1, ys_data1, N0s1) """

        # Input parameters processing
        fparams = [utils.merge_dicts(self.fixed_values, dict(N_0=n0)) for n0 in N0s]  # add N0 to fixed parameters
        if p0 is None:
            p0 = self.gvalues

        # Fitting
        fit = fitting.Fit(xs_data, ys_data, self.calculate_trpl, p0, self.detached_parameters, fparams)

        # Popts, fitted data, R2
        popts = fit.fit()
        fit_ydata = fit.calculate_fits(popts)
        cod = fit.calculate_rss(fit_ydata)
        popt = utils.keep_function_kwargs(self.rate_equations, popts[0])  # optimised values
        p0 = utils.keep_function_kwargs(self.rate_equations, utils.merge_dicts(p0, self.fixed_values))  # guess values

        # Popts full labels
        labels = []
        for key in self.s_display_keys + self.m_display_keys:
            fstring = ' (fixed)' if key in fparams[0] else ''  # add 'fixed' if parameter is fixed
            if key in self.s_display_keys:
                data = popts[0][key] / self.factors[key]
                label = self.quantities[key] + ' (' + self.symbols_html[key] + '): ' + '%.5f' % data + ' &#10005; ' \
                             + self.factors_html[key] + ' ' + self.units_html[key] + fstring
            else:
                data = ['%.5f' % popt[key] for popt in popts]
                label = self.quantities[key] + ' (' + self.symbols_html[key] + '): ' + ', '.join(data) + fstring
            labels.append(label)
        labels.append('Coefficient of determination R<sup>2</sup>: ' + str(cod))

        # Contributions
        contributions = []
        for x_data, p in zip(xs_data, popts):
            concentration = {key: value[0] for key, value in self.calculate_concentrations(x_data, **p).items()}
            kwargs = utils.merge_dicts(p, concentration)
            contributions.append({self.cbt_labels[key]: value for key, value in self.calculate_contributions(x_data, **kwargs).items()})
        contributions = utils.list_to_dict(contributions)

        # All values (guess, optimised, R2 and contributions)
        values = dict()
        no_disp_keys = []
        for key in popt:
            label = self.symbols_html[key] + ' (' + self.factors_html[key] + ' ' + self.units_html[key] + ')'
            if key in fparams[0]:
                values[label + ' (fixed)'] = p0[key] / self.factors[key]
                no_disp_keys.append(label + ' (fixed)')
            else:
                values[label + ' (guess)'] = p0[key] / self.factors[key]
                values[label + ' (opt.)'] = popt[key] / self.factors[key]
                no_disp_keys.append(label + ' (guess)')
        values['R<sub>2</sub>'] = cod
        for key in contributions:
            values['max. ' + key] = np.max(contributions[key])

        return {'popts': popts, 'popt': popt, 'cod': cod, 'contributions': contributions, 'fit_ydata': fit_ydata,
                'labels': labels, 'N0s_labels': utils.get_power_labels(N0s), 'p0': p0, 'N0s': N0s, 'values': values,
                'no_disp_keys': no_disp_keys, 'xs_data': xs_data, 'ys_data': ys_data}

    def get_carrier_accumulation(self, popts, N0s, period):
        """ Calculate the carrier accumulation effect on the TRPL
        :param list popts: list of optimised parameters
        :param list N0s: initial carrier concentrations
        :param float, int period: excitation repetition period

        Example
        -------
        >>> from core import models
        >>> from core import resources
        >>> x_data1 = resources.test_file1[0]
        >>> ys_data1 = resources.test_file1[1:]
        >>> xs_data1 = [x_data1] * len(ys_data1)
        >>> N0s1 = [1e15, 1e16, 1e17]
        >>> fit1 = BTModel().fit(xs_data1, ys_data1, N0s1)
        >>> list(BTModel().get_carrier_accumulation(fit1['popts'], fit1['N0s'], 1000).values())
        [0.02057525783717984, 0.12353636524250478, 0.11835604643279929]"""

        labels = utils.get_power_labels(N0s)
        nca = dict()
        for label, popt, key in zip(labels, popts, N0s):
            x = np.insert(np.logspace(-4, np.log10(period), 10001), 0, 0)
            pulse1 = self.calculate_trpl(x, **popt)
            pulse2 = self.calculate_trpl(x, p=1000, **popt)
            nca[label] = np.max(np.abs(pulse1 / pulse1[0] - pulse2 / pulse2[0])) * 100
        return nca

    def get_carrier_concentrations(self, xs_data, popts, period):
        """ Calculate the carrier concentrations from the optimised parameters
        :param list xs_data: x axis data
        :param list popts: list of optimised parameters
        :param float, int period: excitation repetition period in ns

        Example
        -------
        >>> from core import models
        >>> from core import resources
        >>> x_data1 = resources.test_file1[0]
        >>> ys_data1 = resources.test_file1[1:]
        >>> xs_data1 = [x_data1] * len(ys_data1)
        >>> N0s1 = [1e15, 1e16, 1e17]
        >>> fit1 = BTModel().fit(xs_data1, ys_data1, N0s1)
        >>> _ = BTModel().get_carrier_concentrations(xs_data1, fit1['popts'], 1000.) """

        # Carrier concentrations
        if period == '':
            x_pulse = x = xs_data
            nb_pulses = 1
            xlabel = 'Time (ns)'
        else:
            x = np.array([np.linspace(0, float(period), 1001)] * len(xs_data))
            nb_pulses = 100
            x_pulse = [np.linspace(0, nb_pulses, len(x_) * nb_pulses) for x_ in x]
            xlabel = 'Pulse'

        concentrations = []
        for x_data, popt in zip(x, popts):
            kwargs = {key: popt[key] for key in popt if key not in ('I', 'y_0')}
            concentration = self.calculate_concentrations(x_data, p=nb_pulses, **kwargs)
            concentrations.append({key: np.concatenate(concentration[key]) for key in concentration})

        return x_pulse, xlabel, concentrations

    def grid_fitting(self, progressbar, N0s, **kwargs):
        """ Run a grid fitting analysis
        :param list N0s: initial carrier concentration
        :param progressbar: progressbar
        :param kwargs: keyword arguments passed to the fit method

        >>> from core import models
        >>> from core import resources
        >>> x_data1 = resources.test_file1[0]
        >>> ys_data1 = resources.test_file1[1:]
        >>> xs_data1 = [x_data1] * len(ys_data1)
        >>> N0s1 = [1e15, 1e16, 1e17]
        >>> analysis = BTModel().grid_fitting(None, N0s1, xs_data=xs_data1, ys_data=ys_data1) """

        # Filter out the fixed parameters
        p0s = {key: self.gvalues_range[key] for key in self.gvalues_range if key not in self.fixed_values}

        # Generate all the combination of guess values and filter them
        pkeys, pvalues = zip(*p0s.items())
        all_p0s = [dict(zip(pkeys, v)) for v in itertools.product(*pvalues)]
        all_p0s = utils.filter_dicts(all_p0s, self.param_filters, self.fixed_values)

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

    @staticmethod
    def get_rec_string(process, val=''):
        """ Get the recommendation string for the contributions
        :param str process: name of the process
        :param str val: 'higher' or 'lower' """

        string = 'This fit predicts low %s. The values associated with this process may be inacurate.' % process
        if val:
            string += '\nIt is recommended to measure your sample under %s excitation fluence for this process to become ' \
                      'significant' % val
        return string


class BTModel(Model):
    """ Class for the Bimolecular-Trapping model """

    def __init__(self):
        """ Object constructor """

        ids = ['k_T', 'k_B', 'k_A']
        units = {'k_B': 'cm3/ns', 'k_T': 'ns-1', 'k_A': 'cm6/ns-1'}
        units_html = {'k_B': 'cm<sup>3</sup>/ns', 'k_T': 'ns<sup>-1</sup>', 'k_A': 'cm<sup>6</sup>/ns'}
        factors = {'k_B': 1e-20, 'k_T': 1e-3, 'k_A': 1e-40}
        fvalues = {'k_T': None, 'k_B': None, 'k_A': 0.}
        gvalues = {'k_T': 0.001, 'k_B': 1e-20, 'k_A': 1e-40}
        gvalues_range = {'k_B': [1e-20, 1e-18], 'k_T': [1e-4, 1e-2], 'k_A': [1e-32, 1e-30]}
        n_keys = ('n',)
        n_init = lambda N_0: {'n': N_0}
        conc_ca_ids = ('n',)

        Model.__init__(self, ids, units, units_html, factors, fvalues, gvalues, gvalues_range, n_keys, n_init,
                       conc_ca_ids, [])

    @staticmethod
    def calculate_contributions(t, k_T, k_B, k_A, n, **kwargs):
        """ Calculate the total contributions to the TRPL
        :param k_T: trapping rate constant (ns-1)
        :param k_B: bimolecular rate constant (cm3/ns)
        :param k_A: Auger rate constant (cm6/ns)
        :param n: carrier concentration (cm-3)
        :param t: time (ns)"""

        T = sci.trapz(k_T * n ** 2, t)
        B = sci.trapz(k_B * n ** 3, t)
        A = sci.trapz(k_A * n ** 4, t)
        S = T + B + A
        return {'T': T / S * 100, 'B': B / S * 100, 'A': A / S * 100}

    @staticmethod
    def rate_equations(n, k_T, k_B, k_A, **kwargs):
        """ Rate equation of the BT model
        :param n: carrier concentration (cm-3)
        :param k_T: trapping rate (ns-1)
        :param k_B: bimolecular recombination rate (cm3/ns)
        :param k_A: Auger recombination rate (cm6/ns) """

        return {'n': - k_T * n - k_B * n ** 2 - k_A * n ** 3}

    def calculate_trpl(self, t, N_0, I, y_0, **kwargs):
        """ Calculate the normalised TRPL intensity using the BT model
        :param t: time (ns)
        :param N_0: initial carrier concentration
        :param y_0: background intensity
        :param I: amplitude factor
        :param kwargs: keyword arguments passed to the calculate_concentrations function

        Examples
        --------
        >>> _ = BTModel().calculate_trpl(np.linspace(0, 100), N_0=1e17, p=1, k_B=50e-20, k_T=1e-3, k_A=1e-40, I=1, y_0=0)
        >>> _ = BTModel().calculate_concentrations(np.linspace(0, 100), N_0=1e17, k_B=50e-20, k_T=1e-3, k_A=1e-40, p=1000)"""

        n = self.calculate_concentrations(t, N_0, **kwargs)['n'][-1]
        I_TRPL = n ** 2 / N_0
        signal = I_TRPL / I_TRPL[0] + y_0
        return I * signal / signal[0]

    def get_recommendations(self, contributions, threshold=10):
        """ Get recommendations for the contributions """

        recs = []
        for key in contributions:
            if np.max(contributions[key]) < threshold:
                if 'Trapping' in key and self.fvalues.get('k_T', 1) != 0:
                    recs.append(self.get_rec_string('trapping', 'lower'))
                elif 'Bimolecular' in key and self.fvalues.get('k_B', 1) != 0:
                    recs.append(self.get_rec_string('bimolecular', 'higher'))
                elif 'Auger' in key and self.fvalues.get('k_A', 1) != 0:
                    recs.append(self.get_rec_string('Auger', 'higher'))
        return recs


class BTDModel(Model):
    """ Class for the Bimolecular-Trapping-Detrapping model """

    def __init__(self):
        ids = ['k_B', 'k_T', 'k_D', 'N_T', 'p_0']
        units = {'N_T': 'cm-3', 'p_0': 'cm-3', 'k_B': 'cm3/ns', 'k_T': 'cm3/ns', 'k_D': 'cm3/ns'}
        units_html = {'N_T': 'cm<sup>-3</sup>', 'p_0': 'cm<sup>-3</sup>', 'k_B': 'cm<sup>3</sup>/ns',
                      'k_T': 'cm<sup>3</sup>/ns', 'k_D': 'cm<sup>3</sup>/ns'}
        factors = {'k_B': 1e-20, 'k_T': 1e-20, 'k_D': 1e-20, 'p_0': 1e12, 'N_T': 1e12}
        fvalues = {'k_T': None, 'k_B': None, 'k_D': None, 'N_T': None, 'p_0': None}
        gvalues = {'k_B': 30e-20, 'k_T': 12000e-20, 'k_D': 80e-20, 'N_T': 60e12, 'p_0': 65e12}
        gvalues_range = {'k_B': [1e-20, 1e-18], 'k_T': [1e-18, 1e-16], 'k_D': [1e-20, 1e-18], 'p_0': [1e12, 1e14],
                         'N_T': [1e12, 1e14]}
        n_keys = ('n_e', 'n_t', 'n_h')  # need to be ordered same way as rate_equations input
        n_init = lambda N_0: {'n_e': N_0, 'n_t': 0, 'n_h': N_0}
        conc_ca_ids = ('n_e', 'n_h')

        Model.__init__(self, ids, units, units_html, factors, fvalues, gvalues, gvalues_range, n_keys, n_init,
                       conc_ca_ids, ['k_B < k_T', 'k_D < k_T'])

    @staticmethod
    def calculate_contributions(t, k_T, k_B, k_D, N_T, p_0, n_e, n_t, n_h, **kwargs):
        """ Calculate the total contributions to the TRPL
        :param k_T: trapping rate constant (cm3/ns)
        :param k_B: bimolecular rate constant (cm3/ns)
        :param k_D: Auger detrapping rate constant (cm3/ns)
        :param N_T: trap state concentration (cm-3)
        :param p_0: doping concentration (cm-3)
        :param n_e: electron concentration (cm-3)
        :param n_t: trapped electron concentration (cm-3)
        :param n_h: hole concentration (cm-3)
        :param t: time (ns)"""

        T = sci.trapz(k_T * n_e * (N_T - n_t) * (n_h + p_0), t)
        B = sci.trapz(k_B * n_e * (n_h + p_0) * (n_e + n_h + p_0), t)
        D = sci.trapz(k_D * n_t * (n_h + p_0) * n_e, t)
        S = T + B + D
        return {'T': T / S * 100, 'B': B / S * 100, 'D': D / S * 100}

    @staticmethod
    def rate_equations(n_e, n_t, n_h, k_B, k_T, k_D, p_0, N_T, **kwargs):
        """ Rate equations of the BTD model
        :param n_e: electron concentration (cm-3)
        :param n_t: trapped electron concentration (cm-3)
        :param n_h: hole concentration (cm-3)
        :param k_B: bimolecular recombination rate constant (cm3/ns)
        :param k_T: trapping rate constant (cm3/ns)
        :param k_D: detrapping rate constant (cm3/ns)
        :param p_0: doping concentration (cm-3)
        :param N_T: trap states concentration (cm-3) """

        B = k_B * n_e * (n_h + p_0)
        T = k_T * n_e * (N_T - n_t)
        D = k_D * n_t * (n_h + p_0)
        dne_dt = - B - T
        dnt_dt = T - D
        dnh_dt = - B - D
        return {'n_e': dne_dt, 'n_t': dnt_dt, 'n_h': dnh_dt}

    def calculate_trpl(self, t, N_0, I, y_0, p_0, **kwargs):
        """ Calculate the normalised TRPL intensity
        :param t: time (ns)
        :param N_0: initial carrier concentration (cm-3)
        :param y_0: background intensity
        :param p_0: doping concentration (cm-3)
        :param I: amplitude factor
        :param kwargs: keyword arguments passed to the calculate_concentrations function

        Examples
        --------
        >>> _ = BTDModel().calculate_trpl(np.linspace(0, 100), N_0=1e17, p=1, k_B=50e-20, k_T=12000e-20, N_T=60e12,
        ...     p_0=65e12, k_D=80e-20, I=1, y_0=0)
        >>> _ = BTDModel().calculate_concentrations(np.linspace(0, 100), N_0=1e17, k_B=50e-20, k_T=12000e-20, N_T=60e12,
        ...     p_0=65e12, k_D=80e-20, p=1000) """

        n = self.calculate_concentrations(t, N_0, p_0=p_0, **kwargs)
        I_TRPL = n['n_e'][-1] * (n['n_h'][-1] + p_0) / N_0
        signal = I_TRPL / I_TRPL[0] + y_0
        return I * signal / signal[0]

    def get_recommendations(self, contributions, threshold=10):
        """ Get recommendations for the contributions """

        recs = []
        for key in contributions:
            if np.max(contributions[key]) < threshold:
                if 'Bimolecular' in key and self.fvalues.get('k_B', 1) != 0:
                    recs.append(self.get_rec_string('bimolecular', 'higher'))
                elif 'Trapping' in key and self.fvalues.get('k_T', 1) != 0 and self.fvalues.get('N_T', 1) != 0:
                    recs.append(self.get_rec_string('trapping', 'lower'))
                elif 'Detrapping' in key and self.fvalues.get('k_D', 1) != 0:
                    recs.append(self.get_rec_string('detrapping'))
        recs.append('Note: For the bimolecular-trapping-detrapping model, although a low contribution suggests that the'
                    ' parameter associated with the process are not be accurate, a non-negligible contribution does not '
                    'automatically indicate that the parameters retrieved are accurate due to the complex nature of the '
                    'model. It is recommended to perform a grid fitting analysis with this model.')
        return recs


models = {'Bimolecular-Trapping-Auger': BTModel(), 'Bimolecular-Trapping-Detrapping': BTDModel()}


if __name__ == '__main__':
    import doctest
    doctest.testmod()
