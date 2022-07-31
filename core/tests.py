""" test module """

import numpy as np
import plotly.io as pio
from core import models
from core import fitting
from core import utils
from core import plot
from core import resources

pio.renderers.default = "browser"


xdata, ydata = resources.BTD_TRPL_p[0], resources.BTD_TRPL_p[1:]
xdata = [xdata] * len(ydata)
N0s = (55e12, 164e12, 511e12, 1720e12, 4750e12)
fp = [dict(N_0=n) for n in N0s]
labels = ['N<sub>0</sub> = ' + utils.get_power_text(N0, -1) + ' cm<sup>-3</sup>' for N0 in N0s]
p0_bt = dict(k_B=1e-20, k_T=0.01, k_A=1e-40, y_0=0., I=1)
p0_btd = dict(k_B=10e-20, k_T=1000e-20, k_D=100e-20, p_0=1e12, N_T=1e12, y_0=0., I=1)
bt_model = models.BTModelTRPL()
btd_model = models.BTDModelTRPL()

# ------------------------------------------------------ FITTING -------------------------------------------------------

fps2 = [utils.merge_dicts(f, {'k_A': 0.}) for f in fp]

fit1 = fitting.Fit(xdata, ydata, bt_model.calculate_fit_quantity, p0_bt, ['y_0', 'I'], fp, verbose=2)
popts1 = fit1.fit()
assert {key: '%.3E' % value for key, value in popts1[0].items()} == \
       {'k_B': '5.477E-20', 'k_T': '1.414E-04', 'k_A': '7.545E-35', 'y_0': '6.109E-29', 'I': '1.587E-01', 'N_0': '5.500E+13'}

fit2 = fitting.Fit(xdata, ydata, bt_model.calculate_fit_quantity, p0_bt, ['y_0', 'I'], fps2, verbose=2)
popts2 = fit2.fit()
assert {key: '%.3E' % value for key, value in popts2[0].items()} == \
       {'k_B': '1.828E-19', 'k_T': '1.229E-04', 'y_0': '1.498E-19', 'I': '1.459E-01', 'N_0': '5.500E+13', 'k_A': '0.000E+00'}

fit3 = fitting.Fit(xdata, ydata, btd_model.calculate_fit_quantity, p0_btd, ['y_0', 'I'], fp, verbose=2)
popts3 = fit3.fit()
assert {key: '%.3E' % value for key, value in popts3[0].items()} == \
       {'k_B': '2.594E-19', 'k_T': '1.228E-16', 'k_D': '8.357E-19', 'p_0': '5.978E+13', 'N_T': '5.986E+13', 'y_0': '1.739E-25',
        'I': '1.013E+00', 'N_0': '5.500E+13'}

# ---------------------------------------------------- FITTING PLOT ----------------------------------------------------


plot.plot_fit(xdata, ydata, "TRPL", fit1.calculate_fits(popts1), labels).show()
plot.plot_fit(xdata, ydata, "TRPL", fit3.calculate_fits(popts3), labels).show()


# ------------------------------------------------- CONCENTRATIONS PLOT ------------------------------------------------


concentrations1 = []
for popt in popts1:
    concentration1 = bt_model.calculate_concentrations(xdata[0], **{key: popt[key] for key in popt if key not in ('y_0', 'I')})
    concentrations1.append({key: np.concatenate(concentration1[key]) for key in concentration1})

plot.plot_carrier_concentrations(xdata, concentrations1, N0s, labels, 'Time (ns)', bt_model).show()


concentrations3 = []
for popt in popts3:
    concentration3 = btd_model.calculate_concentrations(xdata[0], **{key: popt[key] for key in popt if key not in ('y_0', 'I')})
    concentrations3.append({key: np.concatenate(concentration3[key]) for key in concentration3})

plot.plot_carrier_concentrations(xdata, concentrations3, N0s, labels, 'Time (ns)', btd_model).show()


# -------------------------------------------------- FITTING ANALYSIS --------------------------------------------------

p0s1 = dict(k_B=[1e-20, 1e-19, 1e-18], k_T=[0.01, 0.001], I=[1.], y_0=['0'])
ffp1 = [utils.merge_dicts(f, {'k_A': 0., 'y_0': 0.}) for f in fp]
analysis1 = fitting.run_grid_fit(p0s1, ffp1, bt_model.param_filters, xs_data=xdata, ys_data=ydata, function=bt_model.calculate_fit_quantity, detached_parameters=['I'])


p0s3 = dict(k_B=[1e-20, 1e-19, 1e-18], k_D=[1e-20, 1e-19, 1e-18], N_T=[1e12], p_0=[1e12], y_0=[0.], I=[1.])
ffp3 = [utils.merge_dicts(f, {'k_T': 5e-19}) for f in fp]
analysis3 = fitting.run_grid_fit(p0s3, ffp3, btd_model.param_filters, xs_data=xdata, ys_data=ydata, function=btd_model.calculate_fit_quantity, detached_parameters=['I'])
