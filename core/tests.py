""" test module """

import numpy as np
import plotly.io as pio
from core import models
from core import fitting
from core import utils
from core import plot
from core import resources

pio.renderers.default = "browser"


xdata, ydata = resources.test_file2[0], resources.test_file2[1:]
xdata = [xdata] * len(ydata)
N0s = (55e12, 164e12, 511e12, 1720e12, 4750e12)
fp = [dict(N_0=n) for n in N0s]
labels = ['N<sub>0</sub> = ' + utils.get_power_text(N0, -1) + ' cm<sup>-3</sup>' for N0 in N0s]
p0_bt = dict(k_B=1e-20, k_T=0.01, k_A=1e-40, y_0=0., I=1)
p0_btd = dict(k_B=10e-20, k_T=1000e-20, k_D=100e-20, p_0=1e12, N_T=1e12, y_0=0., I=1)
bt_model = models.BTModel()
btd_model = models.BTDModel()

# ------------------------------------------------------ FITTING -------------------------------------------------------

fps2 = [utils.merge_dicts(f, {'k_A': 0.}) for f in fp]

fit1 = fitting.Fit(xdata, ydata, bt_model.calculate_trpl, p0_bt, ['y_0', 'I'], fp, verbose=2)
popts1 = fit1.fit()
assert popts1[0] == {'k_B': 5.477158939124385e-20, 'k_T': 0.0001413771107581223, 'k_A': 7.544595504812444e-35,
                     'y_0': 1.8301219117749311e-28, 'I': 0.1587170375190151, 'N_0': 55000000000000.0}

fit2 = fitting.Fit(xdata, ydata, bt_model.calculate_trpl, p0_bt, ['y_0', 'I'], fps2, verbose=2)
popts2 = fit2.fit()
assert popts2[0] == {'k_B': 1.8284927043784638e-19, 'k_T': 0.0001229386881863971, 'y_0': 1.4961948411223975e-19,
                     'I': 0.14586490670784974, 'N_0': 55000000000000.0, 'k_A': 0.0}

fit3 = fitting.Fit(xdata, ydata, btd_model.calculate_trpl, p0_btd, ['y_0', 'I'], fp, verbose=2)
popts3 = fit3.fit()
assert popts3[0] == {'k_B': 2.594833719093791e-19, 'k_T': 1.2272907649841054e-16, 'k_D': 8.348401220327214e-19,  'p_0': 59927506081336.54,
                     'N_T': 59862510166661.47, 'y_0': 1.0154703623609344e-29, 'I': 1.0127702206081532, 'N_0': 55000000000000.0}


# ---------------------------------------------------- FITTING PLOT ----------------------------------------------------

plot.plot_fit([xdata] * len(ydata), ydata, fit1.calculate_fits(popts1[0]), labels).show()
plot.plot_fit([xdata] * len(ydata), ydata, fit3.calculate_fits(popts3[0]), labels).show()


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
analysis1 = fitting.run_grid_fit(p0s1, ffp1, bt_model.param_filters, xs_data=xdata, ys_data=ydata, function=bt_model.calculate_trpl, detached_parameters=['I'])


p0s3 = dict(k_B=[1e-20, 1e-19, 1e-18], k_D=[1e-20, 1e-19, 1e-18], N_T=[1e12], p_0=[1e12], y_0=[0.], I=[1.])
ffp3 = [utils.merge_dicts(f, {'k_T': 5e-19}) for f in fp]
analysis3 = fitting.run_grid_fit(p0s3, ffp3, btd_model.param_filters, xs_data=xdata, ys_data=ydata, function=btd_model.calculate_trpl, detached_parameters=['I'])
