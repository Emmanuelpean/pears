""" main script
Important notes:
- Changing the app mode resets all the results
- Changing the period change data associated with the fit
- Changing any other value does not re-run the fit but displays a message about it
- Changing the data loaded resets everything """

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import copy
import pandas as pd

from core import utils
from core import models
from core import plot
from core import resources

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- SET UP -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# General setup & layout
st.set_page_config('Pears', resources.icon_filename, layout='wide')
st.markdown("""<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>""", unsafe_allow_html=True)  # hide main menu and footer
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)  # set all radio buttons in row

# Load the models and store them in the session state
if 'models' not in st.session_state:
    st.session_state.models = copy.deepcopy(models.models)  # copy of the model object for the session
if 'ran' not in st.session_state:
    st.session_state.ran = False  # True if the fit has been previously ran
if 'results' not in st.session_state:
    st.session_state.results = []  # list of all the results
if 'fit_mode' not in st.session_state:
    st.session_state.fit_mode = None  # last fit mode used
if 'data' not in st.session_state:
    st.session_state.data = [[None], [None]]  # input data
if 'period' not in st.session_state:
    st.session_state.period = ''  # excitation repetition period
if 'carrier_accumulation' not in st.session_state:
    st.session_state.carrier_accumulation = None  # carrier accumulation
if 'preprocess' not in st.session_state:
    st.session_state.preprocess = False
if 'file' not in st.session_state:
    st.session_state.file = None


def reset_all():
    """ Reset the stored values """
    print('Resetting stored values')
    st.session_state.results = []
    st.session_state.carrier_accumulation = None
    st.session_state.period = ''
    st.session_state.ran = False


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- INPUT -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

st.markdown("""%s""" % utils.render_image(resources.logo_text_filename, 50), unsafe_allow_html=True)  # main logo
st.sidebar.markdown("""%s""" % utils.render_image(resources.logo_filename, 20), unsafe_allow_html=True)  # sidebar logo
info_message = st.empty()

# -------------------------------------------------------- MODES -------------------------------------------------------

fit_mode = st.sidebar.selectbox('Mode', resources.app_modes, key='fit_mode_')
if st.session_state.fit_mode != fit_mode:  # store the new fit mode and reset the results and ran
    st.session_state.fit_mode = fit_mode
    reset_all()

# -------------------------------------------------------- DATA --------------------------------------------------------

# File uploader
input_filename = st.sidebar.file_uploader('Data file', key='input_filename_', help='Data file. Any text or cvs file.')

# Data format
data_format_help = 'Select the data format of your file between *X/Y1/Y2/Y3...*: first column is the time (in ns) ' \
                   'followed by the TRPL intensity or *X1/Y1/X2/Y2...*: the time and intensity columns are alternated.'
data_format = st.sidebar.radio('Data format', ['X/Y1/Y2/Y3...', 'X1/Y1/X2/Y2...'], help=data_format_help,
                               key='data_format_')

# Data delimiter
data_delimiter_help = 'Data delimiter. tab/space cannot be used if the file has missing data.'
data_delimiter = st.sidebar.radio('Data delimiter', ['tab/space', 'comma', 'semicolon'], help=data_delimiter_help,
                                  key='data_delimiter')
data_delimiter = {'tab/space': None, 'comma': ',', 'semicolon': ';'}[data_delimiter]

# Quantity
quantity_input = st.sidebar.radio('Quantity', ['TRPL', 'TRMC'], help='Quantity associated with the data', key='quantity_input_')

# Processing data
if quantity_input == 'TRPL':
    preprocess_help = 'Shift the decay maximum intensity to $t=0$ and normalise the intensity.'
else:
    preprocess_help = 'Shift the decay maximum intensity to $t=0$.'
process_input = st.sidebar.checkbox('Pre-process data', help=preprocess_help, key='preprocess_')

# Load the data
data_message = st.empty()
xs_data, ys_data = [None], [None]
if input_filename is not None:
    try:
        xs_data, ys_data = utils.load_data(input_filename.getvalue(), data_delimiter, data_format)

        # Check consistency
        if len(xs_data) != len(ys_data):
            data_message.error('The number of X columns is not equal to the number of Y columns.')
            raise AssertionError()
        if any([len(x_data) == 0 for x_data in xs_data]) or any([len(y_data) == 0 for y_data in ys_data]):
            raise AssertionError()

        # Process the data
        if process_input:
            xs_data, ys_data = utils.process_data(xs_data, ys_data, quantity_input)

        data_message.success('Data successfully loaded')

    except (ValueError, IOError, TypeError, AssertionError):  # if an error occurs during the file reading
        xs_data, ys_data = [None], [None]
        data_message.error('Uh-oh! The data could not be loaded')

else:
    data_message.info('Load a data file')

# Reset the results and display if the new data are different from the new ones
if not np.all([np.all(x1 == x2) for x1, x2 in zip(st.session_state.data[0], xs_data)]) \
        and not np.all([np.all(x1 == x2) for x1, x2 in zip(st.session_state.data[1], ys_data)]):
    reset_all()
    st.session_state.data = [xs_data, ys_data]

# ------------------------------------------------------ FLUENCES ------------------------------------------------------

fluence_message = st.empty()
N0s = None
if xs_data[0] is not None:
    N0_help = r'Photoexcited carrier concentration(s separated by commas), calculated as ' \
              r'$\mathtt{N_0=A\frac{I_0}{E_{photon}D}}$ where $\mathtt{I_0}$ is the excitation pulse fluence ' \
              r'(in $\mathtt{J/cm^2}$), $\mathtt{D}$ is the film thickness (in $\mathtt{cm}$), $\mathtt{E_{photon}}$ ' \
              r'is the the photon energy (in $\mathtt{J}$) and $\mathtt{A}$ is the sample absorptance.'
    N0s_input = st.sidebar.text_input('Photoexcited carrier concentrations (cm-3)', help=N0_help, key='N0s_input_')
    try:
        N0s_ = [float(n) for n in N0s_input.split(',')]
        if len(N0s_) != len(ys_data):
            fluence_message.warning('Uh-oh! The number of initial carrier concentrations is different than the number of'
                                    ' TRPL curves from the file (%i)' % len(ys_data))
        else:
            fluence_message.success('The initial carrier concentrations input is valid')
            N0s = N0s_
    except ValueError:
        if N0s_input != '':
            fluence_message.error('Uh-oh! The initial carrier concentrations input is not valid')
        else:
            fluence_message.info('Enter the initial carrier concentrations')

# -------------------------------------------------------- MODEL -------------------------------------------------------

model, model_name = None, ''
if N0s is not None:
    model_help = 'Choose the model to use between the Bimolecular-Trapping-Auger and the Bimolecular-Trapping-Detrapping models.'
    model_name = st.sidebar.selectbox('Model', list(st.session_state.models), help=model_help, key='model_name_')
    model = st.session_state.models[model_name][quantity_input]

# ------------------------------------------------ FIXED & GUESS VALUES ------------------------------------------------

if N0s is not None:
    param_key = st.sidebar.selectbox('Parameters', model.ids, format_func=lambda k: model.labels[k], key='param_key_')

    # -------------------------------------------------- FIXED VALUES --------------------------------------------------

    # Display the fixed value
    fvalue_help = 'Fixed values for the selected parameter. If not a number, use the guess value below for the fitting.'
    fvalue = st.sidebar.text_input('Fixed value', utils.to_scientific(model.fvalues[param_key]), help=fvalue_help,
                                   key=model_name + param_key + 'fixed')

    # Store the fixed value
    try:
        model.fvalues[param_key] = float(fvalue)
    except ValueError:
        model.fvalues[param_key] = None

    # -------------------------------------------------- GUESS VALUES --------------------------------------------------

    # If no fixed value, display and set the guess value
    if model.fvalues[param_key] is None:

        # Fitting mode
        if fit_mode == resources.fitting_mode:

            # Display the guess value
            gvalue_help = 'Parameter initial guess value for the fitting.'
            gvalue = st.sidebar.text_input('Guess value', utils.to_scientific(model.gvalues[param_key]),
                                           help=gvalue_help, key=model_name + param_key + 'guess')

            # Store the guess value
            try:
                model.gvalues[param_key] = float(gvalue)
            except (ValueError, TypeError):
                pass

        # Grid fitting mode
        else:

            # Display the guess values
            gvalue_help = 'Enter multiple initial guess values, separated with commas.'
            gvalues = st.sidebar.text_input('Guess values', utils.to_scientific(model.gvalues_range[param_key]),
                                            help=gvalue_help, key=model_name + param_key + 'guesses')

            # Store the guess values
            try:
                model.gvalues_range[param_key] = [float(v) for v in gvalues.split(', ')]
            except (ValueError, TypeError):
                pass

# ------------------------------------------------- REPETITION PERIOD --------------------------------------------------

period = ''
if N0s is not None:
    period_help = 'Excitation repetition period. Used to calculate possible carrier accumulation between consecutive ' \
                  'excitation pulses.'
    period = st.sidebar.text_input('Excitation repetition period (ns)', help=period_help, key='period_input')
    try:
        period = float(period)
    except ValueError:
        period = ''

# ----------------------------------------------------- RUN BUTTON -----------------------------------------------------

run_button = False  # when pressed, run the fit
if N0s is not None:
    run_button = st.sidebar.button('Run')
if run_button:  # change the state of ran
    st.session_state.ran = True

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- RESULTS DISPLAY --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

results_container = st.empty()

if st.session_state.ran:  # display the results if the run button has been previously pressed and the fitting mode hasn't been changed

    # Remove the data and fluence messages
    data_message.empty()
    fluence_message.empty()

    # -------------------------------------------------- FITTING MODE --------------------------------------------------

    if run_button:
        info_message.info('Processing')
        if fit_mode == resources.fitting_mode:
            try:
                print('Calling fitting')
                variable = model.fit(xs_data, ys_data, N0s)
                st.session_state.results = copy.deepcopy([variable, model, N0s])
            except ValueError:
                info_message.error('Uh Oh, could not fit the data. Try changing the parameter guess values.')
                raise AssertionError('Fit failed')
        else:
            progressbar = st.sidebar.progress(0)
            st.session_state.period = ''  # force reset
            variable = model.grid_fitting(progressbar, N0s, xs_data=xs_data, ys_data=ys_data)
            st.session_state.results = copy.deepcopy([variable, model, N0s])
        info_message.empty()
    else:
        if st.session_state.results[1] != model or st.session_state.results[2] != N0s:
            info_message.warning('You have changed some of the input settings. Press "run" to apply the changes')
        variable, model, N0s = st.session_state.results

    with results_container.container():

        if fit_mode == resources.fitting_mode:
            fit = variable
        else:
            st.subheader('Parallel plot')

            # Calculate the carrier accumulation
            CA = None
            if period:
                if period != st.session_state.period:  # if period is changed
                    print('Calculating CA')
                    CA = [model.get_carrier_accumulation(fit['popts'], fit['N0s'], period) for fit in variable]
                    st.session_state.carrier_accumulation = CA
                    st.session_state.period = period  # store the period used to calculate the CA
                else:
                    print('Getting stored CA')
                    CA = st.session_state.carrier_accumulation
            else:  # reset the store CA and period
                st.session_state.carrier_accumulation = None
                st.session_state.period = period

            # Add the CA to the values
            popts = []
            for i in range(len(variable)):
                popt = variable[i]['values'].copy()
                if CA is not None:
                    popt['Max. CA (%)'] = np.max(list(CA[i].values()))
                popts.append(popt)

            # Display the parallel plot
            xp = plot.parallel_plot(popts, variable[0]['no_disp_keys'], [k for k in variable[0]['values'].keys() if k not in variable[0]['no_disp_keys']])
            selected = xp.to_streamlit('hip', 'selected_uids').display()
            fit = variable[int(selected[0])]
            st.subheader('Displaying results of fit #%i' % (int(selected[0]) + 1))

        # --------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------ DISPLAYT FIT  -----------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        col1, col2 = st.columns(2)

        # -------------------------------------------- FITTING PLOT & EXPORT -------------------------------------------

        col1.markdown("""#### TRPL fitting""")

        # Plot
        col1.plotly_chart(plot.plot_fit(fit['xs_data'], fit['ys_data'], quantity_input, fit['fit_ydata'], fit['N0s_labels']), use_container_width=True)

        # Export
        header = np.concatenate([['Time (ns)', 'Intensity %i' % i] for i in range(1, len(ys_data) + 1)])
        export_data = utils.matrix_to_string([val for pair in zip(fit['xs_data'], fit['fit_ydata']) for val in pair], header)
        col1.download_button('Download data', export_data, 'pears_fit_data.csv')

        # ---------------------------------------- OPTIMISED PARAMETERS DISPLAY ----------------------------------------

        col1.markdown("""#### Parameters""")
        for label in fit['labels']:
            col1.markdown(label, unsafe_allow_html=True)

        # ----------------------------------------------- CONTRIBUTIONS ------------------------------------------------

        st.markdown("""#### Contributions""")
        contributions = pd.DataFrame(fit['contributions'], index=fit['N0s_labels']).transpose()
        st.markdown(contributions.to_html(escape=False) + '<br>', unsafe_allow_html=True)

        # Analysis
        for s in model.get_recommendations(fit['contributions']):
            st.warning(s)

        # -------------------------------------------- CARRIER ACCUMULATION  -------------------------------------------

        if period:
            st.markdown("""#### Carrier accumulation""")
            nca = model.get_carrier_accumulation(fit['popts'], fit['N0s'], period)
            nca_df = pd.DataFrame(nca, index=['Carrier accumulation (%)'])
            st.markdown(nca_df.to_html(escape=False) + '<br>', unsafe_allow_html=True)

            # Analysis
            max_nca = np.max(list(nca.values()))
            if max_nca > 5.:
                st.warning('This fit predicts significant carrier accumulation leading to a maximum %f %% difference '
                           'between the single pulse and multiple pulse TRPL decays. You might need to increase your '
                           'excitation repetition period to prevent potential carrier accumulation.' % max_nca)
            else:
                st.success('This fit does not predict significant carrier accumulation.')

        # -------------------------------------------- CONCENTRATIONS PLOT ---------------------------------------------

        col2.markdown("""#### Carrier concentrations""")
        concentrations = model.get_carrier_concentrations(fit['xs_data'], fit['popts'], period)
        conc_fig = plot.plot_carrier_concentrations(concentrations[0], concentrations[2], fit['N0s'], fit['N0s_labels'],
                                                    concentrations[1], model)
        col2.plotly_chart(conc_fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- DATA DISPLAY ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

elif xs_data[0] is not None:

    with results_container.container():
        st.markdown("""#### Input data""")
        if N0s is not None:
            labels = utils.get_power_labels(N0s)
        else:
            labels = ['%i' % (i + 1) for i in range(len(ys_data))]
        st.plotly_chart(plot.plot_fit(xs_data, ys_data, quantity_input, labels=labels), use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- GENERAL INFORMATION ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------- APP DESCRIPTION --------------------------------------------------

with st.expander('About', xs_data[0] is None):
    st.info("""*Pears* is a web app to easily fit time-resolved photoluminescence (TRPL) and time-resolved microwave 
photoconductivity (TRMC) data of perovskite materials. Two models can be used, which are extensively discussed for TRPL 
[here](https://doi.org/10.1039/D0CP04950F).
- The Bimolecular-Trapping-Auger model considers assumes no doping and that the trap states remain mostly empty over time
- The Bimolecular-Trapping-Detrapping model considers bimolecular recombination, trapping and detrapping with the presence of doping.\n
Two modes are available.
- The "%s" mode can be used to fit experimental data given a set of guess parameters.
- The "%s" mode runs the fitting optimisation for a range of guess parameters. If all the optimisations do not converge toward the same values,
  then the fitting is inaccurate due to the possibility of multiple solutions.x
App created and maintained by [Emmanuel V. Pean](mailto:emmanuelpean.dev@gmail.com) ([Twitter](https://twitter.com/emmanuel_pean)).  
Version 0.4 (last updated: 9th May 2022).  
Source code: https://github.com/Emmanuelpean/pears""" % (resources.fitting_mode, resources.analysis_mode))

# -------------------------------------------------- MODEL DESCRIPTION -------------------------------------------------

with st.expander('Model & computational details'):
    st.markdown("""The following information can be found with more details [here](https://doi.org/10.1039/D0CP04950F)
    (Note that for simplicity purpose, the $\Delta$ notation for the photoexcited carriers was dropped here for 
    simplicity purposes *e.g.* $\Delta n_e$ in the paper is $n_e$ here).""")
    st.markdown("""#### Models""")
    st.markdown("""Two charge carrier recombination models can be used whose rate equations for the different carrier 
    concentrations are given below.""")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center; '>Bimolecular-trapping model</h3>", unsafe_allow_html=True)
        st.markdown("""%s""" % utils.render_image(resources.btmodel_filename, 65), unsafe_allow_html=True)
        st.latex(r'n_e(t)=n_h(t)=n(t)')
        st.markdown('#')
        st.markdown('#####')
        st.latex(r'\frac{dn}{dt}=-k_Tn-k_Bn^2-k_An^3,\ \ \ n^p(t=0)=n^{p-1}(T)+N_0')
        st.markdown('#')
        st.markdown('#####')
        st.latex(r'I_{TRPL} \propto n^2')
        st.latex(r'I_{TRMC}=2\mu n / N_0')
        st.markdown("""where:
* $n$ is the photoexcited carrier concentration (in $cm^{-3}$)
* $k_T$ is the trapping rate constant (in $ns^{-1}$)
* $k_B$ is the bimolecular recombination rate constant (in $cm^3/ns$)
* $k_A$ is the Auger recombination rate constant (in $cm^6/ns$)
* $\mu$ is the carrier mobility (in $cm^2/(Vs)$)""")
    with col2:
        st.markdown("<h3 style='text-align: center; '>Bimolecular-trapping-detrapping model</h3>",
                    unsafe_allow_html=True)
        st.markdown("""%s""" % utils.render_image(resources.btdmodel_filename, 65), unsafe_allow_html=True)
        st.latex(r'\frac{dn_e}{dt}=-k_B n_e (n_h+p_0 )-k_T n_e [N_T-n_t ],\ \ \ n_e^p(t=0)=n_e^{p-1}(T)+N_0')
        st.latex(r'\frac{dn_t}{dt}=k_T n_e [N_T-n_t]-k_D n_t (n_h+p_0 ),\ \ \ n_t^p(t=0)=n_t^{p-1}(T)')
        st.latex(r'\frac{dn_h}{dt}=-k_B n_e (n_h+p_0 )-k_D n_t (n_h+p_0 ),\ \ \ n_h^p(t=0)=n_h^{p-1}(T)+N_0')
        st.latex(r'I_{TRPL} \propto n_e(n_h+p_0)')
        st.latex(r'I_{TRMC} = \left(\mu_e n_e + \mu_h n_h \right)/N_0')
        st.markdown("""where:
* $n_e$ is the photoexcited electron concentration (in $cm^{-3}$)
* $n_h$ is the photoexcited hole concentration (in $cm^{-3}$)
* $n_t$ is the trapped electron concentration (in $cm^{-3}$)
* $k_B$ is the bimolecular recombination rate constant (in $cm^3/ns$)
* $k_T$ is the trapping rate constant (in $cm^3/ns$)
* $k_D$ is the detrapping rate constant (in $cm^3/ns$)
* $N_T$ is the trap state concentration (in $cm^{-3}$)
* $p_0$ is the dark hole concentration (in $cm^{-3}$)
* $\mu_e$ is the electron mobility (in $cm^2/(Vs)$)
* $\mu_h$ is the hole mobility (in $cm^2/(Vs)$)""")
    st.markdown("""####""")
    st.markdown("""For both models, the photoexcited charge carrier concentration $N_0$ is the concentration of carriers
    excited by a single excitation pulse. The initial condition of a carrier concentration after excitation pulse $p$ is 
    given by the sum of any remaining carriers $n_X^{p-1}(T)$ ($T$ is the excitation repetition period) just before 
    excitation plus the concentration of carrier generated by the pulse $N_0$ (except for the trapped electrons).""")

    st.markdown("""#### Fitting""")
    st.markdown(r"""Fitting is carried using the least square optimisation. For a dataset containing $M$ curves, 
    each containing $N_i$ data points, the residue $SS_{res}$ is:""")
    st.latex(r"""SS_{res}=\sum_i^M\sum_j^{N_i}\left(y_{i,j}-F(t_{i,j},A_{i})\right)^2""")
    st.markdown("""where $y_{i,j}$ is the intensity associated with time $t_{i,j}$ of point $j$ of curve $i$.
    $A_i$ are the model parameters associated with curve $i$ and $F$ is the fitting model given by for the TRPL and TRMC respectively:""")
    st.latex(r'F(t,I_0, y_0, k_B,...)=I_0 \frac{I_{TRPL}(t,k_B,...)}{I_{TRPL}(0, k_B,...)} + y_0')
    st.latex(r'F(t, y_0, k_B,...)=I_{TRMC}(t,k_B,...) + y_0')
    st.markdown(""" where $I_0$ is an intensity factor and $y_0$ is an intensity offset. Contrary to the other parameters of 
    the models (e.g. $k_B$), $I_0$ and $y_0$ are not kept the same between the different TRPL curves *i.e.* the fitting 
    models for curves $A$, $B$,... are:""")
    st.latex(r'F_A(t,I_0^A, y_0^A, k_B,...)=I_0^A \frac{I_{TRPL}(t,k_B,...)}{I_{TRPL}(0, k_B,...)} + y_0^A')
    st.latex(r'F_B(t,I_0^B, y_0^B, k_B,...)=I_0^B \frac{I_{TRPL}(t,k_B,...)}{I_{TRPL}(0, k_B,...)} + y_0^B')
    st.latex('...')
    st.markdown("""By default, $I_0$ and $y_0$ are respectively fixed at 1 and 0 (assuming no background noise and 
    normalised intensity. The quality of the fit is estimated from the coefficient of determination $R^2$:""")
    st.latex(r'R^2=1-\frac{SS_{res}}{SS_{total}}')
    st.markdown(r"""where $SS_{total}$ is defined as the sum of the squared difference between each point and the 
    average of all curves $\bar{y}$:""")
    st.latex(r"""SS_{total}=\sum_i^M\sum_j^{N_i}\left(y_{i,j}-\bar{y}\right)^2""")
    st.markdown("""For fitting, it is assumed that there is no carrier accumulation between excitation pulses due to the
    presence of non-recombined carriers from previous excitation pulses. This requires the TRPL decays to be measured
    with long enough excitation repetition periods such that all carriers can recombine.""")

    st.markdown("""#### Carrier accumulation""")
    st.markdown("""It is possible to  calculate the expected effect of carrier accumulation on the TRPL and TRMC from the 
    parameters retrieved from the fits if the repetition period is provided. The carrier accumulation ($CA$) is 
    calculated as the maximum difference between the simulated TRPL/TRMC after the first ($p=1$) and stabilised ($p=s$) pulses:""")
    st.latex(r'{CA}_{TRPL}=\max\left({\frac{I_{TRPL}^{p=1}(t)}{I_{TRPL}^{p=1}(0)}-\frac{I_{TRPL}^{p=s}(t)}{I_{TRPL}^{p=s}(0)}})\right)')
    st.latex(r'{CA}_{TRMC}=\max\left(I_{TRMC}^{p=1}(t)-I_{TRMC}^{p=s}(t))\right)')
    st.markdown("""The stabilised pulse is defined as when the electron and hole concentrations vary by less than 
    10<sup>-3</sup> % of the photoexcited concentration between two consecutive pulses:""", unsafe_allow_html=True)
    st.latex(r'|n_e^p(t)-n_e^{p+1}(t)|<10^{-5} N_0')
    st.latex(r'|n_h^p(t)-n_h^{p+1}(t)|<10^{-5} N_0')

    st.markdown("""#### Contributions""")
    st.markdown("""The contribution of each process to the TRPL intensity variations over time is calculated from the 
    fitted values (see Equations 22 to 25 [here](https://doi.org/10.1039/D0CP04950F)). It is important to ensure that the 
    contribution of a process (e.g., trapping) is non-negligible so that the associated parameters (e.g., $k_T$ 
    in the case of the bimolecular-trapping model) are accurately retrieved.""", unsafe_allow_html=True)

    st.markdown("""#### Grid fitting""")
    st.markdown("""This mode runs the fitting process for a grid of guess values. The grid is generated from every 
    possible combinations of guess values supplied (e.g. $k_B$: 10<sup>-20</sup>, 10<sup>-19</sup> and 
    $k_T$: 10<sup>-3</sup>, 10<sup>-2</sup> yields 4 sets of guess values: (10<sup>-20</sup>, 10<sup>-3</sup>),
    (10<sup>-20</sup>, 10<sup>-2</sup>), (10<sup>-19</sup>, 10<sup>-3</sup>) and (10<sup>-19</sup>, 10<sup>-2</sup>)). 
    Note that in the case of the bimolecular-trapping-detrapping model, only set of guess values satisfying $k_T>k_B$ 
    and $k_T>k_D$ are considered to keep the computational time reasonable. Fitting is then carried using each set of 
    guess values as schematically represented below: %s
    In the case where all the optimisations converge towards a similar solution, it can be assumed that only 1 solution
    exist and that therefore the parameter values obtained accurately describe the system measured. However, if the fits 
    converge toward multiple solutions, it is not possible to ascertain which solution represents the system accurately."""
                % utils.render_image(resources.opt_guess, 60, 'png'), unsafe_allow_html=True)

# ------------------------------------------------------- HOW TO -------------------------------------------------------

with st.expander('Getting started'):
    st.markdown("""#### Example""")
    data1_link = utils.generate_downloadlink(resources.BT_TRPL_np, text='TRPL data set 1')
    data2_link = utils.generate_downloadlink(resources.BTD_TRPL_p, text='TRPL data set 2')
    data3_link = utils.generate_downloadlink(resources.BTD_TRMC, text='TRMC data set 3')
    st.markdown("""Follow these steps to fit TRPL decays.""")
    st.markdown("""1. Upload your data and select the data format (tab/space here). Select whether TRPL or TRMC is analysed.
    Check the "Pre-process data" box for PEARS to shift the decay(s) and normalise it (the later is done only for TRPL data);
    * _e.g._ %s, 
    * _e.g._ %s
    * _e.g._ %s (select X1/Y1/X2/Y2... data format)""" % (data1_link, data2_link, data3_link), unsafe_allow_html=True)

    st.markdown("""2. Enter the photoexcited carrier concentrations (in $cm^{-3}$) for each TRPL/TRMC decay 
    measured (separated by a comma);
    * _e.g._: 5.58e15, 1.00e16, 2.23e16 (data set 1)
    * _e.g._: 55e12, 164e12, 511e12, 1720e12, 4750e12 (data set 2 & 3)""", unsafe_allow_html=True)

    st.markdown("""3. Choose a fitting model;""")

    st.markdown("""4. (Optional) Set the value of known parameters;""")

    if fit_mode == 'Fitting':
        st.markdown("""5. (Optional) For non fixed parameters, choose the guess value for the optimisation;""")
    else:
        st.markdown("""5. (Optional) For non fixed parameters, choose multiple guess values for each parameter;""")

    st.markdown("""6. (Optional) Enter the excitation repetition period (in ns). If provided, *Pears* calculates the 
    carrier accumulation effect on the TRPL intensity as well as show the evolution of the carrier concentrations
    after multiple consecutive excitation pulses until stabilisation.""")

    st.markdown("""7. Press "run" """)

    st.markdown("""#### Video tutorial""")
    st.video(resources.tutorial_video)

# ------------------------------------------------------ CHANGELOG -----------------------------------------------------

with st.expander('Changelog'):
    st.markdown("""
    #### Date - V 0.4 - Major update
    * Pears can now fit time-resolved microwave photoconductivity data 🎉
    * Both recombination models have been modified to account for the carrier mobility
    * A new "Quantity" input has been added to the sidebar to select the type of data to fit
    #### May 2022 - V 0.3.1.2
    * The Auger rate constant is now fixed to 0 by default.
    * Fixed a bug during which data could not be successfully loaded
    #### April 2022 - V 0.3.1.1
    * Fixed some bugs that were introduced in the previous update
    #### March 2022 - V 0.3.1
    * Added new "Pre-process" option
    * Files with a header can now be uploaded 
    #### November 2021 - V 0.3.0
    * Added intensity parameter to the fitting procedure
    * Added new file input format (X1/Y1/X2/Y2)
    * Updated app layout
    * Added calculation of the process contributions
    * Added calculation of the carrier accumulation effect on the TRPL
    * Added help bubbles
    * Fixed a pesky bug that prevented from changing the parameter fixed values
    * Changed the model descriptions
    * Added new parallel plot visualisation for grid fitting analysis
    * Fits and Grid fit analysis now stay on screen until the run button is pressed again (unless the mode or the data 
    are changed)
    * Added disclaimer
    * Added website icon
    * Added video tutorial
    * Now accept a wider range of data""")

# ----------------------------------------------------- DISCLAIMER -----------------------------------------------------

with st.expander('Disclaimer'):
    st.markdown("""THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,  INCLUDING BUT 
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. 
    IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER 
    LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
    USE OR OTHER DEALINGS IN THE SOFTWARE.""")

# ------------------------------------------------------ ANALYTICS -----------------------------------------------------

components.html("""<script async defer data-website-id="62a61960-56c2-493b-90e0-20e6796ecfa4" 
src="https://pears-tracking.herokuapp.com/umami.js"></script>""")
