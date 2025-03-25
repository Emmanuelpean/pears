"""main script
Important notes:
- Changing the core mode resets all the results
- Changing the period change data associated with the fit
- Changing any other value does not re-run the fit but displays a message about it
- Changing the data loaded resets everything"""

import copy

import numpy as np
import pandas as pd
import streamlit as st

import models
import plot
import resources
import utils

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- SET UP -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# General setup & layout
st.set_page_config("Pears", resources.ICON_PATH, layout="wide")

# Load the models and store them in the session state
if "models" not in st.session_state:
    st.session_state.models = copy.deepcopy(models.models)  # copy of the model object for the session
if "ran" not in st.session_state:
    st.session_state.ran = False  # True if the fit has been previously ran
if "results" not in st.session_state:
    st.session_state.results = []  # list of all the results
if "fit_mode" not in st.session_state:
    st.session_state.fit_mode = None  # last fit mode used
if "data" not in st.session_state:
    st.session_state.data = [[None], [None]]  # input data
if "period" not in st.session_state:
    st.session_state.period = ""  # excitation repetition period
if "carrier_accumulation" not in st.session_state:
    st.session_state.carrier_accumulation = None  # carrier accumulation
if "preprocess" not in st.session_state:
    st.session_state.preprocess = False
if "file" not in st.session_state:
    st.session_state.file = None


def reset_all():
    """Reset the stored values"""
    print("Resetting stored values")
    st.session_state.results = []
    st.session_state.carrier_accumulation = None
    st.session_state.period = ""
    st.session_state.ran = False


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- INPUT -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

logo_placeholder = st.empty()
st.sidebar.markdown(utils.render_image(resources.LOGO_PATH, 20), unsafe_allow_html=True)  # sidebar logo
info_message = st.empty()

# -------------------------------------------------------- MODES -------------------------------------------------------

fit_mode = st.sidebar.selectbox("Mode", resources.APP_MODES, key="fit_mode_")
if st.session_state.fit_mode != fit_mode:  # store the new fit mode and reset the results and ran
    st.session_state.fit_mode = fit_mode
    reset_all()

# -------------------------------------------------------- DATA --------------------------------------------------------

# File uploader
input_filename = st.sidebar.file_uploader(
    label="Data file",
    key="input_filename_",
)

# Data format
data_format_help = """Select the data format of your file between: 
* *X/Y1/Y2/Y3...*: first column is the time (in ns) followed by the signal intensity or 
* *X1/Y1/X2/Y2...*: the time and intensity columns are alternated."""
data_format = st.sidebar.radio(
    label="Data format",
    options=["X/Y1/Y2/Y3...", "X1/Y1/X2/Y2..."],
    help=data_format_help,
    key="data_format_",
    horizontal=True,
)

# Data delimiter
data_delimiter_help = """Data delimiter. tab/space cannot be used if the file has missing data."""
data_delimiter = st.sidebar.radio(
    label="Data delimiter",
    options=[None, ",", ";"],
    help=data_delimiter_help,
    key="data_delimiter",
    horizontal=True,
    format_func=lambda v: {None: "tab/space", ",": ",", ";": ";"}[v],
)

# Quantity
quantity_input = st.sidebar.radio(
    label="Quantity",
    options=["TRPL", "TRMC"],
    help="Quantity associated with the data",
    key="quantity_input_",
    horizontal=True,
)

# Processing data
if quantity_input == "TRPL":
    preprocess_help = """Shift the decay maximum intensity to $t=0$ and normalise the intensity."""
else:
    preprocess_help = """Shift the decay maximum intensity to $t=0$."""
process_input = st.sidebar.checkbox(
    label="Pre-process data",
    help=preprocess_help,
    key="preprocess_",
)

# Load the data
data_message = st.empty()
xs_data, ys_data = [None], [None]
if input_filename is not None:
    try:
        xs_data, ys_data = utils.load_data(input_filename.getvalue(), data_delimiter, data_format)

        # Check consistency
        if len(xs_data) != len(ys_data):
            data_message.error("The number of X columns is not equal to the number of Y columns.")
            raise AssertionError()
        if any([len(x_data) == 0 for x_data in xs_data]) or any([len(y_data) == 0 for y_data in ys_data]):
            raise AssertionError()

        # Process the data
        if process_input:
            xs_data, ys_data = utils.process_data(xs_data, ys_data, quantity_input == "TRPL")

        data_message.success("Data successfully loaded")

    except:  # if an error occurs during the file reading
        xs_data, ys_data = [None], [None]
        data_message.error("Uh-oh! The data could not be loaded")

else:
    data_message.info("Load a data file")

# Reset the results and display if the new data are different from the new ones
diff_cond1 = not utils.are_lists_identical(xs_data, st.session_state.data[0])
diff_cond2 = not utils.are_lists_identical(ys_data, st.session_state.data[1])
if diff_cond1 or diff_cond2:
    reset_all()
    st.session_state.data = [xs_data, ys_data]

if st.session_state.data[0] == [None]:
    logo_placeholder.markdown(utils.render_image(resources.LOGO_TEXT_PATH, 50), unsafe_allow_html=True)  # main logo

# ------------------------------------------------------ FLUENCES ------------------------------------------------------

fluence_message = st.empty()
N0s = None
if xs_data[0] is not None:
    n0_help = r"""Photoexcited carrier concentration(s separated by commas), calculated as 
    $\mathtt{N_0=A\frac{I_0}{E_{photon}D}}$ where $\mathtt{I_0}$ is the excitation pulse fluence 
    (in $\mathtt{J/cm^2}$), $\mathtt{D}$ is the film thickness (in $\mathtt{cm}$), $\mathtt{E_{photon}}$ 
    is the the photon energy (in $\mathtt{J}$) and $\mathtt{A}$ is the sample absorptance."""
    st.sidebar.markdown("Photoexcited carrier concentrations (cm-3)", help=n0_help)
    N0_inputs = []
    for i in range(len(xs_data)):
        columns = st.sidebar.columns([1, 2], vertical_alignment="center")
        columns[0].markdown(f"Decay {i + 1}")
        N0_inputs.append(columns[1].text_input("label", label_visibility="collapsed", key=f"fluence_{i}"))

    # Check validity of N0s
    if all(N0_inputs):
        try:
            N0s = [float(n) for n in N0_inputs]
        except ValueError:
            fluence_message.error("Uh-oh! The initial carrier concentrations input is not valid")

# --------------------------------------------------- MODEL SELECTION --------------------------------------------------

model, model_name = None, ""
if N0s is not None:
    model_help = """Choose the model to use between the Bimolecular-Trapping-Auger and the Bimolecular-Trapping-
    Detrapping models."""
    model_name = st.sidebar.selectbox(
        "Model",
        list(st.session_state.models),
        help=model_help,
        key="model_name_",
    )
    model = st.session_state.models[model_name][quantity_input]

# ------------------------------------------------ FIXED & GUESS VALUES ------------------------------------------------

if N0s is not None:
    param_key = st.sidebar.selectbox(
        "Parameters",
        model.param_ids,
        format_func=lambda k: model.labels[k],
        key="param_key_",
    )

    # -------------------------------------------------- FIXED VALUES --------------------------------------------------

    key = model_name + param_key + "fixed"

    if key not in st.session_state:
        st.session_state[key] = utils.to_scientific(model.fvalues[param_key])

    # Display the fixed value
    fvalue_help = "Fixed values for the selected parameter. If not a number, use the guess value below for the fitting."
    fvalue = st.sidebar.text_input(
        "Fixed value",
        help=fvalue_help,
        key=key,
    )

    # Store the new value
    try:
        model.fvalues[param_key] = float(fvalue)
    except:
        model.fvalues[param_key] = None

    # -------------------------------------------------- GUESS VALUES --------------------------------------------------

    # If no fixed value, display and set the guess value
    if model.fvalues[param_key] is None:

        # Fitting mode
        if fit_mode == resources.FITTING_MODE:

            key = model_name + param_key + "guess"
            if key not in st.session_state:
                st.session_state[key] = utils.to_scientific(model.gvalues[param_key])

            # Display the guess value
            gvalue_help = "Parameter initial guess value for the fitting."
            gvalue = st.sidebar.text_input(
                "Guess value",
                help=gvalue_help,
                key=model_name + param_key + "guess",
            )

            # Store the guess value
            try:
                model.gvalues[param_key] = float(gvalue)
            except (ValueError, TypeError):
                pass

        # Grid fitting mode
        else:

            key = model_name + param_key + "guesses"
            if key not in st.session_state:
                st.session_state[key] = utils.to_scientific(model.gvalues_rangev[param_key])

            # Display the guess values
            gvalue_help = "Enter multiple initial guess values, separated with commas."
            gvalues = st.sidebar.text_input(
                "Guess values",
                help=gvalue_help,
                key=key,
            )

            # Store the guess values
            try:
                model.gvalues_range[param_key] = [float(v) for v in gvalues.split(",")]
            except (ValueError, TypeError):
                pass

# ------------------------------------------------- REPETITION PERIOD --------------------------------------------------

period = ""
if N0s is not None:
    period_help = """Excitation repetition period. Used to calculate possible carrier accumulation between 
    consecutive excitation pulses."""
    period = st.sidebar.text_input(
        "Excitation repetition period (ns)",
        help=period_help,
        key="period_input",
    )

    try:
        period = float(period)
    except ValueError:
        period = ""

# ----------------------------------------------------- RUN BUTTON -----------------------------------------------------

run_button = False  # when pressed, run the fit
if N0s is not None:
    run_button = st.sidebar.button("Run")
if run_button:  # change the state of ran
    st.session_state.ran = True

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- RESULTS DISPLAY --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

results_container = st.empty()

# display the results if the run button has been previously pressed and the fitting mode hasn't been changed
if st.session_state.ran:

    # Remove the data and fluence messages
    data_message.empty()
    fluence_message.empty()

    # -------------------------------------------------- FITTING MODE --------------------------------------------------

    if run_button:
        info_message.info("Processing")
        if fit_mode == resources.FITTING_MODE:
            try:
                print("Calling fitting")
                variable = model.fit(xs_data, ys_data, N0s)
                st.session_state.results = copy.deepcopy([variable, model, N0s])
            except ValueError:
                info_message.error("Uh Oh, could not fit the data. Try changing the parameter guess values.")
                raise AssertionError("Fit failed")
        else:
            progressbar = st.sidebar.progress(0)
            st.session_state.period = ""  # force reset
            variable = model.grid_fitting(progressbar, N0s, xs_data=xs_data, ys_data=ys_data)
            st.session_state.results = copy.deepcopy([variable, model, N0s])
        info_message.empty()
    else:
        if st.session_state.results[1] != model or st.session_state.results[2] != N0s:
            info_message.warning('You have changed some of the input settings. Press "run" to apply the changes')
        variable, model, N0s = st.session_state.results

    with results_container.container():

        if fit_mode == resources.FITTING_MODE:
            fit = variable
        else:
            st.subheader("Parallel plot")

            # Calculate the carrier accumulation
            CA = None
            if period:
                if period != st.session_state.period:  # if period is changed
                    print("Calculating CA")
                    CA = [model.get_carrier_accumulation(fit["popts"], fit["N0s"], period) for fit in variable]
                    st.session_state.carrier_accumulation = CA
                    st.session_state.period = period  # store the period used to calculate the CA
                else:
                    print("Getting stored CA")
                    CA = st.session_state.carrier_accumulation
            else:  # reset the store CA and period
                st.session_state.carrier_accumulation = None
                st.session_state.period = period

            # Add the CA to the values
            popts = []
            for i in range(len(variable)):
                popt = variable[i]["values"].copy()
                if CA is not None:
                    popt["Max. CA (%)"] = np.max(list(CA[i].values()))
                popts.append(popt)

            # Display the parallel plot
            xp = plot.parallel_plot(
                popts,
                variable[0]["no_disp_keys"],
                [k for k in variable[0]["values"].keys() if k not in variable[0]["no_disp_keys"]],
            )
            selected = xp.to_streamlit("hip", "selected_uids").display()
            fit = variable[int(selected[0])]
            st.subheader(f"Displaying results of fit #{int(selected[0]) + 1}")

        # --------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------ DISPLAY FIT  ------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        col1, col2 = st.columns(2)

        # -------------------------------------------- FITTING PLOT & EXPORT -------------------------------------------

        col1.markdown("""#### Fitting results""")

        # Plot
        col1.plotly_chart(
            plot.plot_decays(
                fit["xs_data"],
                fit["ys_data"],
                quantity_input,
                fit["fit_ydata"],
                fit["N0s_labels"],
            ),
            use_container_width=True,
        )

        # Export
        header = np.concatenate([["Time (ns)", "Intensity %i" % i] for i in range(1, len(ys_data) + 1)])
        data = [val for pair in zip(fit["xs_data"], fit["fit_ydata"]) for val in pair]
        export_data = utils.matrix_to_string(data, header)
        col1.download_button("Download data", export_data, "pears_fit_data.csv")

        # ---------------------------------------- OPTIMISED PARAMETERS DISPLAY ----------------------------------------

        col1.markdown("""#### Parameters""")
        for label in fit["labels"]:
            col1.markdown(label, unsafe_allow_html=True)

        # ----------------------------------------------- CONTRIBUTIONS ------------------------------------------------

        st.markdown("""#### Contributions""")
        contributions = pd.DataFrame(fit["contributions"], index=fit["N0s_labels"]).transpose()
        st.markdown(contributions.to_html(escape=False) + "<br>", unsafe_allow_html=True)

        # Analysis
        for s in model.get_recommendations(fit["contributions"]):
            st.warning(s)

        # -------------------------------------------- CARRIER ACCUMULATION  -------------------------------------------

        if period:
            st.markdown("""#### Carrier accumulation""")
            nca = model.get_carrier_accumulation(fit["popts"], fit["N0s"], period)
            nca_df = pd.DataFrame(nca, index=["Carrier accumulation (%)"])
            st.markdown(nca_df.to_html(escape=False) + "<br>", unsafe_allow_html=True)

            # Analysis
            max_nca = np.max(list(nca.values()))
            if max_nca > 5.0:
                st.warning(
                    f"""This fit predicts significant carrier accumulation leading to a maximum {max_nca} %% difference 
                    between the single pulse and multiple pulse TRPL decays. You might need to increase your 
                    excitation repetition period to prevent potential carrier accumulation."""
                )
            else:
                st.success("This fit does not predict significant carrier accumulation.")

        # -------------------------------------------- CONCENTRATIONS PLOT ---------------------------------------------

        col2.markdown("""#### Carrier concentrations""")
        concentrations = model.get_carrier_concentrations(
            fit["xs_data"],
            fit["popts"],
            period,
        )
        conc_fig = plot.plot_carrier_concentrations(
            concentrations[0],
            concentrations[2],
            fit["N0s"],
            fit["N0s_labels"],
            concentrations[1],
            model,
        )
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
            labels = ["%i" % (i + 1) for i in range(len(ys_data))]
        st.plotly_chart(
            plot.plot_decays(
                xs_data,
                ys_data,
                quantity_input,
                labels=labels,
            ),
            use_container_width=True,
        )

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- GENERAL INFORMATION ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------- APP DESCRIPTION --------------------------------------------------

with st.expander("About", xs_data[0] is None):
    st.info(
        f"""*Pears* is a user-friendly web app designed to fit time-resolved photoluminescence (TRPL) and time-resolved 
microwave photoconductivity (TRMC) data of perovskite materials using state-of-the-art charge carrier 
recombination models. Two models are available, extensively discussed for TRPL in [here](https://doi.org/10.1039/D0CP04950F).
- The Bimolecular-Trapping-Auger model assumes no doping and that the trap states remain mostly empty over time 
(due, for example, to fast detrapping). As a result, this model is relatively simpler and less likely to lead to over-fitting.
- The Bimolecular-Trapping-Detrapping model accounts for bimolecular recombination, trapping, and detrapping 
with the presence of doping. Its increased complexity can lead to over-parameterisation and ambiguous results.\n
Two modes are available.
- The "{resources.FITTING_MODE}" mode can be used to fit experimental data given a set of guess parameters.
- The "{resources.ANALYSIS_MODE}" mode runs the fitting optimisation for a range of guess parameters. If all the 
optimisations do not converge toward the same values, then the fitting is inaccurate due to the possibility of multiple 
solutions.\n
App created and maintained by [Emmanuel V. Pean](https://emmanuelpean.streamlit.app/).  
[Version 0.4](https://github.com/Emmanuelpean/pears) (last updated: 9th May 2022)."""
    )  # TODO update date

# -------------------------------------------------- MODEL DESCRIPTION -------------------------------------------------

with st.expander("Model & computational details"):
    st.markdown(
        r"""The following information can be found with more details [here](https://doi.org/10.1039/D0CP04950F)
    (Note that for simplicity purpose, the $\Delta$ notation for the photoexcited carriers was dropped here for 
    simplicity purposes *e.g.* $\Delta n_e$ in the paper is $n_e$ here)."""
    )
    st.markdown("""#### Models""")
    st.markdown(
        """Two charge carrier recombination models can be used whose rate equations for the different carrier 
        concentrations are given below."""
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center; '>Bimolecular-trapping model</h3>", unsafe_allow_html=True)
        st.markdown("""%s""" % utils.render_image(resources.BT_MODEL_PATH, 65), unsafe_allow_html=True)
        st.latex(r"n_e(t)=n_h(t)=n(t)")
        st.latex(r"\frac{dn}{dt}=-k_Tn-k_Bn^2-k_An^3,\ \ \ n^p(t=0)=n^{p-1}(T)+N_0")
        st.latex(r"I_{TRPL} \propto n^2")
        st.latex(r"I_{TRMC}=2\mu n / N_0")
        st.markdown(
            r"""where:
* $n$ is the photoexcited carrier concentration (in $cm^{-3}$)
* $k_T$ is the trapping rate constant (in $ns^{-1}$)
* $k_B$ is the bimolecular recombination rate constant (in $cm^3/ns$)
* $k_A$ is the Auger recombination rate constant (in $cm^6/ns$)
* $\mu$ is the carrier mobility (in $cm^2/(Vs)$)"""
        )
    with col2:
        st.markdown(
            "<h3 style='text-align: center; '>Bimolecular-trapping-detrapping model</h3>", unsafe_allow_html=True
        )
        st.markdown(utils.render_image(resources.BTD_MODEL_PATH, 65), unsafe_allow_html=True)
        st.latex(r"\frac{dn_e}{dt}=-k_B n_e (n_h+p_0 )-k_T n_e [N_T-n_t ],\ \ \ n_e^p(t=0)=n_e^{p-1}(T)+N_0")
        st.latex(r"\frac{dn_t}{dt}=k_T n_e [N_T-n_t]-k_D n_t (n_h+p_0 ),\ \ \ n_t^p(t=0)=n_t^{p-1}(T)")
        st.latex(r"\frac{dn_h}{dt}=-k_B n_e (n_h+p_0 )-k_D n_t (n_h+p_0 ),\ \ \ n_h^p(t=0)=n_h^{p-1}(T)+N_0")
        st.latex(r"I_{TRPL} \propto n_e(n_h+p_0)")
        st.latex(r"I_{TRMC} = \left(\mu_e n_e + \mu_h n_h \right)/N_0")
        st.markdown(
            r"""where:
* $n_e$ is the photoexcited electron concentration (in $cm^{-3}$)
* $n_h$ is the photoexcited hole concentration (in $cm^{-3}$)
* $n_t$ is the trapped electron concentration (in $cm^{-3}$)
* $k_B$ is the bimolecular recombination rate constant (in $cm^3/ns$)
* $k_T$ is the trapping rate constant (in $cm^3/ns$)
* $k_D$ is the detrapping rate constant (in $cm^3/ns$)
* $N_T$ is the trap state concentration (in $cm^{-3}$)
* $p_0$ is the dark hole concentration (in $cm^{-3}$)
* $\mu_e$ is the electron mobility (in $cm^2/(Vs)$)
* $\mu_h$ is the hole mobility (in $cm^2/(Vs)$)"""
        )
    st.markdown("""####""")
    st.markdown(
        """For both models, the photoexcited charge carrier concentration $N_0$ is the concentration of carriers
    excited by a single excitation pulse. The initial condition of a carrier concentration after excitation pulse $p$ is 
    given by the sum of any remaining carriers $n_X^{p-1}(T)$ ($T$ is the excitation repetition period) just before 
    excitation plus the concentration of carrier generated by the pulse $N_0$ (except for the trapped electrons)."""
    )

    st.markdown("""#### Fitting""")
    st.markdown(
        r"""Fitting is carried using the least square optimisation. For a dataset containing $M$ curves, 
    each containing $N_i$ data points, the residue $SS_{res}$ is:"""
    )
    st.latex(r"""SS_{res}=\sum_i^M\sum_j^{N_i}\left(y_{i,j}-F(t_{i,j},A_{i})\right)^2""")
    st.markdown(
        """where $y_{i,j}$ is the intensity associated with time $t_{i,j}$ of point $j$ of curve $i$.
    $A_i$ are the model parameters associated with curve $i$ and $F$ is the fitting model given by for the TRPL and TRMC respectively:"""
    )
    st.latex(r"F(t,I_0, y_0, k_B,...)=I_0 \frac{I_{TRPL}(t,k_B,...)}{I_{TRPL}(0, k_B,...)} + y_0")
    st.latex(r"F(t, y_0, k_B,...)=I_{TRMC}(t,k_B,...) + y_0")
    st.markdown(
        """ where $I_0$ is an intensity factor and $y_0$ is an intensity offset. Contrary to the other parameters of 
    the models (e.g. $k_B$), $I_0$ and $y_0$ are not kept the same between the different TRPL curves *i.e.* the fitting 
    models for curves $A$, $B$,... are:"""
    )
    st.latex(r"F_A(t,I_0^A, y_0^A, k_B,...)=I_0^A \frac{I_{TRPL}(t,k_B,...)}{I_{TRPL}(0, k_B,...)} + y_0^A")
    st.latex(r"F_B(t,I_0^B, y_0^B, k_B,...)=I_0^B \frac{I_{TRPL}(t,k_B,...)}{I_{TRPL}(0, k_B,...)} + y_0^B")
    st.latex("...")
    st.markdown(
        """By default, $I_0$ and $y_0$ are respectively fixed at 1 and 0 (assuming no background noise and 
    normalised intensity. The quality of the fit is estimated from the coefficient of determination $R^2$:"""
    )
    st.latex(r"R^2=1-\frac{SS_{res}}{SS_{total}}")
    st.markdown(
        r"""where $SS_{total}$ is defined as the sum of the squared difference between each point and the 
    average of all curves $\bar{y}$:"""
    )
    st.latex(r"""SS_{total}=\sum_i^M\sum_j^{N_i}\left(y_{i,j}-\bar{y}\right)^2""")
    st.markdown(
        """For fitting, it is assumed that there is no carrier accumulation between excitation pulses due to the
    presence of non-recombined carriers from previous excitation pulses. This requires the TRPL decays to be measured
    with long enough excitation repetition periods such that all carriers can recombine."""
    )

    st.markdown("""#### Carrier accumulation""")
    st.markdown(
        """It is possible to  calculate the expected effect of carrier accumulation on the TRPL and TRMC from the 
    parameters retrieved from the fits if the repetition period is provided. The carrier accumulation ($CA$) is 
    calculated as the maximum difference between the simulated TRPL/TRMC after the first ($p=1$) and stabilised ($p=s$) pulses:"""
    )
    st.latex(
        r"{CA}_{TRPL}=\max\left({\frac{I_{TRPL}^{p=1}(t)}{I_{TRPL}^{p=1}(0)}-\frac{I_{TRPL}^{p=s}(t)}{I_{TRPL}^{p=s}(0)}})\right)"
    )
    st.latex(r"{CA}_{TRMC}=\max\left(I_{TRMC}^{p=1}(t)-I_{TRMC}^{p=s}(t))\right)")
    st.markdown(
        """The stabilised pulse is defined as when the electron and hole concentrations vary by less than 
    10<sup>-3</sup> % of the photoexcited concentration between two consecutive pulses:""",
        unsafe_allow_html=True,
    )
    st.latex(r"|n_e^p(t)-n_e^{p+1}(t)|<10^{-5} N_0")
    st.latex(r"|n_h^p(t)-n_h^{p+1}(t)|<10^{-5} N_0")

    st.markdown("""#### Contributions""")
    st.markdown(
        """The contribution of each process to the TRPL intensity variations over time is calculated from the 
    fitted values (see Equations 22 to 25 [here](https://doi.org/10.1039/D0CP04950F)). It is important to ensure that the 
    contribution of a process (e.g., trapping) is non-negligible so that the associated parameters (e.g., $k_T$ 
    in the case of the bimolecular-trapping model) are accurately retrieved.""",
        unsafe_allow_html=True,
    )

    st.markdown("""#### Grid fitting""")
    st.markdown(
        f"""This mode runs the fitting process for a grid of guess values. The grid is generated from every 
    possible combinations of guess values supplied (e.g. $k_B$: 10<sup>-20</sup>, 10<sup>-19</sup> and 
    $k_T$: 10<sup>-3</sup>, 10<sup>-2</sup> yields 4 sets of guess values: (10<sup>-20</sup>, 10<sup>-3</sup>),
    (10<sup>-20</sup>, 10<sup>-2</sup>), (10<sup>-19</sup>, 10<sup>-3</sup>) and (10<sup>-19</sup>, 10<sup>-2</sup>)). 
    Note that in the case of the bimolecular-trapping-detrapping model, only set of guess values satisfying $k_T>k_B$ 
    and $k_T>k_D$ are considered to keep the computational time reasonable. Fitting is then carried using each set of 
    guess values as schematically represented below: {utils.render_image(resources.OPT_GUESS_PATH, 60, "png")}
    In the case where all the optimisations converge towards a similar solution, it can be assumed that only 1 solution
    exist and that therefore the parameter values obtained accurately describe the system measured. However, if the fits 
    converge toward multiple solutions, it is not possible to ascertain which solution represents the system accurately.""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------- HOW TO -------------------------------------------------------

with st.expander("Getting started"):
    st.markdown("""#### Example""")
    data1_link = utils.generate_download_link(resources.BT_TRPL_DATA, text="TRPL data set 1", name="TRPL data set 1")
    data2_link = utils.generate_download_link(resources.BTD_TRPL_DATA, text="TRPL data set 2", name="TRPL data set 2")
    data3_link = utils.generate_download_link(resources.BTD_TRMC_DATA_2, text="TRMC data set 3", name="TRMC data set 1")
    st.markdown("""Follow these steps to fit TRPL/TRMC decays.""")
    st.markdown(
        f"""1. Upload your data and select the data format (tab/space in this example). Select whether TRPL or TRMC is analysed.
    Check the "Pre-process data" box for PEARS to shift the decay(s) and normalise it (the later is done only for TRPL data);
    * _e.g._ {data1_link}, 
    * _e.g._ {data2_link}
    * _e.g._ {data3_link} (select X1/Y1/X2/Y2... data format)""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """2. Enter the photoexcited carrier concentrations (in $cm^{-3}$) for each TRPL/TRMC decay 
    measured (separated by a comma);
    * _e.g._: 5.58e15, 1.00e16, 2.23e16 (data set 1)
    * _e.g._: 55e12, 164e12, 511e12, 1720e12, 4750e12 (data set 2 & 3)""",
        unsafe_allow_html=True,
    )

    st.markdown("""3. Choose a fitting model;""")

    st.markdown("""4. (Optional) Set the value of known parameters;""")

    if fit_mode == "Fitting":
        st.markdown("""5. (Optional) For non fixed parameters, choose the guess value for the optimisation;""")
    else:
        st.markdown("""5. (Optional) For non fixed parameters, choose multiple guess values for each parameter;""")

    st.markdown(
        """6. (Optional) Enter the excitation repetition period (in ns). If provided, *Pears* calculates the 
    carrier accumulation effect on the TRPL intensity as well as show the evolution of the carrier concentrations
    after multiple consecutive excitation pulses until stabilisation."""
    )

    st.markdown("""7. Press "run" """)

    st.markdown("""#### Video tutorial""")
    st.video(resources.TUTORIAL_PATH)

# ------------------------------------------------------ CHANGELOG -----------------------------------------------------

with st.expander("Changelog"):
    st.markdown(
        """
    #### March 2025 - V 0.4.0.0 - TRMC data support
    * Pears can now fit time-resolved microwave photoconductivity data 🎉.
    * Both recombination models have been modified to account for the carrier mobility.
    * A new "Quantity" input has been added to the sidebar to select the type of data to fit.
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
    * Updated core layout
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
    * Now accept a wider range of data"""
    )

# ----------------------------------------------------- DISCLAIMER -----------------------------------------------------

with st.expander("Disclaimer"):
    st.markdown(
        """THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,  INCLUDING BUT 
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. 
    IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER 
    LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
    USE OR OTHER DEALINGS IN THE SOFTWARE."""
    )
