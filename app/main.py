"""main script
Important notes:
- Changing the app mode or the data resets all the results
- Changing the period change data associated with the fit
- Changing any other value does not re-run the fit but displays a message about it
- Changing the data loaded resets everything"""  # TODO complete that

import copy

import numpy as np
import pandas as pd
import streamlit as st

import models
import plot
import resources
from utility.data import load_data, matrix_to_string, process_data, render_image
from utility.numbers import get_power_labels, to_scientific

__version__ = "0.4.0"
__date__ = "9th May 2022"


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- SET UP -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# General setup & layout
st.set_page_config("Pears", resources.ICON_PATH, layout="wide")

if "results" not in st.session_state:
    st.session_state.results = []  # list of all the results
if "carrier_accumulation" not in st.session_state:
    st.session_state.carrier_accumulation = None  # carrier accumulation
if "models" not in st.session_state:
    st.session_state.models = copy.deepcopy(models.models)


def reset_carrier_accumulation() -> None:
    print("Resetting carrier accumulation")
    st.session_state.carrier_accumulation = None


def reset_results() -> None:
    """Reset the stored values"""

    print("Resetting results")
    st.session_state.results = []
    reset_carrier_accumulation()


# Change the default style
@st.cache_resource
def set_style() -> None:
    """Set the default style"""

    with open(resources.CSS_STYLE_PATH) as ofile:
        st.markdown(f"<style>{ofile.read()}</style>", unsafe_allow_html=True)


set_style()

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- INPUT -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

logo_placeholder = st.empty()
st.sidebar.markdown(render_image(resources.LOGO_PATH, 20), unsafe_allow_html=True)  # sidebar logo
info_message = st.empty()

# -------------------------------------------------------- MODES -------------------------------------------------------

fit_mode = st.sidebar.selectbox("Mode", resources.APP_MODES, on_change=reset_results, key="fit_mode_")

# -------------------------------------------------------- DATA --------------------------------------------------------

# File uploader
input_filename = st.sidebar.file_uploader(
    label="Data file",
    key="input_filename_",
    on_change=reset_results,
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
    options=[",", None, ";"],
    help=data_delimiter_help,
    key="data_delimiter",
    horizontal=True,
    format_func=lambda v: {None: "tab/space", ",": "comma", ";": "semicolon"}[v],
    on_change=reset_results,
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
    preprocess_help = r"""Shift the decay maximum intensity to $\mathsf{t=0}$ and normalise the intensity."""
else:
    preprocess_help = r"""Shift the decay maximum intensity to $\mathsf{t=0}$."""
process_input = st.sidebar.checkbox(
    label="Pre-process data",
    help=preprocess_help,
    key="preprocess_",
    on_change=reset_results,
)

# Load the data
data_message = st.empty()
xs_data, ys_data = [None], [None]
if input_filename is not None:
    try:
        try:
            xs_data, ys_data = load_data(input_filename.getvalue(), data_delimiter, data_format)
        except:
            raise ValueError("Unknown error, Check that the correct delimiter has been selected.")

        # Check number of arrays consistency
        if len(xs_data) != len(ys_data):
            raise ValueError("Mismatch: x data and y data must have the same number of columns.")

        # Check length arrays consistency
        for i, (x, y) in enumerate(zip(xs_data, ys_data)):
            if len(x) != len(y):
                raise ValueError(f"Mismatch at index {i + 1}: x and y columns must have the same length.")

        # Process the data
        if process_input:
            xs_data, ys_data = process_data(xs_data, ys_data, quantity_input == "TRPL")

        data_message.success("Data successfully loaded")

    except Exception as e:  # if an error occurs during the file reading
        xs_data, ys_data = [None], [None]
        data_message.error(f"Uh-oh! The data could not be loaded. Error: {e}")

else:
    data_message.info("Load a data file")


if xs_data[0] is None:
    logo_placeholder.markdown(render_image(resources.LOGO_TEXT_PATH, 50), unsafe_allow_html=True)  # main logo

# ------------------------------------------------------ FLUENCES ------------------------------------------------------

fluence_message = st.empty()
N0s = None
if xs_data[0] is not None:
    n0_help = r"""Photoexcited carrier concentration(s separated by commas), calculated as 
    $\mathsf{N_0=A\frac{I_0}{E_{photon}D}}$ where $\mathsf{I_0}$ is the excitation pulse fluence 
    (in $\mathsf{J/cm^2}$), $\mathsf{D}$ is the film thickness (in $\mathsf{cm}$), $\mathsf{E_{photon}}$ 
    is the the photon energy (in $\mathsf{J}$) and $\mathsf{A}$ is the sample absorptance."""
    st.sidebar.markdown(r"Photoexcited carrier concentrations ($\mathsf{cm^{-3}}$)", help=n0_help)
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
        format_func=lambda v: {"BTD": "Bimolecular-Trapping-Detrapping", "BTA": "Bimolecular-Trapping-Auger"}[v],
    )
    model: models.Model | None = st.session_state.models[model_name][quantity_input]


# ------------------------------------------------ FIXED & GUESS VALUES ------------------------------------------------


if N0s is not None:
    param_key = st.sidebar.selectbox(
        "Parameters",
        model.param_ids,
        format_func=model.get_parameter_label,
        key="param_key_",
    )

    # -------------------------------------------------- FIXED VALUES --------------------------------------------------

    key = model_name + param_key + "fixed"

    if key not in st.session_state:
        st.session_state[key] = to_scientific(model.fvalues[param_key])

    # Display the fixed value
    fvalue_help = "Fixed values for the selected parameter. If not a number, use the guess value below for the fitting."
    fvalue = st.sidebar.text_input(
        "Fixed value",
        help=fvalue_help,
        key=key,
    )

    # Store the new value
    if fvalue:
        try:
            model.fvalues[param_key] = float(fvalue)
        except:
            pass
    else:
        model.fvalues[param_key] = None

    # -------------------------------------------------- GUESS VALUES --------------------------------------------------

    # If no fixed value, display and set the guess value
    if model.fvalues[param_key] is None:

        # Fitting mode
        if fit_mode == resources.FITTING_MODE:

            key = model_name + param_key + "guess"
            if key not in st.session_state:
                st.session_state[key] = to_scientific(model.gvalues[param_key])

            # Display the guess value
            gvalue = st.sidebar.text_input(
                "Guess value",
                help="Parameter initial guess value for the fitting.",
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
                st.session_state[key] = to_scientific(model.gvalues_range[param_key])

            # Display the guess values
            gvalues = st.sidebar.text_input(
                "Guess values",
                help="Enter multiple initial guess values, separated with commas.",
                key=key,
            )

            # Store the guess values
            try:
                model.gvalues_range[param_key] = [float(v) for v in gvalues.split(",")]
            except (ValueError, TypeError):
                pass

# ------------------------------------------------- REPETITION PERIOD --------------------------------------------------


period = None
if N0s is not None:
    period_help = """Excitation repetition period. Used to calculate possible carrier accumulation between 
    consecutive excitation pulses."""
    period = st.sidebar.text_input(
        "Excitation repetition period (ns)",
        help=period_help,
        key="period_input",
        on_change=reset_carrier_accumulation,
    )

    try:
        period = float(period)
    except ValueError:
        pass

# ----------------------------------------------------- RUN BUTTON -----------------------------------------------------

run_button = False  # when pressed, run the fit
if N0s is not None:
    run_button = st.sidebar.button("Run")

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- RESULTS DISPLAY --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

results_container = st.empty()

# display the results if results have been previously stored
if st.session_state.results or run_button:

    # Remove the data and fluence messages
    data_message.empty()
    fluence_message.empty()

    # ----------------------------------------------------- FITTING ----------------------------------------------------

    # If the run button has been clicked
    if run_button:
        info_message.info("Processing")
        reset_results()

        # Fitting
        if fit_mode == resources.FITTING_MODE:
            try:
                print("Calling fitting")
                fit_output = model.fit(xs_data, ys_data, N0s)
                st.session_state.results = copy.deepcopy([fit_output, model, N0s])
            except ValueError:
                bad_fit_message = "The data could not be fitted. Try changing the parameter guess or fixed values."
                info_message.error(bad_fit_message)
                raise AssertionError("Fit failed")

        # Grid Fitting
        else:
            progressbar = st.sidebar.progress(0)  # initialise the progress bar
            fit_output = model.grid_fitting(progressbar, N0s, xs_data=xs_data, ys_data=ys_data)
            st.session_state.results = copy.deepcopy([fit_output, model, N0s])

        info_message.empty()

    # Retrieve the fit output, model and carrier concentrations from the session state
    else:
        # Check if the model settings or the carrier concentrations have changed
        if st.session_state.results[1] != model or st.session_state.results[2] != N0s:
            info_message.warning("You have changed some of the input settings. Press 'Run' to apply these changes.")
        fit_output, model, N0s = st.session_state.results

    # ---------------------------------------------- CARRIER ACCUMULATION ----------------------------------------------

    carrier_accumulation = None
    if period:
        if st.session_state.carrier_accumulation is None:  # if carrier accumulation has not been calculated
            print("Calculating CA")
            if fit_mode == resources.ANALYSIS_MODE:
                carrier_accumulation = [model.get_carrier_accumulation(fit["popts"], period) for fit in fit_output]
            else:
                carrier_accumulation = model.get_carrier_accumulation(fit_output["popts"], period)
            st.session_state.carrier_accumulation = carrier_accumulation  # store the carrier accumulation value
        else:
            print("Getting stored CA")
            carrier_accumulation = st.session_state.carrier_accumulation

    # -------------------------------------------------- PARALLEL PLOT -------------------------------------------------

    with results_container.container():

        if fit_mode == resources.FITTING_MODE:
            fit_displayed = fit_output
            message = """#### Fitting results"""
        else:
            st.subheader("Parallel plot")

            # Add the carrier accumulation to the values
            popts = []
            for i, fit_popt in enumerate(fit_output):
                popt = fit_popt["all_values"].copy()
                if carrier_accumulation is not None:
                    popt["Max. CA (%)"] = np.max(carrier_accumulation[i])
                popts.append(popt)

            # Display the parallel plot
            xp = plot.parallel_plot(popts, fit_output[0]["hidden_keys"])
            selected = xp.to_streamlit("hip", "selected_uids").display()
            fit_displayed = fit_output[int(selected[0])]
            message = f"Displaying results of fit #{int(selected[0]) + 1}"

        # --------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------ DISPLAY FIT  ------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        col1, col2 = st.columns(2)

        # -------------------------------------------- FITTING PLOT & EXPORT -------------------------------------------

        col1.markdown(message)

        # Plot
        figure = plot.plot_decays(
            fit_displayed["xs_data"],
            fit_displayed["ys_data"],
            quantity_input,
            fit_displayed["fit_ydata"],
            fit_displayed["N0s_labels"],
        )
        col1.plotly_chart(figure, use_container_width=True)

        # Export
        header = np.concatenate([["Time (ns)", "Intensity %i" % i] for i in range(1, len(ys_data) + 1)])
        data = [val for pair in zip(fit_displayed["xs_data"], fit_displayed["fit_ydata"]) for val in pair]
        export_data = matrix_to_string(data, header)
        col1.download_button("Download data", export_data, "pears_fit_data.csv")

        # ---------------------------------------- OPTIMISED PARAMETERS DISPLAY ----------------------------------------

        col1.markdown("""#### Parameters""")
        popts_string = '<div class="custom-line-spacing">' + "<br/>".join(fit_displayed["popt_labels"]) + "</div>"
        col1.markdown(popts_string, unsafe_allow_html=True)

        # ----------------------------------------------- CONTRIBUTIONS ------------------------------------------------

        col1.markdown("""#### Contributions""")
        contributions = {model.CBT_LABELS[key]: value for key, value in fit_displayed["contributions"].items()}
        contributions = pd.DataFrame(contributions, index=fit_displayed["N0s_labels"]).transpose()
        col1.markdown(contributions.to_html(escape=False) + "<br>", unsafe_allow_html=True)

        # Contribution analysis
        for s in model.get_contribution_recommendations(fit_displayed["contributions"]):
            col1.warning(s)

        # -------------------------------------------- CARRIER ACCUMULATION  -------------------------------------------

        if period:
            col1.markdown("""#### Carrier accumulation""")
            nca_dict = dict(zip(fit_displayed["N0s_labels"], carrier_accumulation))
            nca_df = pd.DataFrame(nca_dict, index=["Carrier accumulation (%)"])
            col1.markdown(nca_df.to_html(escape=False) + "<br>", unsafe_allow_html=True)

            # Analysis
            max_nca = np.max(carrier_accumulation)
            if max_nca > 5.0:
                ca_warning = f"""This fit predicts significant carrier accumulation leading to a maximum {max_nca:.1f} % difference 
                                 between the single pulse and multiple pulse {quantity_input} decays. You might need to increase your 
                                 excitation repetition period to prevent potential carrier accumulation."""
                col1.warning(ca_warning)
            else:
                col1.success("This fit does not predict significant carrier accumulation.")

        # -------------------------------------------- CONCENTRATIONS PLOT ---------------------------------------------

        col2.markdown("""#### Carrier concentrations""")
        concentrations = model.get_carrier_concentrations(
            fit_displayed["xs_data"],
            fit_displayed["popts"],
            period,
        )
        conc_fig = plot.plot_carrier_concentrations(
            concentrations[0],
            concentrations[2],
            fit_displayed["N0s"],
            fit_displayed["N0s_labels"],
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
            labels = get_power_labels(N0s)
        else:
            labels = None
        figure = plot.plot_decays(
            xs_data,
            ys_data,
            quantity_input,
            labels=labels,
        )
        st.plotly_chart(figure, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- GENERAL INFORMATION ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------- APP DESCRIPTION --------------------------------------------------

with st.expander("About", xs_data[0] is None):
    st.info(
        f"""*Pears* is a user-friendly web app designed to fit time-resolved photoluminescence (TRPL) and time-resolved 
microwave photoconductivity (TRMC) data of perovskite materials using state-of-the-art charge carrier 
recombination models (extensively discussed for TRPL in https://doi.org/10.1039/D0CP04950F):
* The **Bimolecular-Trapping-Auger** (BTA) accounts for bimolecular band-to-band recombination, monomolecular trapping and 
trimolecular Auger recombination. Because Auger recombination are usually non-negligible only at very high excitation fluences, 
Auger recombination are usually ignored within this model (the Auger recombination rate constant $k_A$ is fixed to 0 by default).
This model assumes that the trap states remain mostly empty and that doping is negligible. As a result, this model is 
relatively simple and less likely to lead to over-parameterisation.
* The **Bimolecular-Trapping-Detrapping** (BTD) model accounts for bimolecular band-to-band recombination, bimolecular 
trapping, and bimolecular detrapping. Contrary to the BTA model, the BTD model accounts for the population and depopulation 
of the trap states, as well as the presence of doping. However, the increased complexity of this model can lead 
to over-parameterisation and ambiguous results.\n

*Pears* offers two operational modes:
* The **{resources.FITTING_MODE}** mode is the primary mode for fitting experimental TRPL and TRMC data.
* The **{resources.ANALYSIS_MODE}** mode allows to run the fitting optimisation across a range of guess parameters, helping to 
identify whether multiple sets of parameters yield fitting solutions, as discussed in https://doi.org/10.1002/smtd.202400818. 
If the optimisations do not converge to the same values, the fitting may be inaccurate due to the presence of multiple 
possible solutions.\n
App created and maintained by [Emmanuel V. Pean](https://emmanuelpean.streamlit.app/).  
[Version {__version__}](https://github.com/Emmanuelpean/pears) (last updated: {__date__}).  
If you use *Pears* to fit your TRPL or TRMC data, please cite https://doi.org/10.1021/acs.jcim.3c00217."""
    )

# -------------------------------------------------- MODEL DESCRIPTION -------------------------------------------------

with st.expander("Model & Computational Details"):
    st.markdown(
        r"""The following information can be found with more details [here](https://doi.org/10.1039/D0CP04950F)
    (Note that for simplicity purpose, the $\Delta$ notation for the photoexcited carriers was dropped here for 
    simplicity purposes *e.g.* $\Delta n_e$ in the paper is $n_e$ here)."""
    )
    st.markdown("""#### Models""")
    st.markdown(
        """Both the Bimolecular-Trapping-Auger and Bimolecular-Trapping-Detrapping models can be used which rate equations 
        for the different carrier concentrations are given below."""
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center; '>Bimolecular-Trapping-Auger model</h3>", unsafe_allow_html=True)
        st.markdown(render_image(resources.BT_MODEL_PATH, 65), unsafe_allow_html=True)
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
            "<h3 style='text-align: center; '>Bimolecular-Trapping-Detrapping model</h3>", unsafe_allow_html=True
        )
        st.markdown(render_image(resources.BTD_MODEL_PATH, 65), unsafe_allow_html=True)
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
    st.markdown(
        """For both models, the photoexcited charge carrier concentration $N_0$ is the concentration of carriers
    excited by a single excitation pulse. The initial condition of a electron and hole concentrations after excitation pulse $p$ is 
    given by the sum of any remaining carriers $n_X^{p-1}(T)$ ($T$ is the excitation repetition period) just before 
    excitation plus the concentration of carrier generated by the pulse $N_0$ (except for the trapped electrons)."""
    )

    st.markdown("""#### Fitting""")
    st.markdown(
        """Fitting is done using a least square optimisation. For a dataset containing $M$ curves, each containing 
        $N_i$ data points, the residue $SS_{res}$ is:"""
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
        """**By default, $I_0$ and $y_0$ are respectively fixed at 1 and 0 (assuming no background noise and 
    normalised intensity**. The quality of the fit is estimated with the coefficient of determination $R^2$:"""
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

    st.markdown("""#### Carrier Accumulation""")
    st.markdown(
        """*Pears* can calculate the expected effect of carrier accumulation on the TRPL and TRMC from the 
    parameters retrieved from the fits if the repetition period is provided. The carrier accumulation ($CA$) is 
    calculated as the maximum difference between the simulated TRPL/TRMC after the first ($p=1$) and stabilised ($p=s$) pulses:"""
    )
    st.latex(
        r"{CA}_{TRPL}=\max\left({\frac{I_{TRPL}^{p=1}(t)}{I_{TRPL}^{p=1}(0)}-\frac{I_{TRPL}^{p=s}(t)}{I_{TRPL}^{p=s}(0)}}\right)"
    )  # TODO should not account for y_0 and I
    st.latex(r"{CA}_{TRMC}=\max\left(I_{TRMC}^{p=1}(t)-I_{TRMC}^{p=s}(t)\right)")
    st.markdown(
        """The stabilised pulse is defined as when the electron and hole concentrations vary by less than 
    10<sup>-3</sup> % of the photoexcited concentration between two consecutive pulses:""",
        unsafe_allow_html=True,
    )
    st.latex(r"|n_e^p(t)-n_e^{p+1}(t)|<10^{-5} N_0")
    st.latex(r"|n_h^p(t)-n_h^{p+1}(t)|<10^{-5} N_0")

    st.markdown("""#### Process Contributions""")
    st.markdown(
        """The contribution of each process to the TRPL/TRMC intensity variations over time is calculated from the 
    fitted values (see Equations 22 to 25 in https://doi.org/10.1039/D0CP04950F). It is important to ensure that the 
    contribution of a process (*e.g.*, trapping) is non-negligible so that the associated parameters (*e.g.*, $k_T$ 
    in the case of the Bimolecular-Trapping-Auger model) are accurately retrieved.""",
        unsafe_allow_html=True,
    )

    st.markdown("""#### Grid Fitting""")
    st.markdown(
        f"""This mode runs the fitting process for a grid of guess values. The grid is generated from every 
    possible combinations of guess values supplied (*e.g.*, $k_B$: 10<sup>-20</sup>, 10<sup>-19</sup> and 
    $k_T$: 10<sup>-3</sup>, 10<sup>-2</sup> yields 4 sets of guess values: (10<sup>-20</sup>, 10<sup>-3</sup>),
    (10<sup>-20</sup>, 10<sup>-2</sup>), (10<sup>-19</sup>, 10<sup>-3</sup>) and (10<sup>-19</sup>, 10<sup>-2</sup>)). 
    Note that in the case of the bimolecular-trapping-detrapping model, only set of guess values satisfying $k_T>k_B$ 
    and $k_T>k_D$ are considered to keep the computational time reasonable. Fitting is then carried using each set of 
    guess values as schematically represented below: {render_image(resources.OPT_GUESS_PATH, 60, "png")}
    In the case where all the optimisations converge towards a similar solution, it can be assumed that only 1 solution
    exist and that therefore the parameter values obtained accurately describe the system measured. However, if the fits 
    converge toward multiple solutions, it is not possible to ascertain which solution represents the system accurately.
    In this case, measuring the TRMC (if the TRPL was fitted) or the TRPL (if the TRMC was measured) can allow to determine 
    which solution represents the system best (see https://doi.org/10.1002/smtd.202400818)""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------- HOW TO -------------------------------------------------------

with st.expander("Getting started"):
    st.markdown("""#### How To""")
    st.markdown("""Follow these steps to fit TRPL/TRMC decays.""")
    st.markdown(
        f"""1. Upload your data and select the correct data format and delimiter. The data should be displayed.
        Select the appropriate quantity (TRPL or TRMC). Check the "Pre-process data" if required, to shift the data 
        and normalise them (for TRPL only). The following datasets are provided for testing purpose:
    * {resources.BT_TRPL_LINK}, 
    * {resources.BTD_TRPL_LINK},
    * {resources.BT_TRMC_LINK},
    * {resources.BTD_TRMC_LINK}""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""2. Input the photoexcited carrier concentrations (in cm<sup>-3</sup>) for each decay. Use the values below 
        if you are using the example datasets:
    * _e.g._: {', '.join([('%.2e' % f).replace('+', '') for f in resources.BT_TRPL_DATA[-1]])} (BT model datasets)
    * _e.g._: {', '.join([('%.2e' % f).replace('+', '') for f in resources.BTD_TRPL_DATA[-1]])} (BTD model datasets)""",
        unsafe_allow_html=True,
    )

    st.markdown("""3. Choose the desired fitting model.""")

    st.markdown("""4. (Optional) Set the value of known parameters.""")

    if fit_mode == "Fitting":
        st.markdown("""5. (Optional) For non fixed parameters, choose the guess value for the optimisation.""")
    else:
        st.markdown("""5. (Optional) For non fixed parameters, choose multiple guess values for each parameter.""")

    st.markdown(
        """6. (Optional) Enter the excitation repetition period (in ns) to calculate the carrier accumulation effect on 
        the TRPL/TRMC intensity as well as show the evolution of the carrier concentrations after multiple consecutive 
        excitation pulses until stabilisation. Note that this can be done after fitting."""
    )

    st.markdown("""7. Press "Run" """)

    st.markdown("""#### Video tutorial""")
    st.video(resources.TUTORIAL_PATH)

# ------------------------------------------------------ CHANGELOG -----------------------------------------------------

with st.expander("Changelog"):
    changelog = """
    #### Version 0.4.0.0 - March 2025
    * Pears can now fit time-resolved microwave photoconductivity (TRMC) data.
    * A new "Quantity" input has been added to the sidebar to select the type of data to fit.
    * Changed the colour theme to match the *Pears* logo.
    * Changed the photoexcited carrier concentration input to match the number of decays in the uploaded file.
    #### Version 0.3.1.2 - May 2022
    * The Auger rate constant is now fixed to 0 by default.
    * Fixed a bug during which data could not be successfully loaded.
    #### Version 0.3.1.1 - April 2022
    * Fixed some bugs regarding file upload that were introduced in the previous update.
    #### Version 0.3.1 - March 2022
    * Added new "Pre-process" option to shift the data to zero and normalise them.
    * Files with a header can now be uploaded.
    #### Version 0.3.0 - November 2021
    * Added the I0 intensity parameter to the fitting procedure.
    * Added a new file input format (X1/Y1/X2/Y2).
    * Added the calculation of the process contributions.
    * Added the calculation of the carrier accumulation effect on the TRPL.
    * Added help bubbles.
    * Fixed a pesky bug that prevented from changing the parameter fixed values.
    * Changed the model descriptions.
    * Added new parallel plot visualisation for grid fitting analysis.
    * Fits and Grid Fit analysis now stay on screen until the run button is pressed again (unless the mode or the data 
    are changed).
    * Added disclaimer.
    * Added website icon.
    * Added video tutorial.
    * Now accept a wider range of data."""
    st.markdown(changelog)

# ----------------------------------------------------- DISCLAIMER -----------------------------------------------------

with st.expander("Disclaimer"):
    disclaimer = """THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,  INCLUDING BUT 
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. 
    IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER 
    LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
    USE OR OTHER DEALINGS IN THE SOFTWARE."""
    st.markdown(disclaimer)
