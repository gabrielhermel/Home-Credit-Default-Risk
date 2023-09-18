#!/usr/bin/env python

import streamlit as st
import requests
import yaml

# Read the configuration file
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Get the API URL from the configuration file
fastapi_url = config.get("api", {}).get("url")

# Check that the URL was found
if not fastapi_url:
    st.error("FastAPI URL not defined in config.yaml.")
    st.stop()  # Stop execution

# Initialize session variables
if "max_feats" not in st.session_state:
    # Make a GET request to the FastAPI endpoint for the applicants and approval status
    num_feats_response = requests.get(f"{fastapi_url}/get_num_feats")

    # Check if the request was successful (status code 200)
    if num_feats_response.status_code == 200:
        # Slider value for max number of features
        st.session_state["max_feats"] = num_feats_response.json()
        # Current plot
        st.session_state["curr_plot"] = "none"
    # Handle the case where the request was not successful
    else:
        st.error(
            f"Failed to fetch data. Status code: {num_feats_response.status_code}. "
            + f"The endpoint '{fastapi_url}/get_num_feats' is not accessible."
        )
        st.stop()  # Stop execution

# Make a GET request to the FastAPI endpoint for the applicants and approval status
appl_response = requests.get(f"{fastapi_url}/get_applicants")

# Check if the request was successful (status code 200)
if appl_response.status_code == 200:
    applicants = appl_response.json()
else:
    # Handle the case where the request was not successful
    st.error(
        f"Failed to fetch data. Status code: {appl_response.status_code}. "
        + f"The endpoint '{fastapi_url}/get_applicants' is not accessible."
    )
    st.stop()  # Stop execution

# Configure page layout and tab
st.set_page_config(
    page_title="Prêt à dépenser - Outil d'évaluation des demandes de prêt",
    page_icon="🏦",
    initial_sidebar_state="expanded",
)

# Hide the decorative header bar
st.markdown("<style>header {visibility: hidden;}</style>", unsafe_allow_html=True)

# Set main div to plots' width
st.markdown(
    "<style>section.main > div {max-width:1030px}</style>", unsafe_allow_html=True
)

# Add page title
st.title("Outil d'évaluation des demandes de prêt")


# Clear plots if applicant changed
def appl_change():
    st.session_state["curr_plot"] = "none"


# Create a Streamlit dropdown with options in the desired format
st.selectbox(
    "Sélectionnez un numéro de demande de prêt :",
    [f"{key} : {value}" for key, value in applicants.items()],
    key="applicant",
    on_change=appl_change,
)

# Extract the selected ID number
selected_id = int(st.session_state["applicant"].split(":")[0].strip())

# Apply different text color based on approval status
approval_status = st.session_state["applicant"].split(":")[1].strip()

if approval_status == "Refusée":
    app_num_color = "lightcoral"
else:
    app_num_color = "cornflowerblue"

# Display the selected ID in large bold font with the specified text color
st.markdown(
    "<p style='font-size: 24px; font-weight: bold;'>Demande de prêt sélectionnée : "
    + f"<span style='color: {app_num_color};'>Nᵒ {selected_id} : {approval_status}</span></p>",
    unsafe_allow_html=True,
)

# Create a Streamlit sidebar
st.sidebar.title("Que souhaiteriez-vous savoir sur votre évaluation ?")

# Make sidebar buttons full width
st.markdown(
    "<style>section[data-testid='stSidebar'] div.stButton button { width: 100%; }</style>",
    unsafe_allow_html=True,
)


# Set current plot to global feature importance
def glob_clicked():
    # Update session state current plot
    st.session_state["curr_plot"] = "global"


# Add global feature importance button to the sidebar
if (
    st.sidebar.button("Importance globale des caractéristiques", on_click=glob_clicked)
    or st.session_state["curr_plot"] == "global"
):
    # Add a header
    st.header("Importance globale des caractéristiques")
    # Add a note about SHAP value meaning
    st.write(
        "Les valeurs positives signifient les facteurs qui sont en faveur d'approbation de la demande "
        + "de prêt, tandis que les valeurs négatives indiquent des facteurs qui contribuent au rejet. "
        + "La magnitude des barres correspond à leur impact sur la décision."
    )

    # Make a GET request to the FastAPI endpoint to get the PNG graph
    plot_response = requests.get(
        f"{fastapi_url}/plot_glob_feat_import?num_feats="
        + f"{st.session_state['slider']}"
    )

    # Check if the request was successful (status code 200)
    if plot_response.status_code == 200:
        # Display the image from the response
        st.image(plot_response.content)
    else:
        # Handle the case where the request was not successful
        st.error(
            f"Failed to fetch data. Status code: {plot_response.status_code}. "
            + f"The endpoint '{fastapi_url}/plot_glob_feat_import' is not accessible."
        )
        st.stop()  # Stop execution


# Set current plot to local feature importance
def loc_clicked():
    # Update session state current plot
    st.session_state["curr_plot"] = "local"


# Add local feature importance button to the sidebar
if (
    st.sidebar.button(
        "Probabilité d'approbation et l'importance de mes caractéristiques",
        on_click=loc_clicked,
    )
    or st.session_state["curr_plot"] == "local"
):
    # Make a GET request to the FastAPI endpoint to get the probability PNG
    prob_response = requests.get(
        f"{fastapi_url}/plot_approv_proba/?sk_id={selected_id}"
    )
    # Check whether the request was successful and stop execution if not
    if not prob_response.status_code == 200:
        st.error(
            f"Failed to fetch data. Status code: {prob_response.status_code}. "
            + f"The endpoint '{fastapi_url}/plot_approv_proba' is not accessible."
        )
        st.stop()  # Stop execution

    # Make a GET request to the FastAPI endpoint to get the local feature importance PNG
    local_feat_import_response = requests.get(
        f"{fastapi_url}/plot_local_feat_import/?sk_id={selected_id}&num_feats="
        + f"{st.session_state['slider']}"
    )
    # Check whether the request was successful and stop execution if not
    if not local_feat_import_response.status_code == 200:
        st.error(
            f"Failed to fetch data. Status code: {local_feat_import_response.status_code}. "
            + f"The endpoint '{fastapi_url}/plot_local_feat_import' is not accessible."
        )
        st.stop()  # Stop execution

    # Add a header
    st.header("Probabilité d'approbation")
    # Add a note about the approval probabililty bar
    st.write(
        "La partie de la barre de couleur chaude représente la probabilité du refus de votre demande "
        + "de prêt, tandis que la partie de couleur froide représente la probabilité de son approbation."
    )
    # Display the probability image from the response
    st.image(prob_response.content)

    # Add a header
    st.header("Importance de mes caractéristiques dans la décision")
    # Add a note about SHAP value meaning
    st.write(
        "Les valeurs positives signifient les facteurs qui sont en faveur d'approbation de la demande "
        + "de prêt, tandis que les valeurs négatives indiquent des facteurs qui contribuent au rejet. "
        + "La magnitude des barres correspond à leur impact sur la décision."
    )
    # Display the local feature importance image from the response
    st.image(local_feat_import_response.content)


# Set current plot to feature distributions
def dist_clicked():
    # Update session state current plot
    st.session_state["curr_plot"] = "distribs"


# Add feature distribution button to the sidebar
if (
    st.sidebar.button(
        "Mes caractéristiques par rapport à celles des autres demandeurs",
        on_click=dist_clicked,
    )
    or st.session_state["curr_plot"] == "distribs"
):
    # Make a GET request to the FastAPI endpoint to get the feature distribution PNG
    feat_distrib_response = requests.get(
        f"{fastapi_url}/plot_appl_features/?sk_id={selected_id}&num_feats="
        + f"{st.session_state['slider']}"
    )
    # Check whether the request was successful and stop execution if not
    if not feat_distrib_response.status_code == 200:
        st.error(
            f"Failed to fetch data. Status code: {feat_distrib_response.status_code}. "
            + f"The endpoint '{fastapi_url}/plot_appl_features' is not accessible."
        )
        st.stop()  # Stop execution

    # Add a header
    st.header("Comparaison de vos caractéristiques avec celles des demandeurs approuvés et refusés")
    # Add a note on how to interpret the feature distribution plots
    st.write(
        "Chaque graphique ci-dessous représente une caractéristique de votre profil financier. "
        + "La ligne bleue indique comment cette caractéristique est distribuée parmi les demandes "
        + "de prêt approuvées, tandis que la ligne rouge montre la distribution pour les demandes "
        + "refusées. La ligne verticale en tirets représente la valeur de votre propre "
        + "caractéristique. Si cette caractéristique a joué en votre défaveur, la ligne est en "
        + "rouge ; si elle a contribué favorablement à votre évaluation, elle est en bleu."
    )
    # Display the feature distribution plot from the response
    st.image(feat_distrib_response.content)

# Show the slider to select the number of features
st.sidebar.slider(
    "Nombre de caractéristiques",
    1,
    st.session_state["max_feats"],
    st.session_state["max_feats"] // 2,
    1,
    key="slider",
)
