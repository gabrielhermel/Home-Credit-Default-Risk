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
# Slider value for max number of features
if "max_feats" not in st.session_state:
    # Make a GET request to the FastAPI endpoint for the applicants and approval status
    num_feats_response = requests.get(f"{fastapi_url}/get_num_feats")

    # Check if the request was successful (status code 200)
    if num_feats_response.status_code == 200:
        # Initialize session_state variables
        st.session_state["max_feats"] = num_feats_response.json()
        st.session_state["num_feats"] = st.session_state["max_feats"] // 2
        st.session_state["show_slider"] = False  # Initialize with the slider hidden
        st.session_state["slider"] = st.session_state[
            "num_feats"
        ]  # Initialize slider value
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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide the decorative header bar
hide_decoration_bar_style = """
    <style>
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# Add page title
st.title("Outil d'évaluation des demandes de prêt")

# Create a Streamlit dropdown with options in the desired format
selected_item = st.selectbox(
    "Sélectionnez un numéro de demande de prêt :",
    [f"{key} : {value}" for key, value in applicants.items()],
)

# Extract the selected ID number
selected_id = int(selected_item.split(":")[0].strip())

# Apply different text color based on approval status
approval_status = selected_item.split(":")[1].strip()
if approval_status == "Refusée":
    app_num_color = "lightcoral"
else:
    app_num_color = "cornflowerblue"

# Display the selected ID in large bold font with the specified text color
st.markdown(
    "<p style='font-size: 24px; font-weight: bold;'>Demande de prêt sélectionnée : "
    + f"<span style='color: {app_num_color};'>Nᵒ {selected_id}</span></p>",
    unsafe_allow_html=True,
)


# Function to display feature distribution based on selected num_feats
def display_feat_distrib():
    # Update session state num_feats
    st.session_state["num_feats"] = st.session_state["slider"]

    # Make a GET request to the FastAPI endpoint to get the feature distribution PNG
    feat_distrib_response = requests.get(
        f"{fastapi_url}/plot_appl_features/?sk_id={selected_id}&num_feats="
        + f"{st.session_state['num_feats']}"
    )
    # Check whether the request was successful and stop execution if not
    if not feat_distrib_response.status_code == 200:
        st.error(
            f"Failed to fetch data. Status code: {feat_distrib_response.status_code}. "
            + f"The endpoint '{fastapi_url}/plot_appl_features' is not accessible."
        )
        st.stop()  # Stop execution

    # Add a header
    st.header("Comparaison de vos caractéristiques avec les prêts approuvés et refusés")
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


# Check whether any sidebar buttons other than for feature distribution have been clicked;
# if not, and the show_slider flag is set, then display feature distributions
if (
    st.session_state["show_slider"]
    and not st.session_state["glob_btn"]
    and not st.session_state["prob_btn"]
):
    display_feat_distrib()

# Create a Streamlit sidebar
st.sidebar.title("Que souhaiteriez-vous savoir sur votre évaluation ?")

# Add global feature importance button to the sidebar
if st.sidebar.button("Importance globale des caractéristiques", key="glob_btn"):
    # Hide slider
    st.session_state["show_slider"] = False

    # Add a header
    st.header("Importance globale des caractéristiques")
    # Add a note about SHAP value meaning
    st.write(
        "Les valeurs positives signifient les facteurs qui sont en faveur d'approbation de la demande "
        + "de prêt, tandis que les valeurs négatives indiquent des facteurs qui contribuent au rejet. "
        + "La magnitude des barres correspond à leur impact sur la décision."
    )

    # Make a GET request to the FastAPI endpoint to get the PNG graph
    plot_response = requests.get(f"{fastapi_url}/plot_glob_feat_import")

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

# Add local feature importance button to the sidebar
if st.sidebar.button(
    "Probabilité d'approbation et l'importance de mes caractéristiques", key="prob_btn"
):
    # Hide slider
    st.session_state["show_slider"] = False

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
        f"{fastapi_url}/plot_local_feat_import/?sk_id={selected_id}"
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
        "La partie rouge de la barre représente la probabilité du refus de votre demande de prêt, "
        + "tandis que la partie bleue représente la probabilité de son approbation."
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

# Add feature distribution button to the sidebar
if st.sidebar.button("Distribution des caractéristiques par rapport aux miennes"):
    # Show slider
    st.session_state["show_slider"] = True
    # Call function to display feature distribution plot
    display_feat_distrib()

# Show the slider to select the number of features if the feature distribution button has been clicked
# (and no other sidebar buttons have been clicked more recently)
if st.session_state["show_slider"]:
    st.sidebar.slider(
        "Nombre de caractéristiques",
        1,
        st.session_state["max_feats"],
        st.session_state["num_feats"],
        1,
        key="slider",
    )
