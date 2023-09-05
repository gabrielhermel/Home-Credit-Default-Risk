#!/usr/bin/env python

from fastapi import FastAPI, HTTPException
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# Import Matplotlib and set a non-interactive backend
# (necessary for pytest unit tests so GUI runs in main thread)
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import seaborn as sns
from io import BytesIO
import pandas as pd
import numpy as np
import shap
import pickle
import json
import os

# Define a constant for the path to data directory
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + os.sep

# Define paths to various data files and ensure they exist
# Path to a CSV file containing demo applicant data
demo_applicants_path = os.path.join(DATA_PATH, "demo_applicants.csv")
if not os.path.exists(demo_applicants_path):
    raise FileNotFoundError(f"{demo_applicants_path} not found")

# Load demo applicant data into a Pandas DataFrame
demo_applicants = pd.read_csv(demo_applicants_path, dtype={"SK_ID_CURR": int})

# Repeat the above steps for other data files
# The independant variables used to train the model
X_train_path = os.path.join(DATA_PATH, "X_train.csv")
if not os.path.exists(X_train_path):
    raise FileNotFoundError(f"{X_train_path} not found")

X_train = pd.read_csv(X_train_path)

# A sample of applicants whose loans were approved
approved_sample_path = os.path.join(DATA_PATH, "approved_sample.csv")
if not os.path.exists(approved_sample_path):
    raise FileNotFoundError(f"{approved_sample_path} not found")

approved_sample = pd.read_csv(approved_sample_path)

# A sample of applicants whose loans were denied
denied_sample_path = os.path.join(DATA_PATH, "denied_sample.csv")
if not os.path.exists(denied_sample_path):
    raise FileNotFoundError(f"{denied_sample_path} not found")

denied_sample = pd.read_csv(denied_sample_path)

# The global importance of features based on SHAP values
glob_feat_import_path = os.path.join(DATA_PATH, "glob_feat_import.json")
if not os.path.exists(glob_feat_import_path):
    raise FileNotFoundError(f"{glob_feat_import_path} not found")

with open(glob_feat_import_path, "r") as f:
    glob_feat_import = json.load(f)

# Confirm banary classification model path is valid and load it
model_path = os.path.join(DATA_PATH, "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Seperate applicant SK_IDs from their independant features
SK_ID_CURR_list = demo_applicants["SK_ID_CURR"].astype(int).tolist()
demo_applicants = demo_applicants.drop(columns="SK_ID_CURR")
# Predict dependant variables for all applicants and save to a list
TARGET_list = list(map(int, model.predict(demo_applicants).tolist()))
# Create a dictionary with the applicant's IDs as keys and their approval status as values
ids_and_approv = result_dict = {
    SK_ID: "Approuvée" if TARGET == 1 else "Refusée"
    for SK_ID, TARGET in zip(SK_ID_CURR_list, TARGET_list)
}

# Create a FastAPI instance
app = FastAPI()


# Define a function to compute local feature importances for a given applicant
def compute_local_feat_import(row_series):
    # Define an inner function to make predictions using the model
    def pipeline_predict(data):
        return model.predict_proba(data)[:, 1]  # Predict the probability of class 1

    # Create a SHAP (SHapley Additive exPlanations) explainer for the model
    explainer = shap.Explainer(pipeline_predict, X_train)
    # Calculate SHAP values for the given row of feature values
    shap_values = explainer.shap_values(row_series.values.reshape(1, -1))
    # Get the feature names as a list
    feature_names = row_series.index.to_list()
    # Scale the SHAP values to ensure they are comparable
    scaled_shap_values = shap_values / np.max(np.abs(shap_values))
    # Create a dictionary associating feature names with their scaled SHAP values
    feats_and_vals = {
        feature_names[i]: scaled_shap_values[0][i] for i in range(len(feature_names))
    }
    # Sort the features based on their scaled SHAP values in descending order
    sorted_feats = dict(
        sorted(feats_and_vals.items(), key=lambda item: item[1], reverse=True)
    )

    # Return the sorted feature importances for the given applicant
    return sorted_feats


# Define an API endpoint to retrieve the number of features used in predictions
@app.get("/get_num_feats/")
def get_num_feats():
    return len(X_train.columns.to_list())


# Define an API endpoint to retrieve a list of applicants and their approval statuses
@app.get("/get_applicants/")
def get_applicants():
    return ids_and_approv


# Define an API endpoint to retrieve the features of a specific applicant by SK_ID_CURR
@app.get("/get_applicant_feats/")
def get_applicant_feats(sk_id: int):
    # Check if the provided SK_ID_CURR exists in the list of applicant IDs
    if sk_id not in SK_ID_CURR_list:
        # If the applicant is not found, raise an HTTP exception with a 404 status code
        raise HTTPException(
            status_code=404, detail=f"Applicant with SK_ID {sk_id} not found."
        )

    # Retrieve and return the features of the specified applicant as a dictionary
    return demo_applicants.iloc[SK_ID_CURR_list.index(sk_id)].to_dict()


# Define an API endpoint to create and return a global feature importances plot
@app.get("/plot_glob_feat_import/")
async def plot_glob_feat_import():
    # Set the Seaborn theme
    sns.set(style="darkgrid")

    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(10, 6))

    # Create a bar plot using Seaborn, with feature importances as the x-axis and feature names as the y-axis
    ax = sns.barplot(
        x=list(glob_feat_import.values()),  # Feature importances as x-axis values
        y=list(glob_feat_import.keys()),  # Feature names as y-axis labels
        palette="coolwarm",  # Color palette for the bars
    )

    # Set the x-axis limits to control the plot's horizontal range
    plt.xlim(left=-1.0, right=1.0)

    # Add text labels to the bars indicating the feature importances
    for i, value in enumerate(glob_feat_import.values()):
        text_x = (
            -0.14 if value >= 0 else 0.16
        )  # Adjust text position based on the value
        ax.text(
            text_x,
            i,
            f"{value:.3f}",  # Display the importance value with 3 decimal places
            va="center",
            ha="left"
            if value >= 0
            else "right",  # Adjust text alignment based on the value
            fontsize=10,
            color="black",
        )

    # Ensure a tight layout for the plot
    plt.tight_layout()

    # Create a buffer to store the plot image as bytes
    buffer = BytesIO()

    # Save the plot as a PNG image to the buffer
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    # Return the plot as a streaming response with the media type "image/png"
    return StreamingResponse(buffer, media_type="image/png")


# Define an API endpoint to create and return a local feature importances plot for a specific applicant
@app.get("/plot_local_feat_import/")
async def plot_local_feat_import(sk_id: int):
    # Set the Seaborn theme
    sns.set(style="darkgrid")

    # Check if the provided SK_ID_CURR exists in the list of applicant IDs
    if sk_id not in SK_ID_CURR_list:
        # If the applicant is not found, raise an HTTP exception with a 404 status code
        raise HTTPException(
            status_code=404, detail=f"Applicant with SK_ID {sk_id} not found."
        )

    # Retrieve the features of the specified applicant using their SK_ID_CURR
    applicant_feats = demo_applicants.iloc[SK_ID_CURR_list.index(sk_id)]

    # Compute local feature importances for the applicant using the compute_local_feat_import function
    local_feat_import = compute_local_feat_import(applicant_feats)

    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(10, 6))

    # Create a bar plot using Seaborn, with feature importances as the x-axis and feature names as the y-axis
    ax = sns.barplot(
        x=list(local_feat_import.values()),  # Feature importances as x-axis values
        y=list(local_feat_import.keys()),  # Feature names as y-axis labels
        palette="coolwarm",  # Color palette for the bars
    )

    # Set the x-axis limits to control the plot's horizontal range
    plt.xlim(left=-1.0, right=1.0)

    # Add text labels to the bars indicating the feature importances
    for i, value in enumerate(local_feat_import.values()):
        text_x = (
            -0.14 if value >= 0 else 0.16
        )  # Adjust text position based on the value
        ax.text(
            text_x,
            i,
            f"{value:.3f}",  # Display the importance value with 3 decimal places
            va="center",
            ha="left"
            if value >= 0
            else "right",  # Adjust text alignment based on the value
            fontsize=10,
            color="black",
        )

    # Ensure a tight layout for the plot
    plt.tight_layout()

    # Create a buffer to store the plot image as bytes
    buffer = BytesIO()

    # Save the plot as a PNG image to the buffer
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    # Return the plot as a streaming response with the media type "image/png"
    return StreamingResponse(buffer, media_type="image/png")


# Define an API endpoint to create and return a plot showing the approval probability for a specific applicant
@app.get("/plot_approv_proba/")
async def plot_approv_proba(sk_id: int):
    # Set the Seaborn theme
    sns.set(style="darkgrid")

    # Check if the provided SK_ID_CURR exists in the list of applicant IDs
    if sk_id not in SK_ID_CURR_list:
        # If the applicant is not found, raise an HTTP exception with a 404 status code
        raise HTTPException(
            status_code=404, detail=f"Applicant with SK_ID {sk_id} not found."
        )

    # Predict the approval probabilities for the specified applicant
    probabilities = model.predict_proba(
        demo_applicants.loc[[SK_ID_CURR_list.index(sk_id)]]
    )

    # Extract the probabilities for class 0 and class 1
    prob_class_0 = probabilities[0][0]
    prob_class_1 = probabilities[0][1]

    # Calculate the percentages of class 0 and class 1 probabilities
    percent_class_0 = prob_class_0 * 100
    percent_class_1 = prob_class_1 * 100

    # Create a figure and axis for the plot with a specified size
    fig, ax = plt.subplots(figsize=(6, 0.8))

    # Define colors for the bars using a coolwarm colormap
    colors = plt.cm.coolwarm([0.95, 0.05])

    # Define bar positions and widths for class 0 and class 1 probabilities
    bar_positions = [0]
    bar_widths = [percent_class_0]

    # Create a horizontal bar for class 0 probability
    ax.barh(
        bar_positions,
        bar_widths,
        color=colors[0],
        label=f"{percent_class_0:.2f}% Class 0",
    )

    # Define bar positions and widths for class 1 probability, offset from class 0
    bar_positions = [0]
    bar_widths = [percent_class_1]

    # Create a horizontal bar for class 1 probability
    ax.barh(
        bar_positions,
        bar_widths,
        left=[percent_class_0],  # Position it to the right of class 0 bar
        color=colors[1],
        label=f"{percent_class_1:.2f}% Class 1",
    )

    # Customize the appearance of the plot
    # Set y-ticks and labels
    ax.set_yticks([0])
    ax.set_yticklabels([])

    # Add text labels for class 0 and class 1 percentages
    ax.text(0, -0.9, f"{percent_class_0:.2f}%", ha="left", color="black")
    ax.text(100, -0.9, f"{percent_class_1:.2f}%", ha="right", color="black")

    # Set x-axis limits and y-axis limits
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 0.5)

    # Hide unnecessary spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])

    # Ensure a tight layout for the plot
    plt.tight_layout()

    # Create a buffer to store the plot image as bytes
    buffer = BytesIO()

    # Save the plot as a PNG image to the buffer
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    # Return the plot as a streaming response with the media type "image/png"
    return StreamingResponse(buffer, media_type="image/png")


# Define an API endpoint to create and return a plot of applicant-specific features
@app.get("/plot_appl_features/")
async def plot_appl_features(sk_id: int, num_feats: int):
    # Set the Seaborn theme
    sns.set(style="darkgrid")

    # Check if the provided SK_ID_CURR exists in the list of applicant IDs
    if sk_id not in SK_ID_CURR_list:
        # If the applicant is not found, raise an HTTP exception with a 404 status code
        raise HTTPException(
            status_code=404, detail=f"Applicant with SK_ID {sk_id} not found."
        )

    # Check if the provided number of features is valid
    if num_feats < 0 or num_feats > len(demo_applicants.columns):
        # If the number of features is invalid, raise an HTTP exception with a 404 status code
        raise HTTPException(
            status_code=404, detail=f"{num_feats} is not a valid number of features."
        )

    # Retrieve the feature values for the specified applicant using their SK_ID_CURR
    row_series = demo_applicants.iloc[SK_ID_CURR_list.index(sk_id)]
    # Compute local feature importances for the applicant using the compute_local_feat_import function
    local_feat_import = compute_local_feat_import(row_series)
    # Sort the feature names based on their absolute local feature importances in descending order
    sorted_feat_import = sorted(
        local_feat_import.keys(), key=lambda x: abs(local_feat_import[x]), reverse=True
    )

    # Set the color palette for Seaborn plots
    sns.set_palette("Spectral")

    # Create a figure and axes for the plot based on the number of requested features
    if num_feats == 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Select the first feature from the sorted list
        column = sorted_feat_import[0]

        # Plot kernel density estimations (KDE) for the selected feature for approved and denied samples
        sns.kdeplot(
            approved_sample[column], ax=ax, label="Approuvée", color="cornflowerblue"
        )
        sns.kdeplot(denied_sample[column], ax=ax, label="Refusée", color="lightcoral")

        # Get the value of the selected feature for the applicant
        client_value = row_series[column]
        # Determine the line color based on the local feature importance of the feature
        line_color = (
            "lightcoral" if local_feat_import[column] <= 0 else "cornflowerblue"
        )

        # Add a vertical dashed line indicating the value of the feature for the applicant
        ax.axvline(x=client_value, color=line_color, linestyle="--", label="Client")
        # Y label in French
        ax.set_ylabel("Densité")
        # Set the title and legend for the plot
        ax.set_title(column)
        ax.legend()
    # If more than one feature has been requested
    else:
        num_cols = min(4, num_feats)
        num_rows = (num_feats - 1) // num_cols + 1
        # Create subplots for multiple features in a grid layout
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 3 * num_rows)
        )

        # Iterate through the requested number of features
        for i in range(num_feats):
            row_n = i // num_cols
            col_n = i % num_cols
            column = sorted_feat_import[i]

            if num_rows == 1:
                ax = axes[col_n]
            else:
                ax = axes[row_n, col_n]

            # Plot KDEs for the selected feature for approved and denied samples
            sns.kdeplot(
                approved_sample[column],
                ax=ax,
                label="Approuvée",
                color="cornflowerblue",
            )
            sns.kdeplot(
                denied_sample[column], ax=ax, label="Refusée", color="lightcoral"
            )

            # Get the value of the selected feature for the applicant
            client_value = row_series[column]
            # Determine the line color based on the local feature importance of the feature
            line_color = (
                "lightcoral" if local_feat_import[column] <= 0 else "cornflowerblue"
            )

            # Add a vertical dashed line indicating the value of the feature for the applicant
            ax.axvline(x=client_value, color=line_color, linestyle="--", label="Client")
            # Y label in French
            ax.set_ylabel("Densité")
            # Set the title and legend for each subplot
            ax.set_title(column)
            ax.legend()

        # Remove any empty subplots if the number of requested features is less than the grid size
        for i in range(num_feats, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

    # Ensure a tight layout for the plot
    plt.tight_layout()

    # Create a buffer to store the plot image as bytes
    buffer = BytesIO()

    # Save the plot as a PNG image to the buffer
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    # Return the plot as a streaming response with the media type "image/png"
    return StreamingResponse(buffer, media_type="image/png")
