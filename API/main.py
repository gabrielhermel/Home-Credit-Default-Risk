#!/usr/bin/env python

from fastapi import FastAPI, HTTPException
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from matplotlib import pyplot as plt
import seaborn as sns
from io import BytesIO
import pandas as pd
import numpy as np
import shap
import pickle
import json
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + os.sep

demo_applicants_path = os.path.join(DATA_PATH, "demo_applicants.csv")
if not os.path.exists(demo_applicants_path):
    raise FileNotFoundError(f"{demo_applicants_path} not found")

demo_applicants = pd.read_csv(demo_applicants_path, dtype={"SK_ID_CURR": int})

X_train_path = os.path.join(DATA_PATH, "X_train.csv")
if not os.path.exists(X_train_path):
    raise FileNotFoundError(f"{X_train_path} not found")

X_train = pd.read_csv(X_train_path)

approved_sample_path = os.path.join(DATA_PATH, "approved_sample.csv")
if not os.path.exists(approved_sample_path):
    raise FileNotFoundError(f"{approved_sample_path} not found")

approved_sample = pd.read_csv(approved_sample_path)

denied_sample_path = os.path.join(DATA_PATH, "denied_sample.csv")
if not os.path.exists(denied_sample_path):
    raise FileNotFoundError(f"{denied_sample_path} not found")

denied_sample = pd.read_csv(denied_sample_path)

glob_feat_import_path = os.path.join(DATA_PATH, "glob_feat_import.json")
if not os.path.exists(glob_feat_import_path):
    raise FileNotFoundError(f"{glob_feat_import_path} not found")

with open(glob_feat_import_path, "r") as f:
    glob_feat_import = json.load(f)

model_path = os.path.join(DATA_PATH, "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found")

with open(model_path, "rb") as f:
    model = pickle.load(f)

SK_ID_CURR_list = demo_applicants["SK_ID_CURR"].astype(int).tolist()
demo_applicants = demo_applicants.drop(columns="SK_ID_CURR")
TARGET_list = list(map(int, model.predict(demo_applicants).tolist()))
ids_and_approv = result_dict = {
    SK_ID: "Approuvée" if TARGET == 1 else "Refusée"
    for SK_ID, TARGET in zip(SK_ID_CURR_list, TARGET_list)
}

app = FastAPI()


def compute_local_feat_import(row_series):
    def pipeline_predict(data):
        return model.predict_proba(data)[:, 1]

    explainer = shap.Explainer(pipeline_predict, X_train)
    shap_values = explainer.shap_values(row_series.values.reshape(1, -1))
    feature_names = row_series.index.to_list()
    scaled_shap_values = shap_values / np.max(np.abs(shap_values))
    feats_and_vals = {
        feature_names[i]: scaled_shap_values[0][i] for i in range(len(feature_names))
    }
    sorted_feats = dict(
        sorted(feats_and_vals.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_feats


@app.get("/get_applicants/")
def get_applicants():
    return ids_and_approv


@app.get("/get_applicant_feats/")
def get_applicant_feats(sk_id: int):
    if sk_id not in SK_ID_CURR_list:
        raise HTTPException(
            status_code=404, detail=f"Applicant with SK_ID {sk_id} not found."
        )

    return demo_applicants.iloc[SK_ID_CURR_list.index(sk_id)].to_dict()


@app.get("/plot_glob_feat_import/")
async def plot_glob_feat_import():
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=list(glob_feat_import.values()),
        y=list(glob_feat_import.keys()),
        palette="coolwarm",
    )
    plt.xlim(left=-1.0, right=1.0)

    for i, value in enumerate(glob_feat_import.values()):
        text_x = -0.14 if value >= 0 else 0.16
        ax.text(
            text_x,
            i,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            color="black",
        )

    plt.tight_layout()

    buffer = BytesIO()

    plt.savefig(buffer, format="png")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


@app.get("/plot_local_feat_import/")
async def plot_local_feat_import(sk_id: int):
    if sk_id not in SK_ID_CURR_list:
        raise HTTPException(
            status_code=404, detail=f"Applicant with SK_ID {sk_id} not found."
        )

    applicant_feats = demo_applicants.iloc[SK_ID_CURR_list.index(sk_id)]
    local_feat_import = compute_local_feat_import(applicant_feats)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=list(local_feat_import.values()),
        y=list(local_feat_import.keys()),
        palette="coolwarm",
    )
    plt.xlim(left=-1.0, right=1.0)

    for i, value in enumerate(local_feat_import.values()):
        text_x = -0.14 if value >= 0 else 0.16
        ax.text(
            text_x,
            i,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            color="black",
        )

    plt.tight_layout()

    buffer = BytesIO()

    plt.savefig(buffer, format="png")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


@app.get("/plot_approv_proba/")
async def plot_approv_proba(sk_id: int):
    if sk_id not in SK_ID_CURR_list:
        raise HTTPException(
            status_code=404, detail=f"Applicant with SK_ID {sk_id} not found."
        )

    probabilities = model.predict_proba(
        demo_applicants.loc[[SK_ID_CURR_list.index(sk_id)]]
    )

    prob_class_0 = probabilities[0][0]
    prob_class_1 = probabilities[0][1]

    percent_class_0 = prob_class_0 * 100
    percent_class_1 = prob_class_1 * 100

    fig, ax = plt.subplots(figsize=(6, 0.7))

    colors = plt.cm.coolwarm([0.95, 0.05])

    bar_positions = [0]
    bar_widths = [percent_class_0]
    ax.barh(
        bar_positions,
        bar_widths,
        color=colors[0],
        label=f"{percent_class_0:.2f}% Class 0",
    )

    bar_positions = [0]
    bar_widths = [percent_class_1]
    ax.barh(
        bar_positions,
        bar_widths,
        left=[percent_class_0],
        color=colors[1],
        label=f"{percent_class_1:.2f}% Class 1",
    )

    ax.set_yticks([0])
    ax.set_yticklabels([])

    ax.text(0, -0.9, f"{percent_class_0:.2f}%", ha="left", color="black")
    ax.text(100, -0.9, f"{percent_class_1:.2f}%", ha="right", color="black")

    ax.set_xlim(0, 100)

    ax.set_ylim(-1, 0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])

    plt.tight_layout()

    buffer = BytesIO()

    plt.savefig(buffer, format="png")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


@app.get("/plot_appl_features/")
async def plot_appl_features(sk_id: int, num_feats: int):
    if sk_id not in SK_ID_CURR_list:
        raise HTTPException(
            status_code=404, detail=f"Applicant with SK_ID {sk_id} not found."
        )
    if num_feats < 0 or num_feats > len(demo_applicants.columns):
        raise HTTPException(
            status_code=404, detail=f"{num_feats} is not a valid number of features."
        )

    row_series = demo_applicants.iloc[SK_ID_CURR_list.index(sk_id)]
    local_feat_import = compute_local_feat_import(row_series)

    sorted_feat_import = sorted(
        local_feat_import.keys(), key=lambda x: abs(local_feat_import[x]), reverse=True
    )

    sns.set_palette("Spectral")

    if num_feats == 1:
        fig, ax = plt.subplots(figsize=(6, 4))

        column = sorted_feat_import[0]

        sns.kdeplot(
            approved_sample[column], ax=ax, label="Approuvée", color="cornflowerblue"
        )
        sns.kdeplot(denied_sample[column], ax=ax, label="Refusée", color="lightcoral")

        client_value = row_series[column]
        line_color = (
            "lightcoral" if local_feat_import[column] <= 0 else "cornflowerblue"
        )
        ax.axvline(x=client_value, color=line_color, linestyle="--", label="Client")

        ax.set_title(column)
        ax.legend()
    else:
        num_cols = min(4, num_feats)
        num_rows = (num_feats - 1) // num_cols + 1

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 3 * num_rows)
        )

        for i in range(num_feats):
            row_n = i // num_cols
            col_n = i % num_cols
            column = sorted_feat_import[i]

            if num_rows == 1:
                ax = axes[col_n]
            else:
                ax = axes[row_n, col_n]

            sns.kdeplot(
                approved_sample[column],
                ax=ax,
                label="Approuvée",
                color="cornflowerblue",
            )
            sns.kdeplot(
                denied_sample[column], ax=ax, label="Refusée", color="lightcoral"
            )

            client_value = row_series[column]
            line_color = (
                "lightcoral" if local_feat_import[column] <= 0 else "cornflowerblue"
            )
            ax.axvline(x=client_value, color=line_color, linestyle="--", label="Client")

            ax.set_title(column)
            ax.legend()

        for i in range(num_feats, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

    plt.tight_layout()

    buffer = BytesIO()

    plt.savefig(buffer, format="png")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
