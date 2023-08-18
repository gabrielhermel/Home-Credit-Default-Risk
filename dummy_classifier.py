import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from business_cost_metric import business_cost_metric
import datetime
import json
# from make_train_valid_test import gen_train_valid_test

# Get start timestamp
start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Get preprocessed training set
# preproc_app_train_df, _, _ = gen_train_valid_test(smote=True)
preproc_app_train_df = pd.read_feather("preproc_smote_app_train_df.ftr")

# Separate the features and target variable
X = preproc_app_train_df.drop(columns=["TARGET"])
y = preproc_app_train_df["TARGET"]

# Define Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Initialize metric lists to store the results for each fold
business_costs = []
accuracies = []
aucs = []

# Initialize model with default parameters
model = DummyClassifier(strategy="prior", random_state=0)

# Iterate through the KFold splits
for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    with mlflow.start_run(run_name=f"DummyClassifier") as run:
        # Set run tags
        mlflow.set_tags(
            {
                "param_tuning": "Default",
                "features": "All",
                "start_time": start_time,
                "fold": str(fold),
            }
        )

        # Log model parameters
        mlflow.log_params(json.dumps(model.get_params()))

        # Assign training and validation independent and target sets
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # Train the DummyClassifier
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_proba = model.predict_proba(X_valid)[:, 1]
        y_pred = model.predict(X_valid)

        # Calculate the metrics
        business_cost = business_cost_metric(y_valid, y_proba)
        accuracy = accuracy_score(y_valid, y_pred)
        auc = roc_auc_score(y_valid, y_proba)

        # Append the metrics to their respective lists
        business_costs.append(business_cost)
        accuracies.append(accuracy)
        aucs.append(auc)

        # Log metrics using the MLflow APIs
        mlflow.log_metrics(
            {"business_cost": business_cost, "accuracy": accuracy, "auc": auc}
        )

        # Log the sklearn model and register version
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-dummy-cls-model",
        )

# Log the mean metrics with mlflow
with mlflow.start_run(run_name=f"DummyClassifier") as run:
    mlflow.set_tags(
        {
            "param_tuning": "Default",
            "features": "All",
            "start_time": start_time,
            "fold": "Average",
        }
    )
    mlflow.log_params(json.dumps(model.get_params()))
    mlflow.log_metrics(
        {
            "mean_business_cost": np.mean(business_costs),
            "mean_accuracy": np.mean(accuracies),
            "mean_auc": np.mean(aucs),
        }
    )
