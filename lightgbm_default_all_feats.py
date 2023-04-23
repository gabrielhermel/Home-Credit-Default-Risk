import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from business_cost_metric import business_cost_metric

# Read the Feather file into a DataFrame
df = pd.read_feather("engin_feats.ftr")

# Remove index column
if "index" in df.columns:
    df = df.drop(columns=["index"])

# Filter out rows with null values in the "TARGET" column
df = df[df["TARGET"].notnull()].reset_index()

# Separate the features and target variable
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize metric lists to store the results for each fold
business_costs = []
accuracies = []
aucs = []

# Iterate through the KFold splits
for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Train the LightGBM model with default parameters
    model = lgb.LGBMClassifier()
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

    # Log the metrics and model artifact with mlflow
    with mlflow.start_run():
        mlflow.log_param("model_type", "LightGBM - Default Parameters - All Features")
        mlflow.log_param("fold", fold)
        mlflow.log_metric("business_cost", business_cost)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.lightgbm.log_model(
            model, f"model_artifacts/lightgbm_default_all_feats/fold_{fold}"
        )

# Calculate the mean metrics across the 5 folds
mean_business_cost = np.mean(business_costs)
mean_accuracy = np.mean(accuracies)
mean_auc = np.mean(aucs)

# Log the mean metrics with mlflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LightGBM - Default Parameters - All Features")
    mlflow.log_metric("mean_business_cost", mean_business_cost)
    mlflow.log_metric("mean_accuracy", mean_accuracy)
    mlflow.log_metric("mean_auc", mean_auc)
