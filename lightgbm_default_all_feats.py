import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from business_cost_metric import business_cost_metric
import datetime
import json
from make_train_valid_test import gen_train_valid_test

# Get start timestamp
start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Get preprocessed training set
preproc_app_train_df, _, _ = gen_train_valid_test(smote=True)

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
model = lgb.LGBMClassifier()

# Iterate through the KFold splits
for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Train the LightGBM model
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

    # Log the parameters, metrics and model artifact with mlflow
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", f"LightGBM ({start_time}) ({fold})")
        mlflow.log_param("params", "Default")
        mlflow.log_param("features", "All")
        mlflow.log_param("fold", str(fold))
        mlflow.log_metric("business_cost", business_cost)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.lightgbm.log_model(
            model, f"model_artifacts/lightgbm_default_all_feats/fold_{fold}"
        )
        mlflow.log_param("model_params", json.dumps(model.get_params()))

# Calculate the mean metrics across the 5 folds
mean_business_cost = np.mean(business_costs)
mean_accuracy = np.mean(accuracies)
mean_auc = np.mean(aucs)

# Log the mean metrics with mlflow
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", f"LightGBM ({start_time}) (avg)")
    mlflow.log_param("params", "Default")
    mlflow.log_param("features", "All")
    mlflow.log_param("fold", "Average")
    mlflow.log_metric("business_cost", mean_business_cost)
    mlflow.log_metric("accuracy", mean_accuracy)
    mlflow.log_metric("auc", mean_auc)
    mlflow.log_param("model_params", json.dumps(model.get_params()))
