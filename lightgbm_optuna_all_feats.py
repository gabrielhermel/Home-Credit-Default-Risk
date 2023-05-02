import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score  # , make_scorer
from business_cost_metric import business_cost_metric
import optuna
from optuna.samplers import TPESampler


# Custom scorer for business cost metric
# business_cost_scorer = make_scorer(
#     business_cost_metric, greater_is_better=False, needs_proba=True
# )
def custom_eval_function(y_pred, data):
    y_true = data.get_label()
    score = business_cost_metric(y_true, y_pred)
    return "business_cost", score, False


def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 80),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
    }
    n_estimators = trial.suggest_int("n_estimators", 100, 500)

    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=n_estimators,
        valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)],
        # feval=business_cost_scorer,
        feval=custom_eval_function,
    )

    trial.set_user_attr("num_boost_round", model.best_iteration)
    y_proba = model.predict(X_valid, num_iteration=model.best_iteration)
    business_cost = business_cost_metric(y_valid, y_proba)

    return business_cost


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
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize metric lists to store the results for each fold
business_costs = []
accuracies = []
aucs = []

# Iterate through the KFold splits
for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Train the LightGBM model with hyperparameters tuned by Optuna
    study = optuna.create_study(direction="minimize", sampler=TPESampler())
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=20,
        n_jobs=2,
    )
    best_params = study.best_params
    # Retrieve the best number of boosting rounds from the best trial
    best_num_boost_round = study.best_trial.user_attrs["num_boost_round"]

    # Train the model with the best hyperparameters
    model = lgb.train(
        best_params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=best_num_boost_round,
        valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)],
        # feval=business_cost_scorer,
        feval=custom_eval_function,
    )
    y_proba = model.predict(X_valid, num_iteration=model.best_iteration)
    y_pred = np.round(y_proba).astype(int)

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
        mlflow.log_param("model_type", "LightGBM - Optuna Tuned - All Features")
        mlflow.log_param("fold", fold)
        mlflow.log_metric("business_cost", business_cost)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.lightgbm.log_model(
            model, f"model_artifacts/lightgbm_optuna_tuned_all_feats/fold_{fold}"
        )

# Calculate the mean metrics across the 5 folds
mean_business_cost = np.mean(business_costs)
mean_accuracy = np.mean(accuracies)
mean_auc = np.mean(aucs)

# Log the mean metrics with mlflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LightGBM - Optuna Tuned - All Features")
    mlflow.log_metric("mean_business_cost", mean_business_cost)
    mlflow.log_metric("mean_accuracy", mean_accuracy)
    mlflow.log_metric("mean_auc", mean_auc)
