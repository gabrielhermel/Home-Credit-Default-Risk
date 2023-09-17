# Credit Scoring Model and Dashboard

## Project Overview
To address the increasing demands from clients for transparency in credit decisions, this project aims to implement a credit scoring tool. This tool predicts the likelihood of a client repaying their loan, subsequently classifying the request as approved or denied. The classification algorithm leverages diverse data sources, including behavioral data and data from other financial institutions.

Furthermore, to uphold the company's values of transparency, an interactive dashboard is developed for client relationship managers. The scoring model is deployed through an API, and the dashboard interfaces with this API to obtain and visualize predictions. This dashboard facilitates a transparent explanation of credit decisions, while also allowing clients to access and explore their personal data seamlessly.

## Project Objectives
1. Develop an automated credit scoring model to predict the probability of a client defaulting on their loan.
2. Construct an interactive dashboard for client relationship managers, aiding in the interpretation of model predictions and enhancing client understanding.
3. Deploy the predictive scoring model via an API, from which the interactive dashboard will retrieve predictions.

## Directory Structure and Descriptions

### API (`/api/`)

- `data/`: Contains data files utilized by the API.
  - `approved_sample.csv`: Sample of approved applicants for feature KDE plots generation.
  - `demo_applicants.csv`: Sample to showcase API and dashboard functionality.
  - `denied_sample.csv`: Sample of denied applicants for feature KDE plots generation.
  - `glob_feat_import.json`: Global feature importance metrics for the archived model using Max-abs scaled SHAP values.
  - `model.pkl`: The trained model used by the API for predictions.
  - `X_train.csv`: Training dataset for the model, also used for SHAP explainer creation.
- `main.py`: Contains the FastAPI application.
- `Procfile`: Command to launch the API on the Heroku server.
- `requirements.txt`: PIP dependencies for the API.
- `runtime.txt`: Specifies the Python version for the Heroku server.
- `test_api.py`: Pre-deployment testing script using pytest.

### Business Cost Metric

-`/business_cost_metric.py`: Module for the custom business cost metric used in model optimization.

### Dashboard (`/dashboard/`)

- `config.yaml`: Contains the API URL.
- `dashboard.py`: The Streamlit application for the dashboard.
- `Procfile`: Command to launch the dashboard on Heroku.
- `requirements.txt`: PIP dependencies for the dashboard.
- `runtime.txt`: Specifies the Python version for the Heroku server.
- `.streamlit/config.toml`: Theming and server configuration for the Streamlit dashboard.

### Data Drift Report

- `/data_drift_report.html`: Evidently report detailing potential data drift.

### GitHub Actions (`/.github/workflows/`)

- `deploy_to_heroku.yaml`: CI/CD setup for pytest runs and deployment of the API and dashboard to Heroku.

### Repository Configuration

- `/.gitignore`: Gitignore file, ignoring MLFlow runs specifically.

### Home Credit Default Risk Data (`/input/`)

- Directory for data required by `modeling.ipynb`. Must include:
  - `application_test.csv`
  - `application_train.csv`
  - `bureau_balance.csv`
  - `bureau.csv`
  - `credit_card_balance.csv`
  - `installments_payments.csv`
  - `POS_CASH_balance.csv`
  - `previous_application.csv`
- Not included in repository; must be created and populated to run `modeling.ipynb`.

### Methodological Note

- `/methodological_note.pdf`: A comprehensive methodological note for the project.

### MLflow Archive (`/mlruns/0/`)

- Directory containing archived MLFlow runs.
- Not included in repository; will be generated when executing `/modeling.ipynb`.

### Modeling

- `/modeling.ipynb`: Jupyter notebook performing feature engineering, feature selection, modeling, and logging/archiving processes.

### Root Dependencies

- `/requirements.txt`: PIP dependencies for the modeling notebook and the business cost metric module.