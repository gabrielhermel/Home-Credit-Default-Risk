# API for Credit Scoring Model

## Overview
This API is the core backend for the credit scoring model. Its main function is to predict the likelihood of a client defaulting on their loan based on the applicant's feature values. The interactive dashboard, located in a separate directory, calls this API to obtain and visualize predictions. The API is deployed on a cloud platform. In this case, Heroku was chosen.

## Directory Structure

- `data/`: Holds data files essential to the API's operation.
  - `approved_sample.csv`: Sample of approved applicants used for generating KDE plots.
  - `demo_applicants.csv`: Sample to demonstrate API functionality.
  - `denied_sample.csv`: Sample of denied applicants used for KDE plots generation.
  - `glob_feat_import.json`: Global feature importance using Max-abs scaled SHAP values.
  - `model.pkl`: Trained model employed by the API for predictions.
  - `X_train.csv`: Dataset the model was trained on; used for local SHAP explainer creation.
- `main.py`: Contains the FastAPI application.
- `Procfile`: Command to initiate the API on Heroku.
- `requirements.txt`: PIP requirements for the API.
- `runtime.txt`: Designates the Python version on Heroku.
- `test_api.py`: Pre-deployment pytest script.

## API Endpoints

`/get_num_feats/`: Responds with the number of features used in predictions.
`/get_applicants/`: Responds with a list of applicants and their approval statuses.
`/get_applicant_feats/`: Responds with the features of a specific applicant by SK_ID_CURR.
`/plot_glob_feat_import/`: Serves a global feature importances plot.
`/plot_local_feat_import/`: Serves a local feature importances plot for a specific applicant.
`/plot_approv_proba/`: Serves a plot showing the approval probability for a specific applicant.
`/plot_appl_features/`: Serves a plot of applicant-specific features over distributions of approved and denied applicants.