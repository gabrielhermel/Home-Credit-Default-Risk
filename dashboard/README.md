# Dashboard for Credit Scoring Model

## Overview
This interactive dashboard is designed for customer relationship managers. Its primary objective is to interpret predictions made by a scoring model in a manner comprehensible to those unfamiliar with data science. Users can:

- Visualize and understand scores assigned to each client.
- Access descriptive client-specific information using filters.
- Compare information about a client with the broader customer base.

The dashboard fetches predictions from the API and is hosted on a cloud platform, Heroku, ensuring user accessibility via their workstations.

## Directory Structure

- `config.yaml`: Contains the API URL.
- `dashboard.py`: The Streamlit application for the dashboard.
- `Procfile`: Command to launch the dashboard on Heroku.
- `requirements.txt`: PIP dependencies for the dashboard.
- `runtime.txt`: Specifies the Python version for the Heroku server.
- `.streamlit/config.toml`: Theming and server configuration for the Streamlit dashboard.