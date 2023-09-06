#!/usr/bin/env python

import pytest
import json
from fastapi.testclient import TestClient
import random

# Import the FastAPI application
from main import app

# Define a test client for the FastAPI application
client = TestClient(app)

# Initialize globals
num_feats = None
sk_id = None


# Test get_num_feats endpoint
def test_get_num_feats():
    global num_feats  # Declare num_feats as global

    # Send a GET request to the endpoint
    response = client.get("/get_num_feats/")

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Parse the JSON response
    resp_txt = json.loads(response.text)

    # Check if the parsed data is an integer
    assert isinstance(resp_txt, int)

    # Assign the integer to num_feats
    num_feats = resp_txt


# Test get_applicants endpoint
def test_get_applicants():
    global sk_id  # Declare sk_id as global

    # Send a GET request to the endpoint
    response = client.get("/get_applicants/")

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Parse the JSON response
    resp_data = json.loads(response.text)

    # Check if the parsed data is a dictionary
    assert isinstance(resp_data, dict)

    # Check if the keys are integers and values are strings (in 10 random pairs)
    for key in random.sample(list(resp_data.keys()), 10):
        assert key.isdigit()  # Keys contain only digits
        assert isinstance(resp_data[key], str)

    # Save a random key as an integer in sk_id
    sk_id = int(random.choice(list(resp_data.keys())))


# Test plot_glob_feat_import endpoint
def test_plot_glob_feat_import():
    # Send a GET request to the endpoint
    response = client.get("/plot_glob_feat_import/")

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200
    # Check that the response content is a PNG
    assert response.headers["content-type"] == "image/png"


# Test plot_local_feat_import endpoint
# Use the @pytest.mark.filterwarnings decorator to suppress shap_values() warning
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plot_local_feat_import():
    global sk_id  # Declare sk_id as global

    # Send a GET request to the endpoint
    response = client.get(f"/plot_local_feat_import/?sk_id={sk_id}")

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200
    # Check that the response content is a PNG
    assert response.headers["content-type"] == "image/png"


# Test plot_approv_proba endpoint
def test_plot_approv_proba():
    global sk_id  # Declare sk_id as global

    # Send a GET request to the endpoint
    response = client.get(f"/plot_approv_proba/?sk_id={sk_id}")

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200
    # Check that the response content is a PNG
    assert response.headers["content-type"] == "image/png"


# Test plot_appl_features endpoint
# Use the @pytest.mark.filterwarnings decorator to suppress shap_values() warning
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plot_appl_features():
    global num_feats  # Declare num_feats as global
    global sk_id  # Declare sk_id as global

    # Send a GET request to the endpoint
    response = client.get(f"/plot_appl_features/?sk_id={sk_id}&num_feats={num_feats}")

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200
    # Check that the response content is a PNG
    assert response.headers["content-type"] == "image/png"
