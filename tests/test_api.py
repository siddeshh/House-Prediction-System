import pytest
import os
import sys
from fastapi.testclient import TestClient

# Ensure api module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data

def test_locations_endpoint():
    response = client.get("/locations")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    assert "Kothrud" in data["locations"]
    assert "Aundh" in data["locations"]

def test_valid_prediction():
    payload = {
        "total_sqft": 1250.0,
        "bhk": 2,
        "bath": 2,
        "balcony": 1.0,
        "site_location": "Kothrud",
        "area_type": "Super built-up  Area"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price_lakhs"] > 0
    assert data["predicted_price_inr"] == round(data["predicted_price_lakhs"] * 100000.0, 2)
    assert data["currency"] == "INR"
    assert data["units"] == "Lakhs"

def test_invalid_sqft_rejected():
    # Negative sqft must be rejected by Pydantic validation (422)
    payload = {
        "total_sqft": -50.0,
        "bhk": 2,
        "bath": 2,
        "site_location": "Kothrud"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_invalid_bhk_rejected():
    # 0 BHK must be rejected (422)
    payload = {
        "total_sqft": 1000.0,
        "bhk": 0,
        "bath": 1,
        "site_location": "Kothrud"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_missing_required_location_rejected():
    # Missing site_location must fail with 422
    payload = {
        "total_sqft": 1000.0,
        "bhk": 2,
        "bath": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_unseen_locality_handled_gracefully():
    # Model should gracefully handle unseen locality without crashing
    payload = {
        "total_sqft": 1100.0,
        "bhk": 2,
        "bath": 2,
        "balcony": 1.0,
        "site_location": "NonExistentFantasyLocality123",
        "area_type": "Super built-up  Area"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price_lakhs"] > 0
