import pytest
import os
import joblib
import pandas as pd
import numpy as np

def test_baseline_model_artifact_exists():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.joblib")
    assert os.path.exists(model_path), f"Baseline model artifact not found at {model_path}"

def test_baseline_model_prediction_output():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.joblib")
    pipeline = joblib.load(model_path)
    
    # Test valid 2 BHK in Aundh
    sample_input = pd.DataFrame([{
        'total_sqft': 1000.0,
        'bhk': 2.0,
        'bath': 2.0,
        'balcony': 1.0,
        'site_location': 'Aundh',
        'area_type': 'Super built-up  Area'
    }])
    
    pred = pipeline.predict(sample_input)
    assert len(pred) == 1
    assert pred[0] > 0, f"Predicted price should be positive, got {pred[0]}"
    # Typical price for 1000 sqft in Aundh is between 30L and 150L
    assert 20.0 <= pred[0] <= 200.0, f"Predicted price {pred[0]} outside reasonable sanity range"
