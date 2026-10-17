import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")

_model = None

def get_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads and caches the trained production model pipeline.
    """
    global _model
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
        _model = joblib.load(model_path)
    return _model

def predict_house_price(
    total_sqft: float,
    bhk: int,
    bath: int,
    balcony: float = 1.0,
    site_location: str = "other",
    area_type: str = "Super built-up  Area",
    model_path: str = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """
    Accepts property parameters and returns real-time valuation prediction.
    """
    pipeline = get_model(model_path)
    
    # Structure input dataframe matching exact training columns
    input_df = pd.DataFrame([{
        'total_sqft': float(total_sqft),
        'bath': float(bath),
        'bhk': float(bhk),
        'balcony': float(balcony),
        'site_location': str(site_location).strip(),
        'area_type': str(area_type).strip()
    }])
    
    pred_lakhs = float(pipeline.predict(input_df)[0])
    
    # Clip at realistic minimum (e.g. at least 5 Lakhs)
    pred_lakhs = round(max(pred_lakhs, 5.0), 2)
    
    # Convert Lakhs to INR (1 Lakh = 100,000 INR)
    pred_inr = round(pred_lakhs * 100000.0, 2)
    
    return {
        "predicted_price_lakhs": pred_lakhs,
        "predicted_price_inr": pred_inr,
        "currency": "INR",
        "units": "Lakhs",
        "model_version": "1.0.0-gradient_boosting"
    }

if __name__ == "__main__":
    test_res = predict_house_price(1200, 2, 2, 1.0, "Kothrud")
    print("Test Prediction:", test_res)
