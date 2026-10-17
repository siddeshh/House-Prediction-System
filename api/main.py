import os
import sys
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure src and api are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from predict import predict_house_price, get_model
from schemas import (
    PropertyPredictionRequest,
    PropertyPredictionResponse,
    HealthResponse,
    LocationsResponse
)

app = FastAPI(
    title="Pune House Price Prediction Service",
    description="Production-grade Machine Learning REST API for real estate valuation in Pune.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load available localities from dataset for dropdown choices
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pune_house_data_cleaned.csv")
KNOWN_LOCATIONS: List[str] = []
if os.path.exists(DATA_PATH):
    try:
        _df = pd.read_csv(DATA_PATH)
        KNOWN_LOCATIONS = sorted([str(loc) for loc in _df['site_location'].unique() if loc != 'other'])
    except Exception:
        KNOWN_LOCATIONS = []

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="Service Healthcheck"
)
def healthcheck():
    """Checks model availability and API readiness."""
    try:
        model = get_model()
        is_ready = model is not None
    except Exception:
        is_ready = False
        
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_version="1.0.0-gradient_boosting"
    )

@app.get(
    "/locations",
    response_model=LocationsResponse,
    tags=["Metadata"],
    summary="List Known Pune Localities"
)
def get_locations():
    """Returns the list of top Pune localities recognized by the model."""
    return LocationsResponse(
        count=len(KNOWN_LOCATIONS),
        locations=KNOWN_LOCATIONS
    )

@app.post(
    "/predict",
    response_model=PropertyPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Predict House Valuation"
)
def predict_valuation(request: PropertyPredictionRequest):
    """
    Predicts estimated house price in Lakhs and INR given structural features.
    
    Validation Rules:
    - total_sqft must be between 150 and 30,000 sq.ft.
    - bhk and bath must be between 1 and 15.
    - Locality must be provided (unseen localities are gracefully mapped to 'other').
    """
    try:
        result = predict_house_price(
            total_sqft=request.total_sqft,
            bhk=request.bhk,
            bath=request.bath,
            balcony=request.balcony if request.balcony is not None else 1.0,
            site_location=request.site_location,
            area_type=request.area_type if request.area_type else "Super built-up  Area"
        )
        return PropertyPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction inference failure: {str(e)}"
        )
