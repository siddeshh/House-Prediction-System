from pydantic import BaseModel, Field
from typing import List, Optional

class PropertyPredictionRequest(BaseModel):
    total_sqft: float = Field(
        ...,
        gt=150.0,
        lt=30000.0,
        description="Total property area in square feet (must be between 150 and 30,000 sq.ft.)",
        examples=[1250.0]
    )
    bhk: int = Field(
        ...,
        ge=1,
        le=15,
        description="Number of bedrooms / BHK (1 to 15)",
        examples=[2]
    )
    bath: int = Field(
        ...,
        ge=1,
        le=15,
        description="Number of bathrooms (1 to 15)",
        examples=[2]
    )
    balcony: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description="Number of balconies (0 to 10)",
        examples=[1.0]
    )
    site_location: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Locality name in Pune (e.g. Kothrud, Aundh, Baner)",
        examples=["Kothrud"]
    )
    area_type: Optional[str] = Field(
        "Super built-up  Area",
        description="Area classification: 'Super built-up  Area', 'Built-up  Area', 'Plot  Area', 'Carpet  Area'",
        examples=["Super built-up  Area"]
    )

class PropertyPredictionResponse(BaseModel):
    predicted_price_lakhs: float = Field(..., description="Estimated valuation in Lakhs INR")
    predicted_price_inr: float = Field(..., description="Estimated valuation in INR")
    currency: str = Field("INR", description="Currency identifier")
    units: str = Field("Lakhs", description="Pricing units")
    model_version: str = Field(..., description="Model version producing the prediction")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

class LocationsResponse(BaseModel):
    count: int
    locations: List[str]
