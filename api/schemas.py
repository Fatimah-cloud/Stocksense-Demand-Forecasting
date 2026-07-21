from datetime import date, datetime
from typing import Dict, List

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    store_id: int = Field(..., ge=1, description="Store identifier")
    item_id: int = Field(..., ge=1, description="Item identifier")
    start_date: date = Field(..., description="First date to forecast")
    horizon_days: int = Field(..., ge=1, le=365, description="Number of days to forecast")
    confidence_level: float = Field(
        0.8, gt=0, lt=1, description="Confidence level used for the prediction interval"
    )


class ForecastPoint(BaseModel):
    date: date
    forecast: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    store_id: int
    item_id: int
    model_version: str
    generated_at: datetime
    forecasts: List[ForecastPoint]


class BatchForecastRequest(BaseModel):
    series: List[ForecastRequest] = Field(..., min_length=1, max_length=200)


class BatchForecastError(BaseModel):
    store_id: int
    item_id: int
    error: str


class BatchForecastResponse(BaseModel):
    results: List[ForecastResponse]
    errors: List[BatchForecastError] = Field(default_factory=list)


class ModelInfo(BaseModel):
    model_version: str
    trained_at: datetime
    feature_columns: List[str]
    metrics: Dict[str, float]


class OrderSuggestionRequest(BaseModel):
    store_id: int = Field(..., ge=1)
    item_id: int = Field(..., ge=1)
    start_date: date = Field(..., description="First date of the lead-time window")
    current_stock: float = Field(..., ge=0)
    lead_time_days: int = Field(..., ge=1, le=90, description="Supplier lead time in days")
    service_level: float = Field(
        0.95, gt=0, lt=1, description="Target service level, e.g. 0.95 for 95%"
    )


class OrderSuggestionResponse(BaseModel):
    store_id: int
    item_id: int
    lead_time_days: int
    service_level: float
    expected_demand_lead_time: float
    safety_stock: float
    reorder_point: float
    current_stock: float
    recommended_order_qty: float
    forecast: List[ForecastPoint]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool


class SeriesKPI(BaseModel):
    store_id: int
    item_id: int
    window_days: int
    mae: float
    rmse: float
    mape: float
    bias: float
    n_obs: int


class KPIResponse(BaseModel):
    generated_at: datetime
    series: List[SeriesKPI]
