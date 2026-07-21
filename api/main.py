from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from .config import FEATURE_COLUMNS_PATH, METRICS_PATH, MODEL_PATH
from .data_store import DataStore
from .forecasting import ForecastEngine, InsufficientHistoryError, SeriesNotFoundError
from .metrics import METRICS_CONTENT_TYPE, PrometheusMiddleware, render_metrics
from .monitoring import compute_kpis
from .order_suggestion import compute_order_suggestion
from .schemas import (
    BatchForecastError,
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    KPIResponse,
    ModelInfo,
    OrderSuggestionRequest,
    OrderSuggestionResponse,
    ReadyResponse,
    SeriesKPI,
)

MODEL_VERSION_PREFIX = "xgb-sales-forecaster"


class ModelRegistry:
    """Holds everything loaded at startup: model artifacts + historical data."""

    def __init__(self) -> None:
        self.model = None
        self.feature_columns = None
        self.metrics = None
        self.data_store: Optional[DataStore] = None
        self.engine: Optional[ForecastEngine] = None
        self.model_version = ""
        self.trained_at: Optional[datetime] = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.engine is not None

    def load(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        self.metrics = joblib.load(METRICS_PATH)
        self.data_store = DataStore()
        self.engine = ForecastEngine(
            model=self.model,
            feature_columns=self.feature_columns,
            metrics=self.metrics,
            data_store=self.data_store,
        )

        mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, tz=timezone.utc)
        self.trained_at = mtime
        self.model_version = f"{MODEL_VERSION_PREFIX}-{mtime.strftime('%Y%m%d%H%M%S')}"


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load()
    yield


app = FastAPI(title="Stocksense Demand Forecasting API", lifespan=lifespan)
app.add_middleware(PrometheusMiddleware)


def _require_ready() -> ModelRegistry:
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return registry


def _run_forecast(reg: ModelRegistry, request: ForecastRequest) -> ForecastResponse:
    try:
        points = reg.engine.forecast(
            store_id=request.store_id,
            item_id=request.item_id,
            start_date=request.start_date,
            horizon_days=request.horizon_days,
            confidence_level=request.confidence_level,
        )
    except SeriesNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ForecastResponse(
        store_id=request.store_id,
        item_id=request.item_id,
        model_version=reg.model_version,
        generated_at=datetime.now(timezone.utc),
        forecasts=[ForecastPoint(**p) for p in points],
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready() -> ReadyResponse:
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return ReadyResponse(status="ok", model_loaded=True)


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(request: ForecastRequest) -> ForecastResponse:
    reg = _require_ready()
    return _run_forecast(reg, request)


@app.post("/forecast/batch", response_model=BatchForecastResponse, tags=["forecast"])
def forecast_batch(request: BatchForecastRequest) -> BatchForecastResponse:
    reg = _require_ready()

    results = []
    errors = []
    for series_request in request.series:
        try:
            results.append(_run_forecast(reg, series_request))
        except HTTPException as exc:
            errors.append(
                BatchForecastError(
                    store_id=series_request.store_id,
                    item_id=series_request.item_id,
                    error=str(exc.detail),
                )
            )

    return BatchForecastResponse(results=results, errors=errors)


@app.get("/model", response_model=ModelInfo, tags=["model"])
@app.get("/version", response_model=ModelInfo, tags=["model"], include_in_schema=False)
def model_info() -> ModelInfo:
    reg = _require_ready()
    return ModelInfo(
        model_version=reg.model_version,
        trained_at=reg.trained_at,
        feature_columns=reg.feature_columns,
        metrics=reg.metrics,
    )


@app.post("/order-suggestion", response_model=OrderSuggestionResponse, tags=["order"])
def order_suggestion(request: OrderSuggestionRequest) -> OrderSuggestionResponse:
    reg = _require_ready()

    forecast_request = ForecastRequest(
        store_id=request.store_id,
        item_id=request.item_id,
        start_date=request.start_date,
        horizon_days=request.lead_time_days,
    )
    forecast_response = _run_forecast(reg, forecast_request)

    suggestion = compute_order_suggestion(
        forecast_points=[p.model_dump() for p in forecast_response.forecasts],
        current_stock=request.current_stock,
        lead_time_days=request.lead_time_days,
        service_level=request.service_level,
        rmse=reg.engine.rmse,
    )

    return OrderSuggestionResponse(
        store_id=request.store_id,
        item_id=request.item_id,
        lead_time_days=request.lead_time_days,
        service_level=request.service_level,
        current_stock=request.current_stock,
        forecast=forecast_response.forecasts,
        **suggestion,
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    return Response(content=render_metrics(), media_type=METRICS_CONTENT_TYPE)


@app.get("/monitoring/kpis", response_model=KPIResponse, tags=["monitoring"])
def monitoring_kpis(
    store_id: Optional[int] = Query(None, ge=1),
    item_id: Optional[int] = Query(None, ge=1),
    window_days: int = Query(30, ge=1, le=365),
) -> KPIResponse:
    reg = _require_ready()

    series = compute_kpis(
        model=reg.model,
        feature_columns=reg.feature_columns,
        features=reg.data_store.features,
        store_id=store_id,
        item_id=item_id,
        window_days=window_days,
    )

    return KPIResponse(
        generated_at=datetime.now(timezone.utc),
        series=[SeriesKPI(**s) for s in series],
    )
