# Stocksense Demand Forecasting API

FastAPI service that serves the XGBoost model trained in
`notebooks/Demand forecasting for inventory.ipynb` and saved to
`model/model.pkl` (with `model/feature_columns.pkl` and `model/metrics.pkl`).

## Run locally

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --app-dir "<repo root>"
```

Run the command from the repository root (the parent of `api/`), or set
`--app-dir` to it, since `api.main` is imported as a package (`api/config.py`
resolves `model/` and `data/` relative to the repo root).

Interactive docs: http://127.0.0.1:8000/docs

## How forecasting works

The model was trained on lag/rolling features (`lag_1`, `lag_7`, `lag_30`,
`rolling_mean_7`, `rolling_mean_30`), which means every prediction depends on
recent sales. At startup the API loads `data/train.csv` into an in-memory
per-(store, item) history. For a forecast request:

- Any requested dates that fall inside the historical range are returned as
  actuals (zero-width interval).
- Any requested dates beyond the last known date are forecast recursively:
  the model's own prediction for day *t* is fed back in as the "observed"
  value used to build day *t+1*'s lag/rolling features.

Prediction intervals are `forecast ± z(confidence_level) * RMSE * sqrt(step)`,
using the backtest RMSE from `model/metrics.pkl` and widening with each step
into the future to reflect compounding uncertainty.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness — is the process up. |
| `GET /ready` | Readiness — is the model actually loaded. |
| `POST /forecast` | Single (store, item) forecast with intervals. |
| `POST /forecast/batch` | Same, for many (store, item) pairs in one call (up to 200). Per-series errors are returned in `errors` rather than failing the whole batch. |
| `GET /model` (alias `GET /version`) | `model_version`, `trained_at`, `feature_columns`, and headline backtest metrics (MAE/RMSE/MAPE/Bias). |
| `POST /order-suggestion` | Forecast + current stock + lead time + service level → recommended order quantity via safety-stock math. |
| `GET /metrics` | Prometheus-format operational metrics (request counts, latency). |
| `GET /monitoring/kpis` | Recent MAE/RMSE/MAPE/bias per (store, item) series, computed one-step-ahead against real historical outcomes. Optional `store_id`, `item_id`, `window_days` query params. |

## Order suggestion math

```
expected_demand_lead_time = sum(forecast over lead_time_days)
safety_stock              = z(service_level) * RMSE * sqrt(lead_time_days)
reorder_point              = expected_demand_lead_time + safety_stock
recommended_order_qty      = max(0, reorder_point - current_stock)
```

`RMSE` is the backtest RMSE from `model/metrics.pkl`, used as a proxy for
daily forecast error std-dev (assumed roughly independent day to day, hence
the `sqrt(lead_time_days)` scaling).

## Notes / known limitations

- `model_version` and `trained_at` are derived from `model.pkl`'s file
  modification time since the training notebook doesn't persist an explicit
  version tag — regenerate the pickle (or add real versioning) if you need
  stronger provenance.
- History is loaded once at startup from `data/train.csv`; the API doesn't
  currently accept new actuals at runtime, so `/monitoring/kpis` and the
  recursive bridging are both bounded by that file's date range.
