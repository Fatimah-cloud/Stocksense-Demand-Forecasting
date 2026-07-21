"""Recursive multi-step forecasting on top of the trained XGBoost model.

The model was trained on lag_1 / lag_7 / lag_30 / rolling_mean_7 /
rolling_mean_30 features, which means every day's prediction needs the
(actual or predicted) sales for the preceding days. To forecast a horizon
that starts after the last date in history, we walk forward one day at a
time, feed the model's own prediction back in as if it were an observed
value, and use that extended series to build the next day's features.
"""

from datetime import date, timedelta
from typing import List

import pandas as pd
from scipy.stats import norm

from .config import MAX_BRIDGE_PLUS_HORIZON_DAYS
from .data_store import DataStore


class SeriesNotFoundError(Exception):
    """Raised when a (store, item) pair has no history to forecast from."""


class InsufficientHistoryError(Exception):
    """Raised when there isn't enough history to compute lag/rolling features."""


def _date_features(d: pd.Timestamp) -> dict:
    return {
        "year": d.year,
        "month": d.month,
        "day": d.day,
        "day_of_week": d.dayofweek,
        "week_of_year": int(d.isocalendar().week),
        "quarter": d.quarter,
        "is_weekend": int(d.dayofweek >= 5),
    }


def _z_score(confidence_level: float) -> float:
    return float(norm.ppf(0.5 + confidence_level / 2))


class ForecastEngine:
    def __init__(self, model, feature_columns: List[str], metrics: dict, data_store: DataStore):
        self.model = model
        self.feature_columns = feature_columns
        self.metrics = metrics
        self.data_store = data_store
        self.rmse = float(metrics.get("RMSE", 0.0))

    def forecast(
        self,
        store_id: int,
        item_id: int,
        start_date: date,
        horizon_days: int,
        confidence_level: float = 0.8,
    ) -> List[dict]:
        history = self.data_store.get_history(store_id, item_id)
        if history is None or history.empty:
            raise SeriesNotFoundError(f"No history found for store={store_id}, item={item_id}")

        start_ts = pd.Timestamp(start_date)
        end_ts = start_ts + timedelta(days=horizon_days - 1)
        last_actual_date = history.index.max()

        span_days = (end_ts - min(start_ts, last_actual_date + timedelta(days=1))).days + 1
        if span_days > MAX_BRIDGE_PLUS_HORIZON_DAYS:
            raise InsufficientHistoryError(
                "Requested date range is too far beyond the last known sales date "
                f"({last_actual_date.date()}); narrow the horizon or move start_date closer."
            )

        working = history.copy()
        z = _z_score(confidence_level)

        forecasted: List[dict] = []
        cursor = last_actual_date + timedelta(days=1)
        step = 0
        d = cursor
        while d <= end_ts:
            step += 1
            try:
                lag_1 = working.loc[d - timedelta(days=1)]
                lag_7 = working.loc[d - timedelta(days=7)]
                lag_30 = working.loc[d - timedelta(days=30)]
                window_7 = working.loc[d - timedelta(days=7) : d - timedelta(days=1)]
                window_30 = working.loc[d - timedelta(days=30) : d - timedelta(days=1)]
            except KeyError as exc:
                raise InsufficientHistoryError(
                    f"Not enough history before {d.date()} to compute lag features "
                    f"for store={store_id}, item={item_id}"
                ) from exc

            if len(window_7) < 7 or len(window_30) < 30:
                raise InsufficientHistoryError(
                    f"Not enough history before {d.date()} to compute rolling features "
                    f"for store={store_id}, item={item_id}"
                )

            feature_row = {
                "store": store_id,
                "item": item_id,
                **_date_features(d),
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_30": lag_30,
                "rolling_mean_7": window_7.mean(),
                "rolling_mean_30": window_30.mean(),
            }
            X = pd.DataFrame([feature_row])[self.feature_columns]
            pred = max(float(self.model.predict(X)[0]), 0.0)
            working.loc[d] = pred

            if d >= start_ts:
                spread = z * self.rmse * (step**0.5)
                forecasted.append(
                    {
                        "date": d.date(),
                        "forecast": round(pred, 2),
                        "lower": round(max(pred - spread, 0.0), 2),
                        "upper": round(pred + spread, 2),
                    }
                )
            d += timedelta(days=1)

        results = forecasted
        if start_ts <= last_actual_date:
            actual_slice = history.loc[start_ts : min(end_ts, last_actual_date)]
            actual_points = [
                {
                    "date": idx.date(),
                    "forecast": float(val),
                    "lower": float(val),
                    "upper": float(val),
                }
                for idx, val in actual_slice.items()
            ]
            results = actual_points + forecasted

        results.sort(key=lambda r: r["date"])
        return results
