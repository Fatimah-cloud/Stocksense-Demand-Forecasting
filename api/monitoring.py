"""On-demand forecast-quality KPIs (MAE/RMSE/MAPE/bias) per series.

Reuses the precomputed feature matrix in DataStore (which already carries
the correct historical lag/rolling features and actual sales) so it can
score the model one-step-ahead against real outcomes without any recursive
bridging.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_MAX_SERIES = 25


def compute_kpis(
    model,
    feature_columns: List[str],
    features: pd.DataFrame,
    store_id: Optional[int] = None,
    item_id: Optional[int] = None,
    window_days: int = 30,
    max_series: int = DEFAULT_MAX_SERIES,
) -> List[dict]:
    df = features
    if store_id is not None:
        df = df[df["store"] == store_id]
    if item_id is not None:
        df = df[df["item"] == item_id]

    if df.empty:
        return []

    cutoff = df["date"].max() - pd.Timedelta(days=window_days - 1)
    recent = df[df["date"] >= cutoff]

    results = []
    for (store, item), group in list(recent.groupby(["store", "item"]))[:max_series]:
        X = group[feature_columns]
        y = group["sales"].to_numpy(dtype=float)
        preds = model.predict(X)

        errors = preds - y
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        mape = float(np.mean(np.abs(errors) / (y + 1e-8)) * 100)
        bias = float(np.mean(errors))

        results.append(
            {
                "store_id": int(store),
                "item_id": int(item),
                "window_days": window_days,
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "mape": round(mape, 3),
                "bias": round(bias, 3),
                "n_obs": int(len(group)),
            }
        )

    return results
