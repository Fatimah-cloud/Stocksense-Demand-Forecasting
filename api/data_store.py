"""Loads historical sales data and precomputes the features the model needs.

Two things are kept in memory:
  * ``history``  -- per (store, item) actual sales as a date-indexed Series,
                     used to seed the recursive lag/rolling features when
                     forecasting into the future.
  * ``features`` -- the same feature matrix the model was trained on
                     (date, engineered features, actual sales), used for the
                     monitoring/KPI endpoint so it doesn't need to recompute
                     lag features by hand.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .config import TRAIN_DATA_PATH


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the feature engineering from the training notebook."""

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    grouped = df.groupby(["store", "item"])["sales"]
    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_30"] = grouped.shift(30)
    df["rolling_mean_7"] = grouped.transform(lambda x: x.shift(1).rolling(7).mean())
    df["rolling_mean_30"] = grouped.transform(lambda x: x.shift(1).rolling(30).mean())

    return df


class DataStore:
    def __init__(self, train_path: Path = TRAIN_DATA_PATH):
        self.history: Dict[Tuple[int, int], pd.Series] = {}
        self.features: Optional[pd.DataFrame] = None
        self._load(train_path)

    def _load(self, train_path: Path) -> None:
        train = pd.read_csv(train_path, parse_dates=["date"])
        train = train.sort_values(["store", "item", "date"]).reset_index(drop=True)

        for (store, item), group in train.groupby(["store", "item"]):
            self.history[(int(store), int(item))] = (
                group.set_index("date")["sales"].astype(float).sort_index()
            )

        engineered = engineer_features(train)
        self.features = engineered.dropna().reset_index(drop=True)

    def get_history(self, store_id: int, item_id: int) -> Optional[pd.Series]:
        return self.history.get((store_id, item_id))

    def has_series(self, store_id: int, item_id: int) -> bool:
        return (store_id, item_id) in self.history
