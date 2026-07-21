from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
METRICS_PATH = MODEL_DIR / "metrics.pkl"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"

# Safety limits so a single request can't force the engine to bridge/forecast
# an unreasonable number of days.
MAX_BRIDGE_PLUS_HORIZON_DAYS = 730
MAX_BATCH_SERIES = 200
