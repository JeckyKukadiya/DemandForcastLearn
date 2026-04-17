"""Paths, time-index conventions, and model column names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
MODEL_DIR = ROOT / "models"
CHARTS_DIR = ROOT / "charts"

# sales_train covers date_block_num 0 .. LAST_HIST_MONTH inclusive
LAST_HIST_MONTH = 33
TEST_MONTH = 34
FIRST_FEATURE_MONTH = 3  # need three prior months for lags

CLIP_MIN, CLIP_MAX = 0, 20

# First month index in sales_train is Jan 2013 → date_block_num 0 = 2013-01-01
HISTORY_START_ISO = "2013-01-01"

# Default multi-step forecast from month LAST_HIST_MONTH + 1
FORECAST_HORIZON_MONTHS = 6

FEATURE_COLUMNS = [
    "shop_id",
    "item_id",
    "item_category_id",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_mean",
    "cat_avg_lag1",
    "shop_avg_lag1",
    "month_of_year",
    "month_sin",
    "month_cos",
]
