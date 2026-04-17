"""
Long-format modeling frame: for each month m in [FIRST_FEATURE_MONTH .. TEST_MONTH],
one row per (shop_id, item_id) with lags and category/shop aggregates from the
prior month (helps when the product or shop mix changes).
"""
import numpy as np
import pandas as pd

from config import FIRST_FEATURE_MONTH, LAST_HIST_MONTH, TEST_MONTH


def _month_of_year(date_block_num: int) -> int:
    return int(date_block_num % 12)


def _month_block(
    wide: pd.DataFrame,
    m: int,
    idx: pd.DataFrame,
    item_cat: pd.Series,
) -> pd.DataFrame:
    lag1 = wide[m - 1].to_numpy()
    lag2 = wide[m - 2].to_numpy()
    lag3 = wide[m - 3].to_numpy()
    n = len(idx)

    df = idx.copy()
    df["date_block_num"] = m
    df["lag_1"] = lag1
    df["lag_2"] = lag2
    df["lag_3"] = lag3
    df["lag_mean"] = (lag1 + lag2 + lag3) / 3.0

    cat = df["item_id"].map(item_cat)
    df["item_category_id"] = cat

    prev = pd.DataFrame(
        {
            "shop_id": df["shop_id"],
            "item_id": df["item_id"],
            "item_category_id": cat,
            "prev": lag1,
        }
    )
    df["cat_avg_lag1"] = prev.groupby("item_category_id", sort=False)["prev"].transform("mean")
    df["shop_avg_lag1"] = prev.groupby("shop_id", sort=False)["prev"].transform("mean")

    mob = np.full(n, _month_of_year(m), dtype=np.int32)
    df["month_of_year"] = mob
    df["month_sin"] = np.sin(2 * np.pi * mob / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * mob / 12.0)

    if m <= LAST_HIST_MONTH:
        df["item_cnt_month"] = wide[m].to_numpy()
    else:
        df["item_cnt_month"] = np.nan

    return df


def features_for_month(wide: pd.DataFrame, items: pd.DataFrame, month: int) -> pd.DataFrame:
    """Single-month feature matrix for every (shop_id, item_id) row in wide (used for multi-step forecast)."""
    item_cat = items.set_index("item_id")["item_category_id"]
    idx = wide.index.to_frame(index=False)
    return _month_block(wide, month, idx, item_cat)


def features_for_month_range(
    wide: pd.DataFrame,
    items: pd.DataFrame,
    lo: int,
    hi: int,
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """Stack month blocks from lo..hi (inclusive). Used for training slices without building the full test frame."""
    item_cat = items.set_index("item_id")["item_category_id"]
    idx = wide.index.to_frame(index=False)
    parts = []
    n = hi - lo + 1
    for k, m in enumerate(range(lo, hi + 1)):
        if progress:
            print(f"    building feature month {m} ({k + 1}/{n})...", flush=True)
        parts.append(_month_block(wide, m, idx, item_cat))
    return pd.concat(parts, ignore_index=True)


def create_features(wide: pd.DataFrame, items: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    item_cat = items.set_index("item_id")["item_category_id"]
    idx = wide.index.to_frame(index=False)

    parts = [_month_block(wide, m, idx, item_cat) for m in range(FIRST_FEATURE_MONTH, TEST_MONTH + 1)]
    out = pd.concat(parts, ignore_index=True)

    if TEST_MONTH in out["date_block_num"].values:
        test_keys = test[["shop_id", "item_id", "ID"]].drop_duplicates()
        nov = out[out["date_block_num"] == TEST_MONTH].merge(
            test_keys, on=["shop_id", "item_id"], how="inner"
        )
        rest = out[out["date_block_num"] != TEST_MONTH]
        out = pd.concat([rest, nov], ignore_index=True)

    return out
