"""
Load daily sales, aggregate to monthly totals per (shop, item), and build a
wide matrix (months 0–LAST_HIST_MONTH) indexed by (shop_id, item_id). Rows are
reindexed to the union of historical pairs and test pairs so lags exist for
new shop–item combinations (zeros where there was no history).
"""
import pandas as pd

from config import CLIP_MAX, CLIP_MIN, DATA_DIR, LAST_HIST_MONTH


def load_raw():
    train = pd.read_csv(DATA_DIR / "sales_train.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")
    shops = pd.read_csv(DATA_DIR / "shops.csv")
    cats = pd.read_csv(DATA_DIR / "item_categories.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, items, shops, cats, test


def monthly_sales(train: pd.DataFrame) -> pd.DataFrame:
    sales = (
        train.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
        .sum()
        .rename(columns={"item_cnt_day": "item_cnt_month"})
    )
    sales["item_cnt_month"] = sales["item_cnt_month"].clip(CLIP_MIN, CLIP_MAX)
    return sales


def build_wide_matrix(sales: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    wide = sales.pivot_table(
        index=["shop_id", "item_id"],
        columns="date_block_num",
        values="item_cnt_month",
        aggfunc="sum",
        fill_value=0,
    )
    for c in range(LAST_HIST_MONTH + 1):
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide.reindex(sorted(wide.columns), axis=1).astype("float32")

    train_pairs = wide.index.to_frame(index=False)
    test_pairs = test[["shop_id", "item_id"]].drop_duplicates()
    all_pairs = pd.concat([train_pairs, test_pairs], ignore_index=True).drop_duplicates()

    idx = pd.MultiIndex.from_frame(all_pairs[["shop_id", "item_id"]])
    return wide.reindex(idx, fill_value=0.0)


def preprocess():
    train, items, _shops, _cats, test = load_raw()
    sales = monthly_sales(train)
    wide = build_wide_matrix(sales, test)
    return {"wide": wide, "items": items, "test": test}
