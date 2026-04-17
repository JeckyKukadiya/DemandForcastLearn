"""
Recursive multi-month forecast: extend the wide matrix month by month, predict,
then write:
  - shop×item detail CSV
  - shop-wise totals CSV (sum over items per shop per month)
  - item-wise totals CSV (sum over shops per item per month)
  - charts: small multiples per shop; top items by volume (not one global total).

Requires a trained model (train.py) and the same feature pipeline as predict.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    CHARTS_DIR,
    CLIP_MAX,
    CLIP_MIN,
    DATA_DIR,
    FEATURE_COLUMNS,
    FORECAST_HORIZON_MONTHS,
    HISTORY_START_ISO,
    LAST_HIST_MONTH,
    MODEL_DIR,
)
from features import features_for_month
from preprocess import preprocess


def month_start_date(date_block_num: int) -> pd.Timestamp:
    return pd.Timestamp(HISTORY_START_ISO) + pd.DateOffset(months=int(date_block_num))


def run_forecast(
    months: int,
    test_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (forecast_long, wide_after) where forecast_long has
    shop_id, item_id, date, item_cnt, date_block_num.
    """
    data = preprocess()
    wide = data["wide"].copy()
    items = data["items"]
    test = data["test"]

    model = joblib.load(MODEL_DIR / "model.pkl")

    start = LAST_HIST_MONTH + 1
    end = start + months - 1

    for m in range(start, end + 1):
        feat = features_for_month(wide, items, m)
        pred = np.clip(model.predict(feat[FEATURE_COLUMNS]), CLIP_MIN, CLIP_MAX)
        wide[m] = pred.astype(np.float32)

    parts = []
    for m in range(start, end + 1):
        d = month_start_date(m)
        idx = wide.index.to_frame(index=False)
        idx["date"] = d.strftime("%Y-%m-%d")
        idx["item_cnt"] = wide[m].to_numpy()
        idx["date_block_num"] = m
        parts.append(idx)

    long_df = pd.concat(parts, ignore_index=True)
    if test_only:
        keys = test[["shop_id", "item_id"]].drop_duplicates()
        long_df = long_df.merge(keys, on=["shop_id", "item_id"], how="inner")

    long_df = long_df.sort_values(["date", "shop_id", "item_id"]).reset_index(drop=True)
    long_df["item_cnt"] = long_df["item_cnt"].astype(float).round(5)
    return long_df, wide


def wide_to_aggregate_long(wide: pd.DataFrame, by_shop: bool) -> pd.DataFrame:
    """One row per (shop_id or item_id) × month; totals summed over the other dimension."""
    if by_shop:
        g = wide.groupby(level=0).sum()
        id_col = "shop_id"
    else:
        g = wide.groupby(level=1).sum()
        id_col = "item_id"

    rows = []
    for rid in g.index:
        row = g.loc[rid]
        for m in row.index:
            rows.append(
                {
                    id_col: rid,
                    "date_block_num": int(m),
                    "date": month_start_date(int(m)).strftime("%Y-%m-%d"),
                    "total_item_cnt": round(float(row[m]), 5),
                }
            )
    return pd.DataFrame(rows)


def chart_sales_by_shop(wide: pd.DataFrame, months: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = LAST_HIST_MONTH + 1
    end = start + months - 1
    hist_blocks = list(range(0, LAST_HIST_MONTH + 1))
    fc_blocks = list(range(start, end + 1))

    by_shop = wide.groupby(level=0).sum()
    shop_ids = sorted(by_shop.index.unique())
    n = len(shop_ids)
    cols = 10
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(2.0 * cols, 1.8 * nrows), sharex="col")
    axes_flat = np.atleast_1d(axes).ravel()

    xh = [month_start_date(c) for c in hist_blocks]
    xf = [month_start_date(c) for c in fc_blocks]

    for i, sid in enumerate(shop_ids):
        ax = axes_flat[i]
        yh = [float(by_shop.loc[sid, c]) for c in hist_blocks]
        yf = [float(by_shop.loc[sid, c]) for c in fc_blocks]
        ax.plot(xh, yh, color="#1d4ed8", linewidth=1.2, label="hist")
        ax.plot(
            xf,
            yf,
            color="#ea580c",
            linewidth=1.4,
            linestyle=":",
            marker=".",
            markersize=2,
            label="fc",
        )
        ax.axvline(xh[-1], color="#94a3b8", linestyle="--", linewidth=0.6, alpha=0.8)
        ax.set_title(f"shop {sid}", fontsize=7)
        ax.tick_params(axis="both", labelsize=5)
        ax.grid(True, alpha=0.25)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Shop-wise total units (sum over items): solid = history, dotted = forecast",
        fontsize=11,
        y=1.002,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_sales_by_top_items(
    wide: pd.DataFrame,
    months: int,
    out: Path,
    top_n: int = 12,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = LAST_HIST_MONTH + 1
    end = start + months - 1
    hist_blocks = list(range(0, LAST_HIST_MONTH + 1))
    fc_blocks = list(range(start, end + 1))

    by_item = wide.groupby(level=1).sum()
    recent = list(range(max(0, LAST_HIST_MONTH - 11), LAST_HIST_MONTH + 1))
    vol = by_item[recent].sum(axis=1).sort_values(ascending=False)
    pick = vol.head(top_n).index.tolist()

    cols = 4
    nrows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(2.6 * cols, 2.0 * nrows), sharex="col")
    axes_flat = np.atleast_1d(axes).ravel()

    xh = [month_start_date(c) for c in hist_blocks]
    xf = [month_start_date(c) for c in fc_blocks]

    for i, iid in enumerate(pick):
        ax = axes_flat[i]
        yh = [float(by_item.loc[iid, c]) for c in hist_blocks]
        yf = [float(by_item.loc[iid, c]) for c in fc_blocks]
        ax.plot(xh, yh, color="#1d4ed8", linewidth=1.2)
        ax.plot(
            xf,
            yf,
            color="#ea580c",
            linewidth=1.4,
            linestyle=":",
            marker=".",
            markersize=2,
        )
        ax.axvline(xh[-1], color="#94a3b8", linestyle="--", linewidth=0.6, alpha=0.8)
        ax.set_title(f"item {iid}", fontsize=8)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.25)

    for j in range(len(pick), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Item-wise totals (sum over shops), top {top_n} items by last-12m history volume",
        fontsize=11,
        y=1.002,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multi-month shop–item forecast and charts.")
    p.add_argument(
        "--months",
        type=int,
        default=FORECAST_HORIZON_MONTHS,
        help=f"Number of future months to predict (default {FORECAST_HORIZON_MONTHS}).",
    )
    p.add_argument(
        "--test-only",
        action="store_true",
        help="Restrict detail CSV to shop–item pairs that appear in test.csv (smaller file).",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help=f"Detail CSV path (default: {DATA_DIR}/forecast_shop_item.csv).",
    )
    p.add_argument(
        "--chart-shop",
        type=str,
        default=None,
        help=f"Shop-wise chart path (default: {CHARTS_DIR}/05_forecast_by_shop.png).",
    )
    p.add_argument(
        "--chart-item",
        type=str,
        default=None,
        help=f"Item-wise chart path (default: {CHARTS_DIR}/06_forecast_by_item.png).",
    )
    p.add_argument(
        "--top-items",
        type=int,
        default=12,
        help="Number of top items (by recent history volume) to plot in item-wise chart.",
    )
    args = p.parse_args(argv)

    if args.months < 1:
        print("--months must be >= 1", file=sys.stderr)
        return 1

    try:
        long_df, wide = run_forecast(months=args.months, test_only=args.test_only)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    csv_detail = Path(args.csv) if args.csv else DATA_DIR / "forecast_shop_item.csv"
    csv_shop = DATA_DIR / "forecast_by_shop.csv"
    csv_item = DATA_DIR / "forecast_by_item.csv"
    chart_shop = Path(args.chart_shop) if args.chart_shop else CHARTS_DIR / "05_forecast_by_shop.png"
    chart_item = Path(args.chart_item) if args.chart_item else CHARTS_DIR / "06_forecast_by_item.png"

    out_cols = ["shop_id", "item_id", "date", "item_cnt", "date_block_num"]
    long_df[out_cols].to_csv(csv_detail, index=False)

    agg_shop = wide_to_aggregate_long(wide, by_shop=True)
    agg_item = wide_to_aggregate_long(wide, by_shop=False)
    agg_shop.to_csv(csv_shop, index=False)
    agg_item.to_csv(csv_item, index=False)

    chart_sales_by_shop(wide, args.months, chart_shop)
    chart_sales_by_top_items(wide, args.months, chart_item, top_n=args.top_items)

    print(f"Detail (shop×item): {len(long_df):,} rows → {csv_detail}")
    print(f"Shop-wise totals:   {len(agg_shop):,} rows → {csv_shop}")
    print(f"Item-wise totals:   {len(agg_item):,} rows → {csv_item}")
    print(f"Chart (by shop):    {chart_shop}")
    print(f"Chart (by item):    {chart_item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
