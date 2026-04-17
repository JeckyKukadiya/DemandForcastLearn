"""
Print dataset summaries for exploratory analysis.

  python analyze.py                 # fast: raw files + wide monthly matrix
  python analyze.py --full         # also summarize the full modeling frame
  python analyze.py --charts       # save PNG figures under ../charts/
  python analyze.py --charts --full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import CHARTS_DIR, CLIP_MAX, CLIP_MIN, LAST_HIST_MONTH, TEST_MONTH
from preprocess import load_raw, monthly_sales, preprocess


def _hdr(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def analyze_raw_sales() -> None:
    _hdr("Raw daily sales (sales_train.csv)")
    train, *_rest = load_raw()
    print(f"Rows: {len(train):,}")
    print(f"Columns: {list(train.columns)}")
    print(f"date_block_num range: {train['date_block_num'].min()} – {train['date_block_num'].max()}")
    print(f"Unique shops: {train['shop_id'].nunique():,}")
    print(f"Unique items: {train['item_id'].nunique():,}")
    print()

    print("item_cnt_day:")
    print(train["item_cnt_day"].describe(percentiles=[0.01, 0.5, 0.99]).to_string())
    print()

    print("item_price:")
    print(train["item_price"].describe(percentiles=[0.01, 0.5, 0.99]).to_string())
    print()

    monthly = monthly_sales(train)
    print(f"After monthly groupby + clip [{CLIP_MIN}, {CLIP_MAX}]:")
    print(f"  Monthly rows: {len(monthly):,}")
    print(monthly["item_cnt_month"].describe(percentiles=[0.01, 0.5, 0.99]).to_string())


def analyze_catalog_and_test() -> None:
    _hdr("Catalog & test")
    _train, items, shops, cats, test = load_raw()
    print(f"items.csv rows: {len(items):,}  (unique item_id: {items['item_id'].nunique():,})")
    print(f"shops.csv rows: {len(shops):,}")
    print(f"item_categories.csv rows: {len(cats):,}")
    print(f"test.csv rows: {len(test):,}  (unique shop_id: {test['shop_id'].nunique():,})")
    print()

    tr_items = set(_train["item_id"].unique())
    te_items = set(test["item_id"].unique())
    print(f"Items only in test (not in daily train): {len(te_items - tr_items):,}")
    print(f"Items only in train (not in test): {len(tr_items - te_items):,}")


def analyze_wide() -> None:
    _hdr(f"Preprocessed wide matrix (months 0–{LAST_HIST_MONTH})")
    data = preprocess()
    wide = data["wide"]
    print(f"Shape (shop×item pairs × months): {wide.shape[0]:,} × {wide.shape[1]}")
    print(f"Month columns: {list(wide.columns[:4])} … {list(wide.columns[-3:])}")
    print()

    last = wide[LAST_HIST_MONTH]
    print(f"Month {LAST_HIST_MONTH} total units (sum of clipped monthly counts): {last.sum():,.0f}")
    print(f"Share of pairs with zero sales in month {LAST_HIST_MONTH}: {(last == 0).mean() * 100:.1f}%")
    print()

    by_month = wide.sum(axis=0)
    print("Total units sold per month (all shops/items):")
    print(by_month.to_string())


def analyze_modeling_frame() -> None:
    _hdr("Full modeling frame (pipeline output)")
    from dataset import load_modeling_frame

    df = load_modeling_frame()
    print(f"Rows: {len(df):,}  Columns: {len(df.columns)}")
    print()

    hist = df[df["date_block_num"] <= LAST_HIST_MONTH]
    print("Training months (have target):")
    g = hist.groupby("date_block_num")["item_cnt_month"].agg(["count", "mean", "std", "min", "max"])
    print(g.to_string())
    print()

    nov = df[df["date_block_num"] == TEST_MONTH]
    print(f"Prediction month ({TEST_MONTH}): {len(nov):,} rows (test shop–item pairs)")
    if "ID" in nov.columns:
        print(f"  ID range: {nov['ID'].min()} – {nov['ID'].max()}")


def run_charts(out_dir: Path) -> None:
    from charts import save_analysis_charts

    train, *_ = load_raw()
    monthly = monthly_sales(train)
    wide = preprocess()["wide"]
    paths = save_analysis_charts(out_dir, wide, train, monthly)
    print()
    _hdr("Charts")
    for p in paths:
        print(p)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize competition data and pipeline outputs.")
    p.add_argument(
        "--full",
        action="store_true",
        help="Load the full long-format modeling frame (slower, ~16M rows).",
    )
    p.add_argument(
        "--charts",
        action="store_true",
        help=f"Write PNG charts to {CHARTS_DIR} (requires matplotlib).",
    )
    p.add_argument(
        "--charts-dir",
        type=str,
        default=None,
        help=f"Override output directory for charts (default: {CHARTS_DIR}).",
    )
    args = p.parse_args(argv)

    out_charts = Path(args.charts_dir) if args.charts_dir else CHARTS_DIR

    try:
        analyze_raw_sales()
        analyze_catalog_and_test()
        analyze_wide()
        if args.full:
            analyze_modeling_frame()
        else:
            print()
            print("(Skip full modeling frame. Run with --full to include it.)")

        if args.charts:
            run_charts(out_charts)

        return 0
    except FileNotFoundError as e:
        print(f"Missing file: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"Charts need matplotlib: pip install matplotlib\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
