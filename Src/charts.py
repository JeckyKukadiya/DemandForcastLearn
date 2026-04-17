"""Save exploratory figures for analyze.py (non-interactive backend)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import LAST_HIST_MONTH


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def chart_monthly_totals(wide: pd.DataFrame, out: Path) -> None:
    by_month = wide.sum(axis=0).sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_month.index, by_month.values, marker="o", markersize=3)
    _style_axes(ax, "Total units sold per month (clipped monthly sum)", "date_block_num", "units")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_daily_rows_per_month(train: pd.DataFrame, out: Path) -> None:
    counts = train.groupby("date_block_num", sort=True).size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values, width=0.8, color="steelblue", edgecolor="none")
    _style_axes(ax, "Number of daily sales rows per month", "date_block_num", "row count")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_monthly_item_cnt_distribution(monthly: pd.DataFrame, out: Path) -> None:
    s = monthly["item_cnt_month"]
    hi = int(float(s.max())) + 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=range(0, hi), color="coral", edgecolor="white", linewidth=0.5)
    _style_axes(ax, "Distribution of monthly item_cnt (after clip)", "item_cnt_month", "frequency")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_last_month_nonzero(wide: pd.DataFrame, out: Path) -> None:
    last = wide[LAST_HIST_MONTH]
    zeros = (last == 0).sum()
    nonzero = (last > 0).sum()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["zero sales", "non-zero"], [zeros, nonzero], color=["#94a3b8", "#2563eb"])
    _style_axes(
        ax,
        f"Shop–item pairs in month {LAST_HIST_MONTH}",
        "",
        "count",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_analysis_charts(out_dir: Path, wide: pd.DataFrame, train: pd.DataFrame, monthly: pd.DataFrame) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    specs = [
        ("01_monthly_totals.png", lambda p: chart_monthly_totals(wide, p)),
        ("02_daily_rows_per_month.png", lambda p: chart_daily_rows_per_month(train, p)),
        ("03_monthly_item_cnt_distribution.png", lambda p: chart_monthly_item_cnt_distribution(monthly, p)),
        ("04_last_month_zero_vs_sales.png", lambda p: chart_last_month_nonzero(wide, p)),
    ]
    for name, fn in specs:
        path = out_dir / name
        fn(path)
        paths.append(path)
    return paths
