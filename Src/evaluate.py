"""
Compare model predictions to actual historical targets.

Default (fast): load models/model.pkl, build features for the holdout month only
(~526k rows, ~tens of seconds) — no sklearn import, no 16M-row frame.

Use --retrain-holdout for an honest out-of-sample score: retrain on months 3..LAST-1
without the holdout month (slow: builds ~15M training rows, several minutes).

Writes CSVs and charts.
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
    FIRST_FEATURE_MONTH,
    LAST_HIST_MONTH,
    MODEL_DIR,
)
from features import features_for_month, features_for_month_range
from forecast import month_start_date
from preprocess import preprocess
from train import _make_model


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def _r2(actual: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Regression metrics (pure NumPy — no sklearn/scipy)."""
    err = pred - actual
    n = len(actual)
    rmse = _rmse(actual, pred)
    mae = _mae(actual, pred)
    r2 = _r2(actual, pred)
    mean_bias = float(np.mean(err))

    den = float(np.sum(np.abs(actual)))
    wape = float(100.0 * np.sum(np.abs(err)) / den) if den > 0 else float("nan")

    smape = float(
        100.0 * np.mean(2.0 * np.abs(err) / (np.abs(actual) + np.abs(pred) + 1e-8))
    )

    mask = actual > 0.05
    mape_nonzero = (
        float(100.0 * np.mean(np.abs(err[mask] / actual[mask]))) if mask.any() else float("nan")
    )

    corr = float(np.corrcoef(actual, pred)[0, 1]) if n > 1 else float("nan")

    mask_pos = actual > 0
    n_pos = int(mask_pos.sum())
    if n_pos > 50:
        r2_pos = _r2(actual[mask_pos], pred[mask_pos])
        mae_pos = _mae(actual[mask_pos], pred[mask_pos])
        rmse_pos = _rmse(actual[mask_pos], pred[mask_pos])
    else:
        r2_pos = mae_pos = rmse_pos = float("nan")

    return {
        "n": n,
        "n_actual_gt_0": n_pos,
        "r2": r2,
        "r2_actual_gt_0": r2_pos,
        "rmse": rmse,
        "rmse_actual_gt_0": rmse_pos,
        "mae": mae,
        "mae_actual_gt_0": mae_pos,
        "mean_bias": mean_bias,
        "wape_pct": wape,
        "smape_pct": smape,
        "mape_pct_actual_gt_0.05": mape_nonzero,
        "corr": corr,
    }


def _print_accuracy_block(m: dict[str, float], holdout: int) -> None:
    print()
    print("──────── Model accuracy / error (vs actuals) ────────")
    print(f"  Holdout month:           {holdout}")
    print(f"  Rows scored:             {m['n']:,}")
    print()
    print("  R² (variance explained):  {:.4f}   ← 1.0 = perfect; higher is better".format(m["r2"]))
    print(
        "  As percentage (≈):       {:.1f}%   of variance in actuals explained by the model".format(
            100.0 * m["r2"]
        )
    )
    print("  Correlation(actual,pred): {:.4f}".format(m["corr"]))
    print()
    print("  RMSE:                     {:.5f}".format(m["rmse"]))
    print("  MAE:                      {:.5f}".format(m["mae"]))
    print("  Mean bias (pred−actual): {:.5f}".format(m["mean_bias"]))
    print()
    if not np.isnan(m["r2_actual_gt_0"]):
        print(
            "  Where actual > 0 only ({} rows, sparse-demand is hard):".format(
                m["n_actual_gt_0"]
            )
        )
        print("    R²:                     {:.4f}".format(m["r2_actual_gt_0"]))
        print("    RMSE:                   {:.5f}".format(m["rmse_actual_gt_0"]))
        print("    MAE:                    {:.5f}".format(m["mae_actual_gt_0"]))
        print()
    print("  WAPE:                     {:.2f}%   ( Σ|error| / Σ|actual| )".format(m["wape_pct"]))
    print("    ↑ inflated when most actuals are 0; prefer R²/MAE above.")
    print("  sMAPE:                    {:.2f}%".format(m["smape_pct"]))
    print(
        "  MAPE (actual > 0.05):     {:.2f}%".format(m["mape_pct_actual_gt_0.05"])
    )
    print()
    print("  Note: Regression on monthly counts — not classification accuracy.")
    print("──────────────────────────────────────────────────────")


def chart_actual_vs_predicted(actual: np.ndarray, pred: np.ndarray, out: Path, max_points: int = 40000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    n = len(actual)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        a, p = actual[idx], pred[idx]
    else:
        a, p = actual, pred

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(a, p, s=2, alpha=0.25, c="#1d4ed8")
    lim = max(float(np.max(a)), float(np.max(p)), 20.0)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="perfect")
    ax.set_xlabel("Actual item_cnt_month")
    ax.set_ylabel("Predicted")
    ax.set_title("Holdout: actual vs predicted (subsampled if large)")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_error_histogram(actual: np.ndarray, pred: np.ndarray, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = pred - actual
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err, bins=80, color="#64748b", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#ea580c", linestyle="--", linewidth=1.5)
    ax.set_xlabel("predicted − actual")
    ax.set_ylabel("count")
    ax.set_title("Distribution of prediction error (holdout)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_error_by_shop(summary: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary.sort_values("shop_id")
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(s))
    w = 0.35
    ax.bar(x - w / 2, s["actual_sum"], width=w, label="actual sum", color="#1d4ed8")
    ax.bar(x + w / 2, s["predicted_sum"], width=w, label="predicted sum", color="#ea580c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(s["shop_id"].astype(str), rotation=90, fontsize=6)
    ax.set_xlabel("shop_id")
    ax.set_ylabel("total units (month holdout)")
    ax.set_title("Shop-wise: actual vs predicted totals (holdout month)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(
    retrain_holdout: bool,
    holdout_month: int | None,
) -> int:
    holdout = holdout_month if holdout_month is not None else LAST_HIST_MONTH
    if holdout <= FIRST_FEATURE_MONTH:
        print(f"holdout month must be > {FIRST_FEATURE_MONTH}", file=sys.stderr)
        return 1

    print("Loading monthly matrix (preprocess)...", flush=True)
    data = preprocess()
    wide = data["wide"]
    items = data["items"]

    if retrain_holdout:
        print(
            f"Retrain-holdout mode: building ~{(holdout - FIRST_FEATURE_MONTH) * len(wide):,} "
            "training rows — this takes several minutes.",
            flush=True,
        )
        print("  Training feature months...", flush=True)
        train = features_for_month_range(
            wide,
            items,
            FIRST_FEATURE_MONTH,
            holdout - 1,
            progress=True,
        )
        print("  Holdout month features...", flush=True)
        val = features_for_month(wide, items, holdout)
        model = _make_model()
        print("  Fitting model...", flush=True)
        model.fit(train[FEATURE_COLUMNS], train["item_cnt_month"])
        mode_note = (
            f"Holdout: trained on months {FIRST_FEATURE_MONTH}–{holdout - 1}, "
            f"evaluated on month {holdout} only (honest out-of-sample)."
        )
    else:
        model_path = MODEL_DIR / "model.pkl"
        if not model_path.is_file():
            print(
                f"Missing {model_path}. Run: python train.py\n"
                "Or use: python evaluate.py --retrain-holdout",
                file=sys.stderr,
            )
            return 1
        print("Fast mode: scoring saved model on holdout month only (~526k rows)...", flush=True)
        val = features_for_month(wide, items, holdout)
        model = joblib.load(model_path)
        mode_note = (
            "Using saved model.pkl on month "
            f"{holdout} (in-sample if that month was used in training — fast path)."
        )

    pred = np.clip(model.predict(val[FEATURE_COLUMNS]), CLIP_MIN, CLIP_MAX)
    actual = val["item_cnt_month"].to_numpy(dtype=np.float64)

    val = val.copy()
    val["actual"] = actual
    val["predicted"] = pred
    val["error"] = pred - actual
    val["abs_error"] = np.abs(val["error"])

    m = _metrics(actual, pred)
    print(mode_note)
    print(f"Month {holdout} ({month_start_date(holdout).strftime('%Y-%m-%d')})")
    _print_accuracy_block(m, holdout)

    detail_cols = [
        "shop_id",
        "item_id",
        "date_block_num",
        "actual",
        "predicted",
        "error",
        "abs_error",
    ]
    detail_path = DATA_DIR / "evaluation_detail.csv"
    val[detail_cols].sort_values(["shop_id", "item_id"]).to_csv(detail_path, index=False)

    by_shop = val.groupby("shop_id", sort=True).agg(
        actual_sum=("actual", "sum"),
        predicted_sum=("predicted", "sum"),
    )
    by_shop["diff_sum"] = by_shop["predicted_sum"] - by_shop["actual_sum"]
    by_shop = by_shop.reset_index()
    shop_path = DATA_DIR / "evaluation_by_shop.csv"
    by_shop.to_csv(shop_path, index=False)

    by_item = val.groupby("item_id", sort=True).agg(
        actual_sum=("actual", "sum"),
        predicted_sum=("predicted", "sum"),
    )
    by_item["diff_sum"] = by_item["predicted_sum"] - by_item["actual_sum"]
    by_item = by_item.reset_index()
    item_path = DATA_DIR / "evaluation_by_item.csv"
    by_item.to_csv(item_path, index=False)

    print("Rendering charts...", flush=True)
    chart_actual_vs_predicted(actual, pred, CHARTS_DIR / "07_eval_actual_vs_predicted.png")
    chart_error_histogram(actual, pred, CHARTS_DIR / "08_eval_error_histogram.png")
    chart_error_by_shop(by_shop, CHARTS_DIR / "09_eval_shop_actual_vs_pred_totals.png")

    print(f"Detail: {detail_path}")
    print(f"By shop: {shop_path}")
    print(f"By item: {item_path}")
    print(f"Charts: {CHARTS_DIR}/07_*.png, 08_*.png, 09_*.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare predictions to actuals (holdout month). Default is fast (saved model)."
    )
    p.add_argument(
        "--retrain-holdout",
        action="store_true",
        help=(
            "Retrain on months before holdout, then score (slow, honest OOS). "
            "Default uses model.pkl and only builds the holdout month (fast)."
        ),
    )
    p.add_argument(
        "--holdout-month",
        type=int,
        default=None,
        help=f"Month index to evaluate (default: {LAST_HIST_MONTH}).",
    )
    args = p.parse_args(argv)
    try:
        return run(retrain_holdout=args.retrain_holdout, holdout_month=args.holdout_month)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
