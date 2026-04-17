import argparse

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from config import FEATURE_COLUMNS, FIRST_FEATURE_MONTH, LAST_HIST_MONTH, MODEL_DIR
from dataset import load_modeling_frame


def _make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.06,
        max_depth=10,
        max_leaf_nodes=96,
        min_samples_leaf=50,
        l2_regularization=0.05,
        random_state=42,
    )


def train_model(validate: bool = False) -> None:
    df = load_modeling_frame()

    if validate:
        train = df[
            (df["date_block_num"] >= FIRST_FEATURE_MONTH) & (df["date_block_num"] <= LAST_HIST_MONTH - 1)
        ]
        val = df[df["date_block_num"] == LAST_HIST_MONTH]
        X_tr, y_tr = train[FEATURE_COLUMNS], train["item_cnt_month"]
        X_val, y_val = val[FEATURE_COLUMNS], val["item_cnt_month"]
        model = _make_model()
        model.fit(X_tr, y_tr)
        pred = np.clip(model.predict(X_val), 0, 20)
        rmse = float(np.sqrt(np.mean((pred - y_val.values) ** 2)))
        print(f"Validation RMSE (month {LAST_HIST_MONTH}): {rmse:.4f}")
    else:
        train = df[
            (df["date_block_num"] >= FIRST_FEATURE_MONTH) & (df["date_block_num"] <= LAST_HIST_MONTH)
        ]
        X_tr, y_tr = train[FEATURE_COLUMNS], train["item_cnt_month"]
        model = _make_model()
        model.fit(X_tr, y_tr)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "model.pkl"
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--validate",
        action="store_true",
        help=f"Hold out month {LAST_HIST_MONTH} for RMSE",
    )
    args = p.parse_args()
    train_model(validate=args.validate)
