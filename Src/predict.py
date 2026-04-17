"""Load trained model and write Data/submission.csv for TEST_MONTH."""
import joblib
import numpy as np

from config import DATA_DIR, FEATURE_COLUMNS, MODEL_DIR, TEST_MONTH
from dataset import load_modeling_frame


def main() -> None:
    model = joblib.load(MODEL_DIR / "model.pkl")
    df = load_modeling_frame()
    test_rows = df[df["date_block_num"] == TEST_MONTH].copy()
    if "ID" not in test_rows.columns:
        raise RuntimeError("Expected test merge to include ID column")

    pred = np.clip(model.predict(test_rows[FEATURE_COLUMNS]), 0, 20)
    sub = test_rows[["ID"]].copy()
    sub["item_cnt_month"] = pred
    sub = sub.sort_values("ID")

    out_path = DATA_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"Wrote {len(sub)} rows to {out_path}")


if __name__ == "__main__":
    main()
