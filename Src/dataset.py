"""Single entry: load raw data → wide monthly matrix → long modeling frame."""
from features import create_features
from preprocess import preprocess


def load_modeling_frame():
    """Return the full long-format table used for training and prediction."""
    data = preprocess()
    return create_features(data["wide"], data["items"], data["test"])
