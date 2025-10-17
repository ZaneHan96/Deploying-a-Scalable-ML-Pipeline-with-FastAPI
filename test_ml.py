from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    performance_on_categorical_slice,
    compute_model_metrics,
)

# Keep in sync with training
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

DATA_PATH = Path(__file__).resolve().parent / "data" / "census.csv"


def _load_subset(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Load a small, deterministic subset to keep tests quick."""
    df = pd.read_csv(DATA_PATH)
    if len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=seed)
    return df


def test_process_data_training_mode_shapes():
    """training=True returns aligned X/y and fitted encoder/lb."""
    df = _load_subset()
    train_df, _ = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["salary"]
    )

    X_tr, y_tr, enc, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    assert X_tr.shape[0] == y_tr.shape[0] > 0
    assert hasattr(enc, "categories_")
    assert hasattr(lb, "classes_")
    assert set(np.unique(y_tr)).issubset({0, 1})


def test_train_then_infer_binary_preds_and_shape_match():
    """Train then infer: preds are 0/1 and match test length."""
    df = _load_subset(seed=1)
    tr, te = train_test_split(
        df, test_size=0.2, random_state=1, stratify=df["salary"]
    )
    X_tr, y_tr, enc, lb = process_data(
        tr, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_te, y_te, _, _ = process_data(
        te,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=enc,
        lb=lb,
    )

    model = train_model(X_tr, y_tr)
    preds = inference(model, X_te)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == y_te.shape
    assert set(np.unique(preds)).issubset({0, 1})

    # metrics are computable and in [0, 1]
    p, r, f1 = compute_model_metrics(y_te, preds)
    for v in (p, r, f1):
        assert 0.0 <= v <= 1.0


def test_slice_metrics_with_valid_range():
    """Slice metrics are floats in [0,1] for a real category value."""
    df = _load_subset(seed=2)
    tr, te = train_test_split(
        df, test_size=0.2, random_state=2, stratify=df["salary"]
    )

    X_tr, y_tr, enc, lb = process_data(
        tr, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    model = train_model(X_tr, y_tr)

    col = "sex"
    val = sorted(te[col].unique())[0]

    p, r, f1 = performance_on_categorical_slice(
        data=te,
        column_name=col,
        slice_value=val,
        categorical_features=CAT_FEATURES,
        label="salary",
        encoder=enc,
        lb=lb,
        model=model,
    )
    for v in (p, r, f1):
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
