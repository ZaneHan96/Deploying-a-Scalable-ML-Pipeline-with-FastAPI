import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
)

from ml.data import process_data


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Train and return a logistic regression model."""
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=None,
    )
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(
    y: np.ndarray,
    preds: np.ndarray,
) -> Tuple[float, float, float]:
    """Validate model predictions using precision, recall, and F1."""
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(
    model: LogisticRegression,
    X: np.ndarray,
) -> np.ndarray:
    """Run model inference and return predictions."""
    return model.predict(X)


def save_model(model, path: str) -> None:
    """Save a Python object (model, encoder, or label binarizer)."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    """Load a pickle file and return the deserialized object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data: pd.DataFrame,
    column_name: str,
    slice_value,
    categorical_features,
    label: str,
    encoder,
    lb,
    model,
) -> Tuple[float, float, float]:
    """Compute metrics for a slice where column == slice_value."""
    sliced = data[data[column_name] == slice_value].copy()
    if sliced.empty:
        return 0.0, 0.0, 0.0

    X_slice, y_slice, _, _ = process_data(
        sliced,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_slice)
    precision = precision_score(y_slice, preds, zero_division=1)
    recall = recall_score(y_slice, preds, zero_division=1)
    fbeta = fbeta_score(y_slice, preds, beta=1, zero_division=1)
    return precision, recall, fbeta
