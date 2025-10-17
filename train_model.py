from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"
SLICE_FILE = ROOT / "slice_output.txt"

# ---- Load data ----
print(f"Using data file: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ---- Train/Test split ----
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["salary"]
)

# ---- Categorical features (DO NOT MODIFY) ----
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

# ---- Process data ----
X_train, y_train, encoder, lb = process_data(
    train_df,
    categorical_features=CAT_FEATURES,
    label="salary",
    training=True,
)
X_test, y_test, _, _ = process_data(
    test_df,
    categorical_features=CAT_FEATURES,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# ---- Train & Save ----
model = train_model(X_train, y_train)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / "model.pkl"
encoder_path = MODEL_DIR / "encoder.pkl"
lb_path = MODEL_DIR / "label_binarizer.pkl"

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

# ---- Reload (sanity check) ----
model = load_model(model_path)

# ---- Inference & overall metrics ----
preds = inference(model, X_test)
p, r, f1 = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

# ---- Slice metrics ----
with SLICE_FILE.open("w") as f:
    for col in CAT_FEATURES:
        # deterministic ordering of slice values
        for val in sorted(test_df[col].unique()):
            count = (test_df[col] == val).sum()
            sp, sr, sf = performance_on_categorical_slice(
                data=test_df,
                column_name=col,
                slice_value=val,
                categorical_features=CAT_FEATURES,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model,
            )
            print(f"{col}: {val}, Count: {count:,}", file=f)
            print(f"Precision: {sp:.4f} | Recall: {sr:.4f} | F1: {sf:.4f}", file=f)
            print("-" * 40, file=f)

print(f"Wrote slice metrics to: {SLICE_FILE}")
