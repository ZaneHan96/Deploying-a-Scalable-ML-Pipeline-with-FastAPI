import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import process_data, apply_label
from ml.model import load_model, inference


class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., alias="education-num", example=10)
    marital_status: str = Field(..., alias="marital-status", example="Married-civ-spouse")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")


ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")

encoder = load_model(os.path.join(MODEL_DIR, "encoder.pkl"))
model = load_model(os.path.join(MODEL_DIR, "model.pkl"))

app = FastAPI(title="Census Income API", version="1.0.0")


@app.get("/")
async def root():
    """Simple welcome route."""
    return {"message": "Welcome to the Census Income FastAPI!"}


@app.post("/predict")
async def predict(data: Data):
    """Run model inference and return salary prediction."""
    data_dict = data.dict()
    data_dict = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(data_dict)

    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    X, _, _, _ = process_data(
        df, categorical_features=cat_features, label=None,
        training=False, encoder=encoder, lb=None
    )

    pred = inference(model, X)
    return {"result": apply_label(pred)}
