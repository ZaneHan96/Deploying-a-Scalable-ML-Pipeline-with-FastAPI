import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# ---- Load artifacts ----
project_path = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_path, "model")

# encoder (for categorical features)
path = os.path.join(model_dir, "encoder.pkl")
encoder = load_model(path)

# trained model
path = os.path.join(model_dir, "model.pkl")
model = load_model(path)

# ---- FastAPI app ----
app = FastAPI(title="Census Income Classifier", version="1.0.0")


# GET root greeting
@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Welcome to the FastAPI ML model!"}


# POST inference endpoint
@app.post("/predict")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # process_data in inference mode (no label column provided)
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None,  # label binarizer not needed when no label provided
    )

    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}
