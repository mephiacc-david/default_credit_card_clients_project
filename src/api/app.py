"""REST API для предсказания вероятности дефолта."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Credit Default Prediction API")
MODEL_PATH = Path("models/credit_default_model.joblib")


class ClientData(BaseModel):
    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_0: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


@lru_cache
def get_model():
    return joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Credit Default Prediction API"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(client: ClientData, model=Depends(get_model)):
    input_data = pd.DataFrame([client.model_dump()])
    probability = float(model.predict_proba(input_data)[0, 1])
    return {
        "default_prediction": int(probability >= 0.5),
        "default_probability": probability,
    }
