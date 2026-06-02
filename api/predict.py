from fastapi import APIRouter
import pandas as pd
import joblib
from pathlib import Path

router = APIRouter()

DOSSIER_BASE = Path(__file__).resolve().parent.parent

model = joblib.load(DOSSIER_BASE / "models" / "model.pkl")
model_columns = joblib.load(DOSSIER_BASE / "models" / "model_columns.pkl")

@router.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}