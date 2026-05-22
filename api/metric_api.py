
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
from sklearn.metrics import classification_report

app = FastAPI(
    title="Metrics API",
    version="1.0"
)

# Charger modèle
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "src" / "models" / "trained_model.joblib"
print("MODEL PATH =", MODEL_PATH)
model = joblib.load(MODEL_PATH)

# LOAD TEST DATA
# =========================

X_TEST_PATH = BASE_DIR / "data" / "preprocessed" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "preprocessed" / "y_test.csv"

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH)

# transformer y_test en série
y_test = y_test.squeeze()

# metrics SCORE 
@app.get("/metrics")
def get_metrics():

    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted"))
    }

@app.get("/report")
def report():

    y_pred = model.predict(X_test)

    return classification_report(y_test, y_pred, output_dict=True)

#tester api : uvicorn api.metric_api:app --reload