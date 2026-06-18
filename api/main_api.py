from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Accidents MLOps - API principale",
    description="Regroupe les endpoints de test, prédiction et métriques",
    version="1.0"
)

Instrumentator().instrument(app).expose(app)

# ─────────────────────────────────────────
# Chargement modèle et données (une seule fois au démarrage)
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "src" / "models" / "trained_model.joblib"
X_TEST_PATH = BASE_DIR / "data" / "preprocessed" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "preprocessed" / "y_test.csv"

model  = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()


# ─────────────────────────────────────────
# Schéma de données pour la prédiction
# ─────────────────────────────────────────
class InputData(BaseModel):
    place: int
    catu: int
    sexe: int
    secu1: float
    year_acc: int
    victim_age: int
    catv: int
    obsm: int
    motor: int
    catr: int
    circ: int
    surf: int
    situ: int
    vma: int
    jour: int
    mois: int
    lum: int
    dep: int
    com: int
    agg_: int
    int: int
    atm: int
    col: int
    lat: float
    long: float
    hour: int
    nb_victim: int
    nb_vehicules: int


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
@app.get("/", tags=["Test"])
def home():
    return {"message": "API active"}


# ─────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────
@app.post("/predict", tags=["Prédiction"])
def predict(data: InputData):
    X = pd.DataFrame([data.model_dump()])
    pred = model.predict(X)
    return {"prediction": int(pred[0])}


# ─────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────
@app.get("/metrics", tags=["Métriques"])
def get_metrics():
    y_pred = model.predict(X_test)
    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall":    float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score":  float(f1_score(y_test, y_pred, average="weighted")),
    }


@app.get("/report", tags=["Métriques"])
def report():
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)


# ─────────────────────────────────────────
# Health api (bonne pratique)
# ─────────────────────────────────────────
@app.get("/health", tags=["Test"])
def health():
    return {"status": "je suis une API qui fonctionne au top de sa forme 😏"}


# Lancer : uvicorn api.main_api:app --reload
