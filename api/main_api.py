from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.models.train_model import train 
from fastapi import BackgroundTasks
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
import mlflow
import mlflow.pyfunc
from fastapi import HTTPException

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Accidents Routiers — API de prédiction",
    description=(
        "Classifie la gravité d'un accident de la route en deux catégories :\n\n"
        "- **1 — prioritaire** : victime hospitalisée ou décédée\n"
        "- **0 — non-prioritaire** : victime indemne ou blessée légèrement\n\n"
        "Modèle : **Random Forest Classifier** entraîné sur les données BAAC 2021 "
        "(France métropolitaine)."
    ),
    version="1.0.0",
    contact={"url": "https://github.com/waneib22/mlops-accidents/issues"},
    license_info={"name": "MIT"},
)


# ─────────────────────────────────────────
# Chargement modèle et données (une seule fois au démarrage)
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# MODEL_PATH  = BASE_DIR / "src" / "models" / "trained_model.joblib"
#model  = joblib.load(MODEL_PATH)

#je charge le meilleur modele de mlflow model registry:
#mlflow.set_tracking_uri("http://localhost:8080") #api lancé avec make api hors docker
#mlflow.set_tracking_uri("http://mlflow:8080")  #API lancée dans Docker Compose 
mlflow.set_tracking_uri("https://dagshub.com/Melanie94480/mlops-melanie.mlflow") #recuperera le dernier modele mlflow registry sur dagshub


#Utiliser la dernière version enregistrée:
model = mlflow.pyfunc.load_model("models:/Modele Random Forest/latest") #si version modele specifique : model = mlflow.pyfunc.load_model("models:/Modele Random Forest /1")  #version 1



X_TEST_PATH = BASE_DIR / "data" / "preprocessed" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "preprocessed" / "y_test.csv"


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
    #return {"message": "API active"}
    return RedirectResponse(url="/docs")    


# ─────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────
@app.post("/predict", tags=["Prédiction"])
def predict(data: InputData):

    if model is None:
        raise HTTPException(status_code=503,detail="Model unavailable")

    X = pd.DataFrame([data.model_dump()])
    pred = model.predict(X)
    prediction = int(pred[0])
    label = "prioritaire" if prediction == 1 else "non-prioritaire"
    return {"prediction": prediction, "label": label}

    #model correspond à la dernière version de mlflow



# ─────────────────────────────────────────
# RÉ-ENTRAÎNEMENT
# ─────────────────────────────────────────
def run_retraining():
    global model

    try:
        print("[retrain] Début de l'entraînement...")
        train() # Entraînement + enregistrement dans MLflow du modèle 

        # Recharge le dernier modèle pour metrics/predict ...
        model = mlflow.pyfunc.load_model("models:/Modele Random Forest/latest") # ==> nouveau modele suite au réentrainement
        print("[retrain] Nouveau modèle chargé.")

    except Exception as e:
        print(f"[retrain] Erreur : {e}")


@app.post("/retrain", tags=["Ré-entraînement"], status_code=202)
def retrain(background_tasks: BackgroundTasks): #entrainement en arrièere plan
    background_tasks.add_task(run_retraining)
    return {
        "status": "accepted",
        "message": "Retraining lancé."
    }

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
@app.get("/health", tags=["Health"])
def health():
    status = {
        "api": "ok",
        "model_loaded": model is not None,
        "mlflow_tracking": mlflow.get_tracking_uri(),
        "service": "accidents-api",
        "version": "1.0.0",
        "message": "je suis une API qui fonctionne au top de sa forme 😏"

    }
    return status


# Lancer : uvicorn api.main_api:app --reload


# ─────────────────────────────────────────
# Monitoring prometheus /Grafana
# ─────────────────────────────────────────
Instrumentator().instrument(app).expose(
    app,
    endpoint="/monitoring",
    include_in_schema=True,
    tags=["Monitoring Prometheus"],
)
