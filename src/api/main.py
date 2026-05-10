import logging
import os
import subprocess
import threading
import time as _time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from src.config.config import FEATURES, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("accidents_api")

# ── Prometheus metrics ────────────────────────────────────────────────────────

PREDICTION_COUNTER = Counter(
    "accidents_predictions_total",
    "Total number of predictions made",
    ["label", "confidence"],
)
PREDICTION_LATENCY = Histogram(
    "accidents_prediction_latency_seconds",
    "Time spent processing a prediction request",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_LOADED_GAUGE = Gauge(
    "accidents_model_loaded",
    "Whether the ML model is currently loaded (1=yes, 0=no)",
)
RETRAIN_COUNTER = Counter(
    "accidents_retrain_requests_total",
    "Total number of retraining requests triggered",
)

# ── App state ─────────────────────────────────────────────────────────────────

ml_model: dict = {}
_startup_time: float = 0.0
_stats_lock = threading.Lock()
_stats: dict = {"total": 0, "prioritaire": 0, "non_prioritaire": 0}

_TAGS = [
    {"name": "monitoring", "description": "Santé, métriques et opérations."},
    {"name": "inference", "description": "Prédiction de la gravité d'un accident."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = _time.time()
    try:
        if MODEL_PATH.exists():
            ml_model["classifier"] = joblib.load(MODEL_PATH)
            MODEL_LOADED_GAUGE.set(1)
            logger.info("model_loaded=true type=%s path=%s", type(ml_model["classifier"]).__name__, MODEL_PATH)
        else:
            ml_model["classifier"] = None
            MODEL_LOADED_GAUGE.set(0)
            logger.warning("model_loaded=false path=%s — mode dégradé", MODEL_PATH)
    except Exception as exc:
        ml_model["classifier"] = None
        MODEL_LOADED_GAUGE.set(0)
        logger.error("model_load_error=%s", exc, exc_info=True)
    yield
    ml_model.clear()
    logger.info("shutdown complete")


app = FastAPI(
    title="Accidents Routiers — API de prédiction",
    description=(
        "Classifie la gravité d'un accident de la route en deux catégories :\n\n"
        "- **1 — prioritaire** : victime hospitalisée ou décédée\n"
        "- **0 — non-prioritaire** : victime indemne ou blessée légèrement\n\n"
        "Modèle : **Random Forest Classifier** entraîné sur les données BAAC 2021 "
        "(France métropolitaine).\n\n"
        "Monitoring : métriques Prometheus exposées sur `/metrics`, "
        "dashboard Grafana disponible sur `http://localhost:3000`."
    ),
    version="1.0.0",
    contact={"name": "Ibrahima Wane", "email": "ibrahima.wane@outlook.fr"},
    license_info={"name": "MIT"},
    openapi_tags=_TAGS,
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class AccidentFeatures(BaseModel):
    place: float
    catu: float
    sexe: float
    secu1: float
    year_acc: float
    victim_age: float
    catv: float
    obsm: float
    motor: float
    catr: float
    circ: float
    surf: float
    situ: float
    vma: float
    jour: float
    mois: float
    lum: float
    dep: float
    com: float
    agg_: float
    int: float
    atm: float
    col: float
    lat: float
    long: float
    hour: float
    nb_victim: float
    nb_vehicules: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
                "year_acc": 2021, "victim_age": 60, "catv": 2, "obsm": 1,
                "motor": 1, "catr": 3, "circ": 2, "surf": 1, "situ": 1,
                "vma": 50, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
                "com": 77317, "agg_": 2, "int": 1, "atm": 0, "col": 6,
                "lat": 48.60, "long": 2.89, "hour": 17,
                "nb_victim": 2, "nb_vehicules": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    confidence: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]
    n_features: int
    uptime_seconds: float
    api_version: str


class ModelInfoResponse(BaseModel):
    type: str
    n_features: int
    features: list
    params: dict


class StatsResponse(BaseModel):
    total_predictions: int
    predictions_by_label: dict
    uptime_seconds: float


# ── Background task ───────────────────────────────────────────────────────────

def _run_training() -> None:
    logger.info("retrain=started")
    try:
        result = subprocess.run(
            ["python3", "src/models/train_model.py"],
            env={**os.environ, "PYTHONPATH": "."},
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0 and MODEL_PATH.exists():
            ml_model["classifier"] = joblib.load(MODEL_PATH)
            MODEL_LOADED_GAUGE.set(1)
            logger.info("retrain=success")
        else:
            logger.error("retrain=failed stderr=%s", result.stderr[:500])
    except Exception as exc:
        logger.error("retrain=error error=%s", exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    model = ml_model.get("classifier")
    model_loaded = model is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=type(model).__name__ if model_loaded else None,
        n_features=len(FEATURES),
        uptime_seconds=round(_time.time() - _startup_time, 2),
        api_version=app.version,
    )


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats", response_model=StatsResponse, tags=["monitoring"])
def stats():
    """Compteurs JSON lisibles par un humain."""
    with _stats_lock:
        return StatsResponse(
            total_predictions=_stats["total"],
            predictions_by_label={
                "prioritaire": _stats["prioritaire"],
                "non_prioritaire": _stats["non_prioritaire"],
            },
            uptime_seconds=round(_time.time() - _startup_time, 2),
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["monitoring"])
def model_info():
    """Métadonnées du modèle chargé en mémoire."""
    model = ml_model.get("classifier")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return ModelInfoResponse(
        type=type(model).__name__,
        n_features=len(FEATURES),
        features=FEATURES,
        params=model.get_params() if hasattr(model, "get_params") else {},
    )


@app.post("/retrain", status_code=202, tags=["monitoring"])
def retrain(background_tasks: BackgroundTasks):
    """Déclenche un ré-entraînement du modèle en arrière-plan."""
    RETRAIN_COUNTER.inc()
    background_tasks.add_task(_run_training)
    logger.info("retrain=requested")
    return {"status": "accepted", "message": "Retraining started in background."}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: AccidentFeatures):
    if ml_model.get("classifier") is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Lancez d'abord l'entraînement.",
        )

    t0 = _time.time()

    input_data = pd.DataFrame([features.model_dump()])[FEATURES]
    prediction = int(ml_model["classifier"].predict(input_data)[0])
    proba = ml_model["classifier"].predict_proba(input_data)[0]
    probability = round(float(np.max(proba)), 4)

    duration = _time.time() - t0

    if probability >= 0.80:
        confidence = "high"
    elif probability >= 0.60:
        confidence = "medium"
    else:
        confidence = "low"

    label = "prioritaire" if prediction == 1 else "non-prioritaire"

    PREDICTION_COUNTER.labels(label=label, confidence=confidence).inc()
    PREDICTION_LATENCY.observe(duration)

    with _stats_lock:
        _stats["total"] += 1
        if prediction == 1:
            _stats["prioritaire"] += 1
        else:
            _stats["non_prioritaire"] += 1

    logger.info(
        "prediction=%d label=%s probability=%.4f confidence=%s latency=%.4fs",
        prediction, label, probability, confidence, duration,
    )

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability=probability,
        confidence=confidence,
    )
