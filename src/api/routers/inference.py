import logging
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.metrics import (
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    ml_model,
    stats,
    stats_lock,
)
from src.api.schemas import AccidentFeatures, PredictionResponse
from src.config.config import FEATURES

logger = logging.getLogger("accidents_api")

router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
def predict(features: AccidentFeatures):
    if ml_model.get("classifier") is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Lancez d'abord l'entraînement.",
        )

    t0 = time.time()

    input_data = pd.DataFrame([features.model_dump()])[FEATURES]
    prediction = int(ml_model["classifier"].predict(input_data)[0])
    proba = ml_model["classifier"].predict_proba(input_data)[0]
    probability = round(float(np.max(proba)), 4)

    duration = time.time() - t0

    if probability >= 0.80:
        confidence = "high"
    elif probability >= 0.60:
        confidence = "medium"
    else:
        confidence = "low"

    label = "prioritaire" if prediction == 1 else "non-prioritaire"

    PREDICTION_COUNTER.labels(label=label, confidence=confidence).inc()
    PREDICTION_LATENCY.observe(duration)

    with stats_lock:
        stats["total"] += 1
        if prediction == 1:
            stats["prioritaire"] += 1
        else:
            stats["non_prioritaire"] += 1

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
