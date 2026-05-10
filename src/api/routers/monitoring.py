import logging
import os
import subprocess
import time

import joblib
from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.metrics import ml_model, startup_time, stats, stats_lock
from src.api.schemas import HealthResponse, ModelInfoResponse, StatsResponse
from src.config.config import FEATURES, MODEL_PATH

logger = logging.getLogger("accidents_api")

router = APIRouter(tags=["monitoring"])


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
            logger.info("retrain=success")
        else:
            logger.error("retrain=failed stderr=%s", result.stderr[:500])
    except Exception as exc:
        logger.error("retrain=error error=%s", exc)


@router.get("/health", response_model=HealthResponse)
def health():
    model = ml_model.get("classifier")
    model_loaded = model is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=type(model).__name__ if model_loaded else None,
        n_features=len(FEATURES),
        uptime_seconds=round(time.time() - startup_time, 2),
        api_version="1.0.0",
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    with stats_lock:
        return StatsResponse(
            total_predictions=stats["total"],
            predictions_by_label={
                "prioritaire": stats["prioritaire"],
                "non_prioritaire": stats["non_prioritaire"],
            },
            uptime_seconds=round(time.time() - startup_time, 2),
        )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    model = ml_model.get("classifier")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return ModelInfoResponse(
        type=type(model).__name__,
        n_features=len(FEATURES),
        features=FEATURES,
        params=model.get_params() if hasattr(model, "get_params") else {},
    )


@router.post("/retrain", status_code=202)
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_training)
    logger.info("retrain=requested")
    return {"status": "accepted", "message": "Retraining started in background."}
