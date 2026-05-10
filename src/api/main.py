import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

import src.api.metrics as state
from src.api.routers import inference, monitoring
from src.config.config import MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("accidents_api")

_TAGS = [
    {"name": "monitoring", "description": "Santé et métriques opérationnelles de l'API."},
    {"name": "inference", "description": "Prédiction de la gravité d'un accident de la route."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import time
    state.startup_time = time.time()
    try:
        if MODEL_PATH.exists():
            state.ml_model["classifier"] = joblib.load(MODEL_PATH)
            logger.info(
                "model_loaded=true type=%s path=%s",
                type(state.ml_model["classifier"]).__name__,
                MODEL_PATH,
            )
        else:
            state.ml_model["classifier"] = None
            logger.warning("model_loaded=false path=%s — mode dégradé", MODEL_PATH)
    except Exception as exc:
        state.ml_model["classifier"] = None
        logger.error("model_load_error=%s", exc, exc_info=True)
    yield
    state.ml_model.clear()
    logger.info("shutdown complete")


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
    contact={"name": "Ibrahima Wane", "email": "ibrahima.wane@outlook.fr"},
    license_info={"name": "MIT"},
    openapi_tags=_TAGS,
    lifespan=lifespan,
)

app.include_router(monitoring.router)
app.include_router(inference.router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
