from typing import Optional

from pydantic import BaseModel


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
