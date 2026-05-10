from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "src" / "models"
MODEL_PATH = MODELS_DIR / "trained_model.joblib"

RAW_FILES = {
    "users": "usagers-2021.csv",
    "caract": "caracteristiques-2021.csv",
    "places": "lieux-2021.csv",
    "vehicles": "vehicules-2021.csv",
}

S3_BASE_URL = "https://mlops-project-db.s3.eu-west-1.amazonaws.com/accidents/"

FEATURES = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age",
    "catv", "obsm", "motor", "catr", "circ", "surf", "situ", "vma",
    "jour", "mois", "lum", "dep", "com", "agg_", "int", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]

TEST_SIZE = 0.3
RANDOM_STATE = 42
