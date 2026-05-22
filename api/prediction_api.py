from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path

app = FastAPI(
    title="Accidents ML API predictions",
    version="1.0"
)

# Charger modèle
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "src" / "models" / "trained_model.joblib"
print("MODEL PATH =", MODEL_PATH)

model = joblib.load(MODEL_PATH)


# INPUT DATA (sous le format du fichier test_features.json données sur github)
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
    lum : int
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

# activation api
@app.get("/")
def home():
    return {
        "message": "Prediction API active"
    }


# PREDICTION
@app.post("/predict")
def predict(data: InputData):

    X = np.array([[  
        data.place, data.catu, data.sexe, data.secu1,
        data.year_acc, data.victim_age, data.catv, data.obsm,
        data.motor, data.catr, data.circ, data.surf,
        data.situ, data.vma, data.jour, data.mois, data.lum,
        data.dep, data.com, data.agg_, data.int,
        data.atm, data.col, data.lat, data.long,
        data.hour, data.nb_victim, data.nb_vehicules
    ]])

    pred = model.predict(X)

    return {"prediction": int(pred[0])}


#tester api : uvicorn api.prediction_api:app --reload


#Rappel :
#prediction resultat10:52Claude a répondu : La prédiction retourne la gravité de l'accident.La prédiction retourne la gravité de l'accident. Dans le dataset accidents routiers français, la variable cible est grav :

# 1 : indemne 
# 2 : tué
# 3 : blessé hospitalisé
# 4 : blessé léger
