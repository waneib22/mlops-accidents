# Accidents Routiers — MLOps Pipeline

![CI](https://github.com/ibrahimawane/mlops-accidents-routiers/actions/workflows/python-app.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Prédiction de la gravité d'accidents de la route français (données BAAC 2021) — classification binaire exposée via une API REST.

| Classe | Signification |
|--------|--------------|
| **1 — prioritaire** | Victime hospitalisée ou décédée |
| **0 — non-prioritaire** | Victime indemne ou blessée légèrement |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  accidents_api  (python:3.10-slim)               │   │
│  │                                                  │   │
│  │  GET  /            → redirect /docs              │   │
│  │  GET  /health      → statut + métriques modèle   │   │
│  │  GET  /metrics     → compteurs de prédictions    │   │
│  │  POST /predict     → inférence RandomForest      │   │
│  │                                                  │   │
│  │  :8000                                           │   │
│  └──────────────────────────────────────────────────┘   │
│          │                       │                      │
│     ./data:/app/data    ./src/models:/app/src/models    │
└─────────────────────────────────────────────────────────┘
```

```
Template_MLOps_accidents/
├── src/
│   ├── api/
│   │   └── main.py          ← FastAPI app
│   ├── config/
│   │   └── config.py        ← chemins, features, constantes
│   ├── data/
│   │   ├── make_dataset.py  ← preprocessing pipeline
│   │   └── import_raw_data.py
│   └── models/
│       └── train_model.py   ← entraînement RandomForest
├── tests/
│   ├── test_api.py          ← 12 tests API
│   └── test_data.py         ← 11 tests pipeline data
├── .github/workflows/
│   └── python-app.yml       ← CI : lint → test (coverage ≥ 60%)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Quick Start

### Option 1 — Make (recommandé)

```bash
make install      # installe les dépendances
make train        # entraîne le modèle (nécessite les données)
make api          # démarre l'API sur http://localhost:8000
```

### Option 2 — Docker

```bash
docker compose up --build
# API disponible sur http://localhost:8000
```

### Option 3 — Manuel

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 src/models/train_model.py
PYTHONPATH=. uvicorn src.api.main:app --reload
```

---

## API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Redirige vers `/docs` |
| `GET` | `/docs` | Documentation interactive (Swagger UI) |
| `GET` | `/health` | Statut de l'API et du modèle |
| `GET` | `/metrics` | Compteurs de prédictions en mémoire |
| `POST` | `/predict` | Prédiction de gravité |

### Exemple `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
    "year_acc": 2021, "victim_age": 60, "catv": 2, "obsm": 1,
    "motor": 1, "catr": 3, "circ": 2, "surf": 1, "situ": 1,
    "vma": 50, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
    "com": 77317, "agg_": 2, "int": 1, "atm": 0, "col": 6,
    "lat": 48.60, "long": 2.89, "hour": 17,
    "nb_victim": 2, "nb_vehicules": 1
  }'
```

```json
{
  "prediction": 1,
  "label": "prioritaire",
  "probability": 0.8423,
  "confidence": "high"
}
```

---

## Pipeline de données

```
data/raw/
├── usagers-2021.csv
├── caracteristiques-2021.csv
├── lieux-2021.csv
└── vehicules-2021.csv
         │
         ▼  src/data/make_dataset.py
data/processed/
├── X_train.csv   (70%)
├── X_test.csv    (30%)
├── y_train.csv
└── y_test.csv
         │
         ▼  src/models/train_model.py
src/models/trained_model.joblib
```

Les données brutes sont disponibles sur le bucket S3 DataScientest :
`https://mlops-project-db.s3.eu-west-1.amazonaws.com/accidents/`

---

## Développement

```bash
make lint     # vérification flake8
make test     # pytest + coverage (rapport HTML dans htmlcov/)
make clean    # nettoyage des artefacts
```

### Lancer les tests

```bash
make test
# ou directement :
PYTHONPATH=. python3 -m pytest tests/ -v
```

Résultat attendu : **23 tests passants**, coverage > 60%.

---

## Modèle

| Paramètre | Valeur |
|-----------|--------|
| Algorithme | `RandomForestClassifier` |
| `n_estimators` | 100 (défaut) |
| `random_state` | 42 |
| `n_jobs` | -1 (tous les cœurs) |
| Features | 28 variables (voir `src/config/config.py`) |
| Split | 70% train / 30% test |

---

## CI/CD

Le pipeline GitHub Actions se déclenche sur chaque push vers `main`, `master` ou `develop` :

1. **Lint** — flake8 sur `src/` et `tests/`
2. **Test** — pytest avec coverage ≥ 60%

---

<p align="center">
  Ibrahima Wane — Projet MLOps <a href="https://datascientest.com">DataScientest</a> × Liora
</p>
