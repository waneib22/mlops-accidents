# Accidents Routiers — MLOps Pipeline

![CI](https://github.com/waneib22/mlops-accidents/actions/workflows/python-app.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Projet MLOps de classification de la gravité d'accidents de la route en France.  
Les données proviennent des **Bases de données annuelles des accidents corporels (BAAC) 2021** publiées sur [data.gouv.fr](https://www.data.gouv.fr).

---

## Table des matières

- [Contexte](#contexte)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Pipeline de données](#pipeline-de-données)
- [Entraînement du modèle](#entraînement-du-modèle)
- [API](#api)
- [Docker](#docker)
- [Tests](#tests)
- [Structure du projet](#structure-du-projet)
- [Contributeurs](#contributeurs)

---

## Contexte

Ce projet implémente la **Phase 1 (Fondations)** d'un pipeline MLOps complet.  
L'objectif est de prédire la gravité d'un accident de la route en deux classes :

| Classe | Label | Signification |
|--------|-------|--------------|
| `1` | **prioritaire** | Victime hospitalisée ou décédée |
| `0` | **non-prioritaire** | Victime indemne ou blessée légèrement |

Le modèle est un **Random Forest Classifier** entraîné sur ~54 000 accidents (split 70/30).

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Docker Compose                   │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  accidents_api  (python:3.10-slim)          │  │
│  │                                             │  │
│  │  GET  /            → redirect /docs         │  │
│  │  GET  /health      → statut du modèle       │  │
│  │  GET  /stats       → compteurs JSON         │  │
│  │  GET  /model/info  → hyperparamètres        │  │
│  │  POST /predict     → inférence              │  │
│  │  POST /retrain     → ré-entraînement        │  │
│  │                                             │  │
│  │  :8000                                      │  │
│  └────────────────────────────────────────────┘  │
│       │                        │                  │
│  ./data:/app/data    ./src/models:/app/src/models │
└──────────────────────────────────────────────────┘
```

### Structure interne de l'API

```
src/api/
├── main.py           ← application FastAPI + lifespan
├── schemas.py        ← modèles Pydantic (requêtes / réponses)
├── metrics.py        ← état partagé en mémoire
└── routers/
    ├── monitoring.py ← /health /stats /model/info /retrain
    └── inference.py  ← /predict
```

---

## Prérequis

| Outil | Version minimale |
|-------|-----------------|
| Python | 3.10 |
| pip | 23+ |
| Docker Desktop | 24+ |
| Make | — |

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/waneib22/mlops-accidents.git
cd mlops-accidents

# 2. Créer et activer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
make install
```

---

## Pipeline de données

Les données brutes sont téléchargées depuis le bucket S3 DataScientest.

```bash
# Étape 1 — Télécharger les 4 fichiers CSV bruts
PYTHONPATH=. python3 src/data/import_raw_data.py

# Étape 2 — Préprocesser et créer les jeux train/test
PYTHONPATH=. python3 src/data/make_dataset.py
```

Cela produit dans `data/processed/` :

| Fichier | Contenu |
|---------|---------|
| `X_train.csv` | Features d'entraînement (70%) |
| `X_test.csv` | Features de test (30%) |
| `y_train.csv` | Labels d'entraînement |
| `y_test.csv` | Labels de test |

Transformations appliquées :
- Fusion des 4 tables (usagers, caractéristiques, lieux, véhicules)
- Recodage de la variable cible `grav` en binaire
- Extraction de l'heure depuis `hrmn`
- Calcul de l'âge de la victime
- Remplacement des codes Corse (`2A` → `201`, `2B` → `202`)
- Conversion lat/long (virgule → point)
- Remplacement des valeurs `-1` par `NaN`
- Imputation par le mode sur les colonnes sélectionnées

Pour plus de détails : [docs/data_pipeline.md](docs/data_pipeline.md)

---

## Entraînement du modèle

```bash
make train
```

Le modèle est sauvegardé dans `src/models/trained_model.joblib`.

| Paramètre | Valeur |
|-----------|--------|
| Algorithme | `RandomForestClassifier` |
| `n_estimators` | 100 |
| `random_state` | 42 |
| `n_jobs` | -1 |
| Accuracy (test set) | ~77% |

Pour plus de détails : [docs/model.md](docs/model.md)

---

## API

### Lancer en local

```bash
make api
# → http://localhost:8000/docs
```

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Redirige vers `/docs` |
| `GET` | `/docs` | Documentation interactive Swagger |
| `GET` | `/health` | Statut de l'API et du modèle |
| `GET` | `/stats` | Compteurs de prédictions |
| `GET` | `/model/info` | Hyperparamètres du modèle chargé |
| `POST` | `/predict` | Prédiction de gravité |
| `POST` | `/retrain` | Déclenche un ré-entraînement |

### Exemple de prédiction

```bash
make predict
```

Ou manuellement :

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

Réponse :

```json
{
  "prediction": 1,
  "label": "prioritaire",
  "probability": 0.8423,
  "confidence": "high"
}
```

Pour la référence complète des endpoints : [docs/api.md](docs/api.md)

---

## Docker

### Lancer la stack complète

```bash
make docker-up
```

L'API est disponible sur `http://localhost:8000`.

### Commandes utiles

```bash
make docker-down   # arrêter les containers
docker compose ps  # vérifier l'état des services
docker compose logs api --follow  # suivre les logs de l'API
```

### Variables d'environnement

Copier `.env.example` en `.env` et adapter si nécessaire :

```bash
cp .env.example .env
```

---

## Tests

```bash
make test
```

La suite de tests couvre :

| Fichier | Scope | Nombre de tests |
|---------|-------|----------------|
| `tests/test_data.py` | Transformations données, config | 11 |
| `tests/test_api.py` | Endpoints API (mocks) | 13 |

Coverage minimum requis : **60%** (vérifié par la CI).

---

## Commandes disponibles

```bash
make help       # liste toutes les commandes
make install    # installe les dépendances
make lint       # vérifie le style (flake8)
make test       # lance les tests avec coverage
make api        # démarre l'API en mode développement
make train      # entraîne le modèle
make health     # vérifie l'état de l'API
make predict    # envoie une prédiction exemple
make retrain    # déclenche un ré-entraînement via l'API
make docker-up  # lance la stack Docker
make docker-down# arrête la stack Docker
make clean      # nettoie les artefacts
```

---

## Structure du projet

```
mlops-accidents/
├── .github/workflows/
│   └── python-app.yml     ← CI : lint → test (coverage ≥ 60%)
├── config/                ← (réservé aux phases suivantes)
├── data/
│   ├── raw/               ← données brutes (ignorées par git)
│   └── processed/         ← données préprocessées (ignorées par git)
├── docs/
│   ├── api.md             ← référence complète des endpoints
│   ├── data_pipeline.md   ← détail du pipeline de données
│   └── model.md           ← description du modèle
├── notebooks/             ← exploration initiale des données
├── src/
│   ├── api/
│   │   ├── main.py        ← app FastAPI + lifespan
│   │   ├── schemas.py     ← modèles Pydantic
│   │   ├── metrics.py     ← état partagé
│   │   └── routers/
│   │       ├── monitoring.py
│   │       └── inference.py
│   ├── config/
│   │   └── config.py      ← configuration centralisée
│   ├── data/
│   │   ├── import_raw_data.py
│   │   └── make_dataset.py
│   └── models/
│       └── train_model.py
├── tests/
│   ├── test_api.py
│   └── test_data.py
├── .env.example
├── Dockerfile
├── Makefile
├── docker-compose.yml
└── requirements.txt
```

---

## Licence

Ce projet est sous licence [MIT](LICENSE).

---

<p align="center">
  Projet MLOps — <a href="https://datascientest.com">DataScientest</a> × Liora — 2025/2026
</p>
