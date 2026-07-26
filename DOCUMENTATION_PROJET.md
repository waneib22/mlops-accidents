# Documentation technique — Accidents Routiers MLOps Pipeline
### Phase 1 : Fondations

---

# Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Architecture globale](#2-architecture-globale)
3. [Structure du projet](#3-structure-du-projet)
4. [Configuration centralisée](#4-configuration-centralisée)
5. [Pipeline de données](#5-pipeline-de-données)
   - 5.1 Import des données brutes
   - 5.2 Préprocessing
6. [Entraînement du modèle](#6-entraînement-du-modèle)
7. [API FastAPI](#7-api-fastapi)
   - 7.1 Point d'entrée — main.py
   - 7.2 Schémas Pydantic — schemas.py
   - 7.3 État partagé — metrics.py
   - 7.4 Router monitoring
   - 7.5 Router inference
8. [Tests automatisés](#8-tests-automatisés)
   - 8.1 Tests données
   - 8.2 Tests API
9. [Docker et déploiement](#9-docker-et-déploiement)
   - 9.1 Dockerfile
   - 9.2 docker-compose.yml
10. [Intégration continue (CI/CD)](#10-intégration-continue-cicd)
11. [Makefile — commandes disponibles](#11-makefile--commandes-disponibles)
12. [Guide de démarrage rapide](#12-guide-de-démarrage-rapide)
13. [Évolutions prévues (Phase 2+)](#13-évolutions-prévues-phase-2)

---

# 1. Contexte et objectifs

## 1.1 Problème métier

En France, les accidents de la route font chaque année des milliers de victimes. Les services de secours doivent prioriser leurs interventions rapidement. Ce projet construit un modèle de machine learning capable de **prédire automatiquement la gravité d'un accident** à partir des données disponibles au moment du signalement.

## 1.2 Données sources

Les données proviennent des **Bases de données annuelles des accidents corporels (BAAC) 2021**, publiées par le Ministère de l'Intérieur français et disponibles sur data.gouv.fr. Elles sont composées de quatre fichiers CSV :

| Fichier | Contenu |
|---------|---------|
| `caracteristiques-2021.csv` | Informations générales sur chaque accident (date, lieu, conditions météo) |
| `lieux-2021.csv` | Caractéristiques du lieu de l'accident (type de route, vitesse maximale…) |
| `usagers-2021.csv` | Informations sur les personnes impliquées (âge, sexe, gravité des blessures) |
| `vehicules-2021.csv` | Informations sur les véhicules impliqués (catégorie, motorisation…) |

## 1.3 Objectif de classification

La variable cible est la **gravité de la blessure** (`grav`), recodée en deux classes binaires :

| Classe | Label | Définition BAAC |
|--------|-------|-----------------|
| `1` | **prioritaire** | Victime hospitalisée (code 2) ou décédée (code 3) |
| `0` | **non-prioritaire** | Victime indemne (code 1) ou légèrement blessée (code 4) |

## 1.4 Périmètre de la Phase 1

La Phase 1 couvre les **fondations du pipeline MLOps** :
- Ingestion et préprocessing des données
- Entraînement d'un modèle baseline (Random Forest)
- API de prédiction exposée via FastAPI
- Conteneurisation Docker
- Tests automatisés
- Pipeline CI/CD avec GitHub Actions

---

# 2. Architecture globale

## 2.1 Vue d'ensemble

```
┌──────────────────────────────────────────────────────────┐
│                     Docker Compose                        │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           accidents_api  (python:3.10-slim)          │  │
│  │                                                      │  │
│  │   GET  /              → redirect vers /docs          │  │
│  │   GET  /health        → statut du modèle             │  │
│  │   GET  /stats         → compteurs de prédictions     │  │
│  │   GET  /model/info    → hyperparamètres du modèle    │  │
│  │   POST /predict       → prédiction de gravité        │  │
│  │   POST /retrain       → ré-entraînement              │  │
│  │                                                      │  │
│  │   Port : 8000                                        │  │
│  └─────────────────────────────────────────────────────┘  │
│         │                           │                      │
│   ./data:/app/data      ./src/models:/app/src/models       │
└──────────────────────────────────────────────────────────┘
```

## 2.2 Architecture interne de l'API

L'API est découpée en fichiers selon le principe **un fichier = une responsabilité** :

```
src/api/
├── main.py           ← Initialisation FastAPI + chargement du modèle (lifespan)
├── schemas.py        ← Contrats de données Pydantic (requêtes / réponses)
├── metrics.py        ← État partagé en mémoire (modèle, compteurs)
└── routers/
    ├── monitoring.py ← Endpoints opérationnels : /health /stats /model/info /retrain
    └── inference.py  ← Endpoint de prédiction : /predict
```

## 2.3 Flux de données

```
Données BAAC 2021 (S3)
        │
        ▼
import_raw_data.py  →  data/raw/*.csv
        │
        ▼
make_dataset.py     →  data/processed/X_train.csv
                                       X_test.csv
                                       y_train.csv
                                       y_test.csv
        │
        ▼
train_model.py      →  src/models/trained_model.joblib
        │
        ▼
API FastAPI (port 8000)
        │
        ▼
POST /predict  →  {"prediction": 1, "label": "prioritaire", "probability": 0.84}
```

---

# 3. Structure du projet

```
mlops-accidents/
│
├── .github/
│   └── workflows/
│       └── python-app.yml      ← Pipeline CI/CD : lint puis tests (coverage ≥ 60%)
│
├── data/
│   ├── raw/                    ← Fichiers CSV bruts téléchargés depuis S3 (ignorés par git)
│   └── processed/              ← Fichiers CSV préprocessés : X_train, X_test, y_train, y_test (ignorés par git)
│
├── docs/
│   ├── api.md                  ← Référence complète des endpoints
│   ├── data_pipeline.md        ← Détail des transformations de données
│   └── model.md                ← Description du modèle et de ses performances
│
├── notebooks/                  ← Exploration initiale des données (Jupyter)
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             ← Application FastAPI + lifespan
│   │   ├── schemas.py          ← Modèles Pydantic
│   │   ├── metrics.py          ← État partagé en mémoire
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── monitoring.py   ← /health /stats /model/info /retrain
│   │       └── inference.py    ← /predict
│   │
│   ├── config/
│   │   └── config.py           ← Configuration centralisée (chemins, features, constantes)
│   │
│   ├── data/
│   │   ├── import_raw_data.py  ← Téléchargement des CSV depuis S3
│   │   └── make_dataset.py     ← Fusion, nettoyage, split train/test
│   │
│   └── models/
│       ├── train_model.py      ← Entraînement et sauvegarde du modèle
│       └── trained_model.joblib← Modèle sérialisé (ignoré par git)
│
├── tests/
│   ├── test_data.py            ← Tests des transformations de données (11 tests)
│   └── test_api.py             ← Tests des endpoints API avec mocks (13 tests)
│
├── .env.example                ← Template de variables d'environnement
├── .gitignore                  ← Exclusions git (data/, modèles, venv, __pycache__…)
├── CONTRIBUTING.md             ← Guide de contribution
├── Dockerfile                  ← Image Docker de l'API
├── Makefile                    ← Commandes raccourcies pour toutes les tâches
├── README.md                   ← Documentation principale du projet
├── docker-compose.yml          ← Orchestration Docker (service API)
├── requirements.txt            ← Dépendances Python
└── setup.py                    ← Configuration du package Python (installation éditable)
```

---

# 4. Configuration centralisée

## 4.1 Fichier : `src/config/config.py`

Toute la configuration du projet est centralisée dans ce fichier unique. Cela évite de disperser des chemins ou des constantes dans plusieurs fichiers, et facilite la maintenance.

```python
"""Configuration centralisée du projet.

Chemins, constantes et liste des features partagés par tous les modules.
Modifier ce fichier suffit pour propager un changement à l'ensemble du pipeline.
"""
from pathlib import Path

# Racine du projet (remontée de 2 niveaux depuis src/config/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dossiers de données
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Dossier et chemin du modèle
MODELS_DIR = PROJECT_ROOT / "src" / "models"
MODEL_PATH = MODELS_DIR / "trained_model.joblib"

# Noms des fichiers CSV bruts attendus dans data/raw/
RAW_FILES = {
    "users": "usagers-2021.csv",
    "caract": "caracteristiques-2021.csv",
    "places": "lieux-2021.csv",
    "vehicles": "vehicules-2021.csv",
}

# URL du bucket S3 DataScientest
S3_BASE_URL = "https://mlops-project-db.s3.eu-west-1.amazonaws.com/accidents/"

# Liste ordonnée des 28 features utilisées par le modèle
FEATURES = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age",
    "catv", "obsm", "motor", "catr", "circ", "surf", "situ", "vma",
    "jour", "mois", "lum", "dep", "com", "agg_", "int", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]

# Paramètres du split train/test
TEST_SIZE = 0.3
RANDOM_STATE = 42
```

## 4.2 Utilisation dans les autres modules

Ce fichier est importé par tous les modules qui ont besoin d'un chemin ou d'une constante :

```python
# Dans train_model.py
from src.config.config import DATA_PROCESSED_DIR, MODEL_PATH

# Dans les routers de l'API
from src.config.config import FEATURES, MODEL_PATH

# Dans les tests
from src.config.config import FEATURES, DATA_PROCESSED_DIR
```

---

# 5. Pipeline de données

## 5.1 Import des données brutes

### Fichier : `src/data/import_raw_data.py`

Ce script télécharge les quatre fichiers CSV depuis le bucket S3 DataScientest et les enregistre dans `data/raw/`.

```python
import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder


def import_raw_data(raw_data_relative_path,
                    filenames,
                    bucket_folder_url):
    """Télécharge les fichiers CSV depuis le bucket S3 vers data/raw/."""
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)

    for filename in filenames:
        input_file = os.path.join(bucket_folder_url, filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            print(f'downloading {input_file} as {os.path.basename(output_file)}')
            response = requests.get(input_file)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.text.encode('utf-8'))
            else:
                print(f'Error accessing {input_file}:', response.status_code)


def main(
    raw_data_relative_path="./data/raw",
    filenames=["caracteristiques-2021.csv", "lieux-2021.csv",
               "usagers-2021.csv", "vehicules-2021.csv"],
    bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/accidents/"
):
    """Télécharge les données brutes depuis AWS S3 dans ./data/raw."""
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
```

**Commande d'exécution :**
```bash
PYTHONPATH=. python3 src/data/import_raw_data.py
```

**Fichiers produits dans `data/raw/` :**
- `usagers-2021.csv`
- `caracteristiques-2021.csv`
- `lieux-2021.csv`
- `vehicules-2021.csv`

---

## 5.2 Préprocessing

### Fichier : `src/data/make_dataset.py`

Ce script est le cœur du pipeline de données. Il fusionne les quatre tables, applique les transformations nécessaires, et produit les jeux train/test.

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """Lance le pipeline de préprocessing sur les données brutes."""
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath = click.prompt(
        'Enter the file path for the input data', type=click.Path(exists=True)
    )
    input_filepath_users = os.path.join(input_filepath, "usagers-2021.csv")
    input_filepath_caract = os.path.join(input_filepath, "caracteristiques-2021.csv")
    input_filepath_places = os.path.join(input_filepath, "lieux-2021.csv")
    input_filepath_veh = os.path.join(input_filepath, "vehicules-2021.csv")
    output_filepath = click.prompt(
        'Enter the file path for the output preprocessed data',
        type=click.Path()
    )
    process_data(
        input_filepath_users, input_filepath_caract,
        input_filepath_places, input_filepath_veh,
        output_filepath
    )


def process_data(input_filepath_users, input_filepath_caract,
                 input_filepath_places, input_filepath_veh,
                 output_folderpath):
    """Préprocesse les données BAAC et produit les fichiers train/test."""

    # ── 1. Chargement des 4 fichiers CSV ──────────────────────────────────
    df_users  = pd.read_csv(input_filepath_users,  sep=";")
    df_caract = pd.read_csv(input_filepath_caract, sep=";", header=0, low_memory=False)
    df_places = pd.read_csv(input_filepath_places, sep=";", encoding='utf-8')
    df_veh    = pd.read_csv(input_filepath_veh,    sep=";")

    # ── 2. Création de colonnes dérivées ──────────────────────────────────
    nb_victim    = pd.crosstab(df_users.Num_Acc, "count").reset_index()
    nb_vehicules = pd.crosstab(df_veh.Num_Acc,   "count").reset_index()

    # Année de l'accident extraite de l'identifiant d'accident
    df_users["year_acc"]   = df_users["Num_Acc"].astype(str).apply(lambda x: x[:4]).astype(int)
    # Âge de la victime
    df_users["victim_age"] = df_users["year_acc"] - df_users["an_nais"]
    # Nettoyage des âges aberrants
    for i in df_users["victim_age"]:
        if (i > 120) | (i < 0):
            df_users["victim_age"].replace(i, np.nan)

    # Extraction de l'heure depuis hrmn — détection du format ("HH:MM" ou entier HHMM)
    hrmn_str = df_caract["hrmn"].astype(str).str.strip()
    if hrmn_str.str.contains(":").any():
        df_caract["hour"] = hrmn_str.str.split(":").str[0].astype(int)
    else:
        df_caract["hour"] = hrmn_str.astype(int) // 100

    df_caract.drop(['hrmn', 'an'], inplace=True, axis=1)
    df_users.drop(['an_nais'],     inplace=True, axis=1)

    # ── 3. Recodage de la variable cible grav ─────────────────────────────
    # Étape 1 : réorganisation des codes (1→1, 2→3, 3→4, 4→2)
    df_users["grav"] = df_users["grav"].replace([1, 2, 3, 4], [1, 3, 4, 2])
    # Étape 2 : binarisation (0=non-prioritaire, 1=prioritaire)
    # Résultat : 1→1, 2→3→1, 3→4→1, 4→2→0

    # ── 4. Renommages et remplacements ────────────────────────────────────
    df_caract.rename({"agg": "agg_"}, inplace=True, axis=1)
    # Codes Corse : 2A → 201, 2B → 202
    df_caract["dep"] = df_caract["dep"].str.replace("2A", "201").str.replace("2B", "202")
    df_caract["com"] = df_caract["com"].str.replace("2A", "201").str.replace("2B", "202")

    # ── 5. Conversions de types ───────────────────────────────────────────
    df_caract[["dep", "com", "hour"]] = df_caract[["dep", "com", "hour"]].astype(int)
    # Coordonnées GPS : remplacement de la virgule décimale française par un point
    df_caract["lat"]  = df_caract["lat"].str.replace(',', '.').astype(float)
    df_caract["long"] = df_caract["long"].str.replace(',', '.').astype(float)

    # ── 6. Regroupement de modalités ──────────────────────────────────────
    # atm : conditions atmosphériques → binaire (0=normal, 1=dégradé)
    dico_atm = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}
    df_caract["atm"] = df_caract["atm"].replace(dico_atm)

    # catv : catégorie de véhicule → 7 groupes
    catv_value     = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
                      30,31,32,33,34,35,36,37,38,39,40,41,42,43,50,60,80,99]
    catv_value_new = [0,1,1,2,1,1,6,2,5,5,5,5,5,4,4,4,4,4,3,3,4,4,1,1,1,1,1,
                      6,6,3,3,3,3,1,1,1,1,1,0,0]
    df_veh["catv"] = df_veh["catv"].replace(catv_value, catv_value_new)

    # ── 7. Fusion des 4 tables ────────────────────────────────────────────
    fusion1 = df_users.merge(df_veh, on=["Num_Acc", "num_veh", "id_vehicule"], how="inner")
    fusion1 = fusion1.sort_values(by="grav", ascending=False)
    # Garder uniquement la victime la plus grave par accident
    fusion1 = fusion1.drop_duplicates(subset=['Num_Acc'], keep="first")
    fusion2 = fusion1.merge(df_places, on="Num_Acc", how="left")
    df      = fusion2.merge(df_caract, on='Num_Acc',  how="left")

    # Ajout des comptages de victimes et véhicules
    df = df.merge(nb_victim,    on="Num_Acc", how="inner")
    df.rename({"count": "nb_victim"},    axis=1, inplace=True)
    df = df.merge(nb_vehicules, on="Num_Acc", how="inner")
    df.rename({"count": "nb_vehicules"}, axis=1, inplace=True)

    # ── 8. Binarisation finale de grav ────────────────────────────────────
    df["grav"] = df["grav"].replace([2, 3, 4], [0, 1, 1])

    # ── 9. Remplacement des valeurs manquantes ────────────────────────────
    col_to_replace0_na = ["trajet", "catv", "motor"]
    col_to_replace1_na = ["trajet", "secu1", "catv", "obsm", "motor",
                          "circ", "surf", "situ", "vma", "atm", "col"]
    df[col_to_replace1_na] = df[col_to_replace1_na].replace(-1, np.nan)
    df[col_to_replace0_na] = df[col_to_replace0_na].replace(0,  np.nan)

    # ── 10. Suppression des colonnes inutiles ─────────────────────────────
    list_to_drop = [
        'senc', 'larrout', 'actp', 'manv', 'choc', 'nbv', 'prof', 'plan',
        'Num_Acc', 'id_vehicule', 'num_veh', 'pr', 'pr1', 'voie', 'trajet',
        'secu2', 'secu3', 'adr', 'v1', 'lartpc', 'occutc', 'v2', 'vosp',
        'locp', 'etatp', 'infra', 'obs'
    ]
    df.drop(list_to_drop, axis=1, inplace=True)

    # ── 11. Suppression des lignes avec NaN critiques ─────────────────────
    col_to_drop_lines = ['catv', 'vma', 'secu1', 'obsm', 'atm']
    df = df.dropna(subset=col_to_drop_lines, axis=0)

    # ── 12. Séparation features / cible ───────────────────────────────────
    target = df['grav']
    feats  = df.drop(['grav'], axis=1)

    # ── 13. Split train / test (70% / 30%) ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=0.3, random_state=42
    )

    # ── 14. Imputation des NaN résiduels par le mode ──────────────────────
    col_to_fill_na = ["surf", "circ", "col", "motor"]
    X_train[col_to_fill_na] = X_train[col_to_fill_na].fillna(
        X_train[col_to_fill_na].mode().iloc[0]
    )
    X_test[col_to_fill_na] = X_test[col_to_fill_na].fillna(
        X_train[col_to_fill_na].mode().iloc[0]  # mode calculé sur train uniquement
    )

    # ── 15. Sauvegarde ────────────────────────────────────────────────────
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

    for file, filename in zip(
        [X_train, X_test, y_train, y_test],
        ['X_train', 'X_test', 'y_train', 'y_test']
    ):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
```

**Commande d'exécution :**
```bash
PYTHONPATH=. python3 src/data/make_dataset.py
```

### Résumé des transformations

| Étape | Transformation | Détail |
|-------|---------------|--------|
| 1 | Chargement | 4 fichiers CSV lus avec séparateur `;` |
| 2 | Colonnes dérivées | `year_acc`, `victim_age`, `hour` calculés |
| 3 | Recodage `grav` | 4 codes BAAC → 2 classes (0/1) |
| 4 | Corse | `2A` → `201`, `2B` → `202` |
| 5 | GPS | Virgule décimale → point (`48,86` → `48.86`) |
| 6 | Regroupement | `atm` en binaire, `catv` en 7 groupes |
| 7 | Fusion | 4 tables jointes sur `Num_Acc` + `num_veh` |
| 8 | NaN | `-1` et `0` → `NaN`, puis imputation par le mode |
| 9 | Split | 70% train / 30% test, stratifié, `random_state=42` |

### Fichiers produits dans `data/processed/`

| Fichier | Description | Taille approx. |
|---------|-------------|----------------|
| `X_train.csv` | Features d'entraînement | ~37 800 lignes × 28 colonnes |
| `X_test.csv` | Features de test | ~16 200 lignes × 28 colonnes |
| `y_train.csv` | Labels d'entraînement | ~37 800 lignes |
| `y_test.csv` | Labels de test | ~16 200 lignes |

---

# 6. Entraînement du modèle

## 6.1 Fichier : `src/models/train_model.py`

```python
"""Script d'entraînement du modèle RandomForestClassifier.

Lit X_train/y_train depuis data/processed/, entraîne le modèle, évalue
l'accuracy sur X_test/y_test, puis sérialise le modèle avec joblib.
"""
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn import ensemble

from src.config.config import DATA_PROCESSED_DIR, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train():
    """Entraîne le RandomForestClassifier et sauvegarde le modèle."""
    logger.info("Chargement des données...")
    X_train = pd.read_csv(DATA_PROCESSED_DIR / "X_train.csv")
    X_test  = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y_train = np.ravel(pd.read_csv(DATA_PROCESSED_DIR / "y_train.csv"))
    y_test  = np.ravel(pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv"))

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    logger.info("Entraînement du RandomForestClassifier...")
    model = ensemble.RandomForestClassifier(n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    logger.info(f"Accuracy sur le test set : {score:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Modèle sauvegardé dans {MODEL_PATH}")

    return model, score


if __name__ == "__main__":
    train()
```

## 6.2 Algorithme : Random Forest Classifier

Un **Random Forest** est un ensemble de N arbres de décision entraînés sur des sous-échantillons aléatoires du jeu d'entraînement (technique dite de **bagging**). La prédiction finale est obtenue par **vote majoritaire** entre tous les arbres.

### Avantages pour ce cas d'usage

- Robuste aux valeurs aberrantes et aux features corrélées
- Fournit des probabilités naturellement via `predict_proba()`
- Pas besoin de normalisation des features
- Interprétable via l'importance des features

## 6.3 Hyperparamètres utilisés

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `n_estimators` | 100 | Bon compromis précision/temps de calcul |
| `random_state` | 42 | Reproductibilité garantie |
| `n_jobs` | -1 | Utilise tous les cœurs CPU disponibles |
| `max_depth` | `None` | Arbres développés jusqu'aux feuilles pures |

## 6.4 Performance

| Métrique | Valeur |
|----------|--------|
| Accuracy (test set, 30%) | ~77% |

## 6.5 Sérialisation

Le modèle est sauvegardé avec **joblib** au format binaire compressé :

```
src/models/trained_model.joblib
```

Ce fichier est ignoré par git. Il doit être regénéré à chaque déploiement ou monté via un volume Docker.

**Important — compatibilité sklearn :** le fichier `.joblib` doit être chargé avec la même version de scikit-learn que celle utilisée lors de l'entraînement. Pour garantir la cohérence, il est recommandé d'entraîner le modèle directement dans le conteneur Docker :

```bash
docker exec accidents_api python3 src/models/train_model.py
```

## 6.6 Commande d'entraînement

```bash
# En local
make train

# Ou directement
PYTHONPATH=. python3 src/models/train_model.py
```

Logs attendus lors de l'entraînement :
```
2025-05-10 14:32:01 - INFO - Chargement des données...
2025-05-10 14:32:02 - INFO - Train: (37800, 28), Test: (16200, 28)
2025-05-10 14:32:02 - INFO - Entraînement du RandomForestClassifier...
2025-05-10 14:32:18 - INFO - Accuracy sur le test set : 0.7742
2025-05-10 14:32:18 - INFO - Modèle sauvegardé dans .../trained_model.joblib
```

---

# 7. API FastAPI

L'API est construite avec **FastAPI**, un framework Python moderne et rapide pour la construction d'APIs REST. Elle expose 6 endpoints documentés automatiquement via une interface Swagger accessible à `http://localhost:8000/docs`.

## 7.1 Point d'entrée — `src/api/main.py`

Ce fichier initialise l'application FastAPI, charge le modèle au démarrage, et enregistre les deux routers.

```python
"""FastAPI application entry point.

Initialise l'application, charge le modèle au démarrage via le contexte
``lifespan``, et enregistre les deux routers (monitoring, inference).
"""
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
    {"name": "inference",  "description": "Prédiction de la gravité d'un accident de la route."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application : chargement du modèle au démarrage."""
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
    yield  # L'application tourne ici
    state.ml_model.clear()
    logger.info("shutdown complete")


app = FastAPI(
    title="Accidents Routiers — API de prédiction",
    description=(
        "Classifie la gravité d'un accident de la route en deux catégories :\n\n"
        "- **1 — prioritaire** : victime hospitalisée ou décédée\n"
        "- **0 — non-prioritaire** : victime indemne ou blessée légèrement\n\n"
        "Modèle : **Random Forest Classifier** entraîné sur les données BAAC 2021."
    ),
    version="1.0.0",
    contact={"url": "https://github.com/waneib22/mlops-accidents/issues"},
    license_info={"name": "MIT"},
    openapi_tags=_TAGS,
    lifespan=lifespan,
)

app.include_router(monitoring.router)
app.include_router(inference.router)


@app.get("/", include_in_schema=False)
def root():
    """Redirige la racine vers la documentation Swagger."""
    return RedirectResponse(url="/docs")
```

### Point clé : le `lifespan`

Le **lifespan** est un gestionnaire de contexte asynchrone qui s'exécute :
- **Au démarrage** (`avant yield`) : charge le modèle joblib en mémoire
- **À l'arrêt** (`après yield`) : libère les ressources

Si le fichier `.joblib` est absent (ex. : premier démarrage avant entraînement), l'API démarre quand même en **mode dégradé** (`model_loaded=false`). Les endpoints `/predict` et `/model/info` retourneront alors une erreur 503.

---

## 7.2 Schémas Pydantic — `src/api/schemas.py`

Pydantic valide automatiquement les données entrantes et sortantes. Chaque classe définit un **contrat d'interface** pour un endpoint.

```python
"""Pydantic schemas for request/response validation."""
from typing import Optional
from pydantic import BaseModel


class AccidentFeatures(BaseModel):
    """Les 28 variables décrivant un accident, envoyées dans le corps de POST /predict."""
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
    """Réponse de l'endpoint POST /predict."""
    prediction: int    # 0 ou 1
    label: str         # "prioritaire" ou "non-prioritaire"
    probability: float # probabilité de la classe prédite
    confidence: str    # "high", "medium" ou "low"


class HealthResponse(BaseModel):
    """Réponse de l'endpoint GET /health."""
    status: str          # "ok" ou "degraded"
    model_loaded: bool
    model_type: Optional[str]
    n_features: int
    uptime_seconds: float
    api_version: str


class ModelInfoResponse(BaseModel):
    """Réponse de l'endpoint GET /model/info."""
    type: str
    n_features: int
    features: list
    params: dict


class StatsResponse(BaseModel):
    """Réponse de l'endpoint GET /stats."""
    total_predictions: int
    predictions_by_label: dict
    uptime_seconds: float
```

---

## 7.3 État partagé — `src/api/metrics.py`

Ce module contient les variables d'état partagées entre tous les routers de l'API. En Python, un module importé est un **singleton** : tous les imports partagent le même objet en mémoire.

```python
"""État partagé en mémoire entre les routers de l'API."""
import threading

# Dictionnaire contenant le modèle chargé : {"classifier": <modèle sklearn>}
ml_model: dict = {}

# Timestamp de démarrage de l'API (utilisé pour calculer uptime)
startup_time: float = 0.0

# Compteurs de prédictions
stats: dict = {"total": 0, "prioritaire": 0, "non_prioritaire": 0}

# Verrou pour les mises à jour thread-safe des compteurs
stats_lock = threading.Lock()
```

**Pourquoi un verrou (`threading.Lock`) ?**
FastAPI peut traiter plusieurs requêtes simultanément dans des threads différents. Sans verrou, deux requêtes arrivant en même temps pourraient lire et écrire les compteurs en même temps, produisant des valeurs incorrectes (race condition). Le verrou garantit qu'une seule requête à la fois peut modifier les compteurs.

---

## 7.4 Router monitoring — `src/api/routers/monitoring.py`

```python
"""Router de monitoring — endpoints opérationnels.

Expose : /health, /stats, /model/info, /retrain.
"""
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
    """Exécute train_model.py en sous-processus et recharge le modèle."""
    logger.info("retrain=started")
    try:
        result = subprocess.run(
            ["python3", "src/models/train_model.py"],
            env={**os.environ, "PYTHONPATH": "."},
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
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
    """Retourne le statut opérationnel de l'API et du modèle."""
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
    """Retourne les compteurs de prédictions depuis le démarrage."""
    with stats_lock:
        return StatsResponse(
            total_predictions=stats["total"],
            predictions_by_label={
                "prioritaire":    stats["prioritaire"],
                "non_prioritaire": stats["non_prioritaire"],
            },
            uptime_seconds=round(time.time() - startup_time, 2),
        )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Retourne les métadonnées et hyperparamètres du modèle chargé."""
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
    """Déclenche un ré-entraînement du modèle en arrière-plan (non bloquant)."""
    background_tasks.add_task(_run_training)
    logger.info("retrain=requested")
    return {"status": "accepted", "message": "Retraining started in background."}
```

### Description des endpoints

#### `GET /health`
Retourne le statut de l'API. Toujours HTTP 200, mais le champ `status` vaut `"degraded"` si le modèle n'est pas chargé.

Exemple de réponse :
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "n_features": 28,
  "uptime_seconds": 142.35,
  "api_version": "1.0.0"
}
```

#### `GET /stats`
Compteurs de prédictions accumulés depuis le démarrage (réinitialisés au redémarrage).

```json
{
  "total_predictions": 42,
  "predictions_by_label": {
    "prioritaire": 31,
    "non_prioritaire": 11
  },
  "uptime_seconds": 310.88
}
```

#### `GET /model/info`
Retourne 503 si le modèle n'est pas chargé, sinon :

```json
{
  "type": "RandomForestClassifier",
  "n_features": 28,
  "features": ["place", "catu", "sexe", ...],
  "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
}
```

#### `POST /retrain`
Lance l'entraînement en tâche de fond et retourne immédiatement (HTTP 202 Accepted) :

```json
{"status": "accepted", "message": "Retraining started in background."}
```

---

## 7.5 Router inference — `src/api/routers/inference.py`

```python
"""Router d'inférence — endpoint de prédiction.

Expose : POST /predict.
"""
import logging
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.metrics import ml_model, stats, stats_lock
from src.api.schemas import AccidentFeatures, PredictionResponse
from src.config.config import FEATURES

logger = logging.getLogger("accidents_api")
router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
def predict(features: AccidentFeatures):
    """Prédit la gravité d'un accident à partir des 28 features."""
    if ml_model.get("classifier") is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Lancez d'abord l'entraînement.",
        )

    t0 = time.time()

    # Conversion en DataFrame dans l'ordre exact des features attendues par le modèle
    input_data = pd.DataFrame([features.model_dump()])[FEATURES]
    prediction  = int(ml_model["classifier"].predict(input_data)[0])
    proba       = ml_model["classifier"].predict_proba(input_data)[0]
    probability = round(float(np.max(proba)), 4)
    duration    = time.time() - t0

    # Niveau de confiance basé sur la probabilité
    if probability >= 0.80:
        confidence = "high"
    elif probability >= 0.60:
        confidence = "medium"
    else:
        confidence = "low"

    label = "prioritaire" if prediction == 1 else "non-prioritaire"

    # Mise à jour thread-safe des compteurs
    with stats_lock:
        stats["total"] += 1
        if prediction == 1:
            stats["prioritaire"] += 1
        else:
            stats["non_prioritaire"] += 1

    # Log structuré pour chaque prédiction
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
```

### Description de l'endpoint

#### `POST /predict`

**Corps de la requête** — les 28 features de l'accident :

```json
{
  "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
  "year_acc": 2021, "victim_age": 60, "catv": 2, "obsm": 1,
  "motor": 1, "catr": 3, "circ": 2, "surf": 1, "situ": 1,
  "vma": 50, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
  "com": 77317, "agg_": 2, "int": 1, "atm": 0, "col": 6,
  "lat": 48.60, "long": 2.89, "hour": 17,
  "nb_victim": 2, "nb_vehicules": 1
}
```

**Réponse 200 :**

```json
{
  "prediction": 1,
  "label": "prioritaire",
  "probability": 0.8423,
  "confidence": "high"
}
```

**Règles de confiance :**

| `probability` | `confidence` |
|---------------|-------------|
| ≥ 0.80 | `high` |
| ≥ 0.60 | `medium` |
| < 0.60 | `low` |

**Codes d'erreur :**
- `422` : champ manquant ou invalide (ex. : oubli du champ `vma`)
- `503` : modèle non chargé

---

# 8. Tests automatisés

## 8.1 Tests des données — `tests/test_data.py`

Ces tests vérifient que chaque transformation appliquée dans `make_dataset.py` produit le résultat attendu, sans avoir besoin des fichiers CSV réels.

```python
import numpy as np
import pandas as pd
import pytest
from src.config.config import FEATURES, DATA_PROCESSED_DIR


# ── Données de test ───────────────────────────────────────────────────────────

def make_sample_users():
    return pd.DataFrame({
        "Num_Acc": [202100001, 202100002],
        "num_veh": [1, 1],
        "id_vehicule": [1, 2],
        "grav": [1, 3],
        "an_nais": [1985, 1970],
        "sexe": [1, 2],
        "secu1": [1.0, 2.0],
        "secu2": [0.0, 0.0],
        "secu3": [0.0, 0.0],
        "trajet": [1, 2],
        "place": [1, 2],
        "catu": [1, 1],
        "locp": [0, 0],
        "actp": [0, 0],
        "etatp": [0, 0],
    })


def make_sample_caract():
    return pd.DataFrame({
        "Num_Acc": [202100001, 202100002],
        "jour": [15, 20],
        "mois": [6, 9],
        "an": [2021, 2021],
        "hrmn": ["1430", "0900"],
        "lum": [1, 3],
        "dep": ["75", "69"],
        "com": ["75001", "69001"],
        "agg": [2, 1],
        "int": [1, 2],
        "atm": [1, 2],
        "col": [2, 3],
        "adr": ["", ""],
        "lat": ["48,8566", "45,7640"],
        "long": ["2,3522", "4,8357"],
        "catr": [3, 4],
    })


# ── Tests de configuration ────────────────────────────────────────────────────

class TestConfig:
    def test_features_list_not_empty(self):
        assert len(FEATURES) == 28

    def test_features_contains_expected_keys(self):
        expected = ["place", "catu", "sexe", "vma", "lat", "long", "nb_victim", "nb_vehicules"]
        for key in expected:
            assert key in FEATURES

    def test_data_processed_dir_is_path(self):
        assert DATA_PROCESSED_DIR.name == "processed"


# ── Tests des transformations ─────────────────────────────────────────────────

class TestDataTransformations:
    def test_grav_binary_mapping(self):
        # Codes BAAC : 1=indemne, 2=blessé léger, 3=hospitalisé, 4=décédé
        # Étape 1 : réorganisation → [1,3,4,2]
        # Étape 2 : binarisation [2,3,4]→[0,1,1]
        # Résultat attendu : 1→1, 2→1, 3→1, 4→0
        df = pd.DataFrame({"grav": [1, 2, 3, 4]})
        df["grav"] = df["grav"].replace([1, 2, 3, 4], [1, 3, 4, 2])
        df["grav"] = df["grav"].replace([2, 3, 4], [0, 1, 1])
        assert list(df["grav"]) == [1, 1, 1, 0]

    def test_victim_age_calculation(self):
        df = pd.DataFrame({
            "Num_Acc": [202100001, 202100002],
            "an_nais": [1990, 1980],
        })
        df["year_acc"]   = df["Num_Acc"].astype(str).apply(lambda x: int(x[:4]))
        df["victim_age"] = df["year_acc"] - df["an_nais"]
        assert list(df["victim_age"]) == [31, 41]

    def test_hour_extraction_from_hrmn(self):
        # hrmn en format entier HHMM : 1430 = 14h30
        df = pd.DataFrame({"hrmn": [1430, 900, 2359, 0]})
        df["hour"] = df["hrmn"].astype(int) // 100
        assert list(df["hour"]) == [14, 9, 23, 0]

    def test_atm_grouping(self):
        # 1=normal, 8=normal, 9=normal → 0 ; reste → 1
        dico = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}
        df = pd.DataFrame({"atm": [1, 2, 8, 5]})
        df["atm"] = df["atm"].replace(dico)
        assert list(df["atm"]) == [0, 1, 0, 1]

    def test_lat_long_conversion(self):
        df = pd.DataFrame({"lat": ["48,8566", "45,7640"], "long": ["2,3522", "4,8357"]})
        df["lat"]  = df["lat"].str.replace(",", ".").astype(float)
        df["long"] = df["long"].str.replace(",", ".").astype(float)
        assert abs(df["lat"][0]  - 48.8566) < 1e-4
        assert abs(df["long"][0] - 2.3522)  < 1e-4

    def test_corse_department_replacement(self):
        df = pd.DataFrame({"dep": ["2A", "2B", "75"]})
        df["dep"] = df["dep"].str.replace("2A", "201").str.replace("2B", "202")
        assert list(df["dep"]) == ["201", "202", "75"]

    def test_nan_replacement_for_minus_one(self):
        df = pd.DataFrame({"secu1": [1.0, -1.0, 2.0], "motor": [0, 1, -1]})
        df[["secu1", "motor"]] = df[["secu1", "motor"]].replace(-1, np.nan)
        assert pd.isna(df["secu1"][1])
        assert pd.isna(df["motor"][2])

    def test_train_test_split_ratio(self):
        from sklearn.model_selection import train_test_split
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        assert len(X_train) == 70
        assert len(X_test)  == 30
```

---

## 8.2 Tests de l'API — `tests/test_api.py`

Ces tests vérifient chaque endpoint en **isolant le modèle** avec un mock : on simule un modèle sklearn sans avoir besoin du fichier `.joblib`.

```python
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


SAMPLE_FEATURES = {
    "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
    "year_acc": 2021, "victim_age": 60, "catv": 2, "obsm": 1,
    "motor": 1, "catr": 3, "circ": 2, "surf": 1, "situ": 1,
    "vma": 50, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
    "com": 77317, "agg_": 2, "int": 1, "atm": 0, "col": 6,
    "lat": 48.60, "long": 2.89, "hour": 17,
    "nb_victim": 2, "nb_vehicules": 1,
}


@pytest.fixture
def client_with_model():
    """Client de test avec un modèle mocké retournant prediction=1, proba=0.8."""
    mock_model = MagicMock()
    mock_model.predict.return_value       = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    mock_model.get_params.return_value    = {"n_estimators": 100, "random_state": 42}

    import src.api.metrics as state
    from src.api.main import app
    state.ml_model["classifier"] = mock_model
    with TestClient(app) as c:
        state.ml_model["classifier"] = mock_model
        yield c


@pytest.fixture
def client_without_model():
    """Client de test sans modèle chargé (mode dégradé)."""
    import src.api.metrics as state
    from src.api.main import app
    state.ml_model["classifier"] = None
    with TestClient(app) as c:
        state.ml_model["classifier"] = None
        yield c


class TestRootEndpoint:
    def test_root_redirects_to_docs(self, client_with_model):
        response = client_with_model.get("/", follow_redirects=False)
        assert response.status_code in [307, 308]
        assert "/docs" in response.headers["location"]


class TestHealthEndpoint:
    def test_health_ok_when_model_loaded(self, client_with_model):
        response = client_with_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"]       == "ok"
        assert data["model_loaded"] is True
        assert data["model_type"]   is not None
        assert data["n_features"]   == 28
        assert data["uptime_seconds"] >= 0
        assert data["api_version"]  == "1.0.0"

    def test_health_degraded_when_no_model(self, client_without_model):
        response = client_without_model.get("/health")
        data = response.json()
        assert data["status"]       == "degraded"
        assert data["model_loaded"] is False
        assert data["model_type"]   is None


class TestStatsEndpoint:
    def test_stats_returns_expected_schema(self, client_with_model):
        response = client_with_model.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions"    in data
        assert "predictions_by_label" in data
        assert "uptime_seconds"       in data
        assert "prioritaire"     in data["predictions_by_label"]
        assert "non_prioritaire" in data["predictions_by_label"]

    def test_stats_increments_on_predict(self, client_with_model):
        import src.api.metrics as state
        before = state.stats["total"]
        client_with_model.post("/predict", json=SAMPLE_FEATURES)
        assert state.stats["total"] == before + 1


class TestModelInfoEndpoint:
    def test_model_info_returns_metadata(self, client_with_model):
        response = client_with_model.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["n_features"] == 28
        assert len(data["features"]) == 28
        assert "type"   in data
        assert "params" in data

    def test_model_info_returns_503_without_model(self, client_without_model):
        response = client_without_model.get("/model/info")
        assert response.status_code == 503


class TestRetrainEndpoint:
    def test_retrain_returns_202(self, client_with_model):
        with patch("src.api.routers.monitoring._run_training"):
            response = client_with_model.post("/retrain")
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"


class TestPredictEndpoint:
    def test_predict_returns_valid_response(self, client_with_model):
        response = client_with_model.post("/predict", json=SAMPLE_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]
        assert data["label"]      in ["prioritaire", "non-prioritaire"]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["confidence"] in ["high", "medium", "low"]

    def test_predict_confidence_matches_probability(self, client_with_model):
        response = client_with_model.post("/predict", json=SAMPLE_FEATURES)
        data = response.json()
        p = data["probability"]
        if p >= 0.80:
            assert data["confidence"] == "high"
        elif p >= 0.60:
            assert data["confidence"] == "medium"
        else:
            assert data["confidence"] == "low"

    def test_predict_label_matches_prediction(self, client_with_model):
        response = client_with_model.post("/predict", json=SAMPLE_FEATURES)
        data = response.json()
        expected = "prioritaire" if data["prediction"] == 1 else "non-prioritaire"
        assert data["label"] == expected

    def test_predict_without_model_returns_503(self, client_without_model):
        response = client_without_model.post("/predict", json=SAMPLE_FEATURES)
        assert response.status_code == 503

    def test_predict_missing_field_returns_422(self, client_with_model):
        incomplete = {k: v for k, v in SAMPLE_FEATURES.items() if k != "vma"}
        response = client_with_model.post("/predict", json=incomplete)
        assert response.status_code == 422
```

## 8.3 Lancer les tests

```bash
make test
```

Équivalent à :
```bash
PYTHONPATH=. coverage run -m pytest tests/ -v
coverage report --min-coverage=60
coverage html
```

Le rapport HTML de couverture est généré dans `htmlcov/index.html`.

## 8.4 Résumé de la suite de tests

| Fichier | Classe | Tests |
|---------|--------|-------|
| `test_data.py` | `TestConfig` | 3 tests sur la configuration |
| `test_data.py` | `TestDataTransformations` | 8 tests sur les transformations |
| `test_api.py` | `TestRootEndpoint` | 1 test de redirection |
| `test_api.py` | `TestHealthEndpoint` | 2 tests (ok + dégradé) |
| `test_api.py` | `TestStatsEndpoint` | 2 tests (schéma + incrément) |
| `test_api.py` | `TestModelInfoEndpoint` | 2 tests (ok + 503) |
| `test_api.py` | `TestRetrainEndpoint` | 1 test (202 accepted) |
| `test_api.py` | `TestPredictEndpoint` | 5 tests (réponse, confiance, label, 503, 422) |
| **Total** | | **24 tests** |

---

# 9. Docker et déploiement

## 9.1 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copie des fichiers de configuration des dépendances EN PREMIER
# (optimisation du cache Docker : cette couche n'est reconstruite
# que si requirements.txt ou setup.py changent)
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Pourquoi `COPY requirements.txt setup.py ./` en premier ?

Le `requirements.txt` contient `-e .` (installation en mode éditable), qui nécessite le fichier `setup.py`. En copiant ces deux fichiers avant le `COPY . .`, on s'assure que Docker peut mettre en cache la couche `pip install` et ne la réexécute pas à chaque changement de code source — seulement quand les dépendances changent réellement.

## 9.2 docker-compose.yml

```yaml
services:

  api:
    build: .
    container_name: accidents_api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data          # Données persistées sur l'hôte
      - ./src/models:/app/src/models  # Modèle persisté sur l'hôte
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

### Volumes

| Volume | Rôle |
|--------|------|
| `./data:/app/data` | Les données CSV (brutes et traitées) sont lues depuis l'hôte |
| `./src/models:/app/src/models` | Le modèle `.joblib` est partagé entre l'hôte et le conteneur |

Grâce aux volumes, le modèle entraîné en local est automatiquement disponible dans le conteneur, et inversement.

### Healthcheck

Docker vérifie toutes les 30 secondes que l'endpoint `/health` répond. Si 3 vérifications consécutives échouent, le conteneur est marqué `unhealthy`.

## 9.3 Commandes Docker

```bash
# Construire et démarrer
make docker-up
# Équivalent : docker compose up --build

# Arrêter
make docker-down
# Équivalent : docker compose down

# Voir l'état des services
docker compose ps

# Suivre les logs en temps réel
docker compose logs api --follow

# Entraîner le modèle DANS le conteneur (recommandé)
docker exec accidents_api python3 src/models/train_model.py
```

---

# 10. Intégration continue (CI/CD)

## 10.1 Fichier : `.github/workflows/python-app.yml`

```yaml
name: CI - Accidents MLOps

on:
  push:
    branches: ["master", "main", "develop"]
  pull_request:
    branches: ["master", "main"]

permissions:
  contents: read

jobs:
  lint:
    name: Lint (flake8)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install flake8
        run: pip install flake8
      - name: Lint - erreurs bloquantes
        run: flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
      - name: Lint - style (non bloquant)
        run: flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

  test:
    name: Tests (pytest)
    runs-on: ubuntu-latest
    needs: lint          # Le job test ne démarre qu'après lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: "pip"   # Cache des dépendances pip entre les runs
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests with coverage
        run: |
          coverage run -m pytest tests/ -v
          coverage report --min-coverage=60
        env:
          PYTHONPATH: ${{ github.workspace }}
```

## 10.2 Fonctionnement du pipeline

```
Push / Pull Request
        │
        ▼
┌─────────────────┐
│   Job : lint     │   flake8 vérifie les erreurs de syntaxe bloquantes
│   (ubuntu-latest)│   (imports non définis, erreurs de syntaxe Python, etc.)
└────────┬────────┘
         │ Si lint OK
         ▼
┌─────────────────┐
│  Job : test      │   pytest + coverage
│  (ubuntu-latest) │   La couverture de code doit être ≥ 60%
└─────────────────┘   Sinon la CI échoue et bloque le merge
```

## 10.3 Règles flake8

| Code | Sévérité | Description |
|------|----------|-------------|
| `E9` | Bloquant | Erreurs de syntaxe Python |
| `F63` | Bloquant | Mauvais usage de `assert` |
| `F7` | Bloquant | Erreurs de syntaxe dans les annotations |
| `F82` | Bloquant | Variables non définies |
| Style | Non bloquant | Lignes > 127 caractères, complexité > 10 |

---

# 11. Makefile — commandes disponibles

```makefile
.PHONY: help install lint test api train docker-up docker-down clean

PYTHON   := python3
UVICORN  := uvicorn
SRC_DIRS := src tests

help:
	@echo "  Accidents Routiers — MLOps Pipeline"
	@echo "  make install     Install all dependencies"
	@echo "  make lint        Run flake8 linter"
	@echo "  make test        Run test suite with coverage"
	@echo "  make api         Start API in development mode"
	@echo "  make train       Train the model"
	@echo "  make health      Check API health"
	@echo "  make predict     Send an example prediction"
	@echo "  make retrain     Trigger retraining via API"
	@echo "  make docker-up   Build & start full stack"
	@echo "  make docker-down Stop Docker stack"
	@echo "  make clean       Remove build artifacts & cache"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	flake8 $(SRC_DIRS) --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 $(SRC_DIRS) --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test:
	PYTHONPATH=. coverage run -m pytest tests/ -v
	coverage report --min-coverage=60
	coverage html

api:
	PYTHONPATH=. $(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

train:
	PYTHONPATH=. $(PYTHON) src/models/train_model.py

health:
	@curl -s http://localhost:8000/health | python3 -m json.tool

predict:
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"place":10,"catu":3,...}' \
		| python3 -m json.tool

retrain:
	@curl -s -X POST http://localhost:8000/retrain | python3 -m json.tool

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
```

### Tableau récapitulatif

| Commande | Description |
|----------|-------------|
| `make help` | Affiche toutes les commandes disponibles |
| `make install` | Installe les dépendances Python (`pip install -r requirements.txt`) |
| `make lint` | Vérifie le style du code avec flake8 |
| `make test` | Lance les tests pytest avec rapport de couverture |
| `make api` | Démarre l'API en mode développement avec rechargement automatique |
| `make train` | Entraîne le modèle en local |
| `make health` | Appelle `GET /health` et affiche le résultat formaté |
| `make predict` | Envoie une requête de prédiction exemple à l'API |
| `make retrain` | Déclenche un ré-entraînement via `POST /retrain` |
| `make docker-up` | Construit l'image Docker et démarre le service |
| `make docker-down` | Arrête et supprime les conteneurs |
| `make clean` | Supprime les fichiers `.pyc`, `__pycache__`, `.coverage`, etc. |

---

# 12. Guide de démarrage rapide

## Méthode 1 — En local (développement)

```bash
# 1. Cloner le dépôt
git clone https://github.com/waneib22/mlops-accidents.git
cd mlops-accidents

# 2. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
make install

# 4. Télécharger les données brutes depuis S3
PYTHONPATH=. python3 src/data/import_raw_data.py

# 5. Lancer le préprocessing
PYTHONPATH=. python3 src/data/make_dataset.py

# 6. Entraîner le modèle
make train

# 7. Démarrer l'API
make api
# → http://localhost:8000/docs
```

## Méthode 2 — Via Docker (recommandé pour la production)

```bash
# 1. Cloner le dépôt
git clone https://github.com/waneib22/mlops-accidents.git
cd mlops-accidents

# 2. (Optionnel) Copier le fichier d'environnement
cp .env.example .env

# 3. Démarrer la stack Docker
make docker-up
# → http://localhost:8000/docs

# 4. Télécharger les données dans le conteneur
docker exec accidents_api python3 src/data/import_raw_data.py

# 5. Préprocesser les données dans le conteneur
docker exec accidents_api bash -c "PYTHONPATH=. python3 src/data/make_dataset.py"

# 6. Entraîner le modèle dans le conteneur
docker exec accidents_api python3 src/models/train_model.py

# 7. Vérifier que le modèle est bien chargé
make health
```

## Tester l'API

```bash
# Vérifier la santé
curl http://localhost:8000/health | python3 -m json.tool

# Prédiction
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
  }' | python3 -m json.tool

# Statistiques
curl http://localhost:8000/stats | python3 -m json.tool

# Déclencher un ré-entraînement
curl -X POST http://localhost:8000/retrain | python3 -m json.tool
```

---

# 13. Évolutions prévues (Phase 2+)

## Phase 2 — Expérimentation et tracking

- **MLflow** : tracking des expériences (paramètres, métriques, artefacts)
- Comparaison d'algorithmes : XGBoost, LightGBM, Logistic Regression
- Optimisation des hyperparamètres : GridSearchCV / Optuna
- Métriques supplémentaires : F1-score, ROC-AUC, PR-AUC, matrice de confusion
- Gestion du déséquilibre de classes : `class_weight="balanced"` ou SMOTE

## Phase 3 — Production et monitoring

- **Prometheus** : métriques de l'API (latence, nombre de requêtes, taux d'erreur)
- **Grafana** : tableau de bord de monitoring en temps réel
- Détection du **data drift** et du **model drift**
- Alertes automatiques en cas de dégradation des performances

## Phase 4 — Orchestration avancée

- **Airflow** ou **Prefect** : orchestration du pipeline de données
- **Kubernetes** : déploiement et mise à l'échelle automatique
- **Feature Store** : partage de features entre équipes
- Registry de modèles (MLflow Model Registry ou Seldon)

---

*Document généré dans le cadre du projet MLOps — DataScientest × Liora — 2025/2026*
