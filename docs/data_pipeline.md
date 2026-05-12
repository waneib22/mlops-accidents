# Pipeline de données

Ce document décrit les deux étapes du pipeline de données : l'import des fichiers bruts depuis S3 et le préprocessing pour produire les jeux d'entraînement/test.

---

## Source des données

Les données proviennent des **Bases de données annuelles des accidents corporels (BAAC) 2021**, publiées par le Ministère de l'Intérieur sur [data.gouv.fr](https://www.data.gouv.fr).

Les fichiers sont hébergés sur un bucket S3 DataScientest :

```
https://mlops-project-db.s3.eu-west-1.amazonaws.com/accidents/
```

---

## Étape 1 — Import des données brutes

**Script :** `src/data/import_raw_data.py`  
**Commande :** `PYTHONPATH=. python3 src/data/import_raw_data.py`

### Fichiers téléchargés

| Fichier local | Contenu |
|---------------|---------|
| `data/raw/usagers-2021.csv` | Données sur les usagers impliqués |
| `data/raw/caracteristiques-2021.csv` | Caractéristiques générales de l'accident |
| `data/raw/lieux-2021.csv` | Informations sur le lieu de l'accident |
| `data/raw/vehicules-2021.csv` | Informations sur les véhicules impliqués |

Le dossier `data/raw/` est ignoré par git (`.gitignore`).

---

## Étape 2 — Préprocessing

**Script :** `src/data/make_dataset.py`  
**Commande :** `PYTHONPATH=. python3 src/data/make_dataset.py`

### Fusion des tables

Les 4 fichiers sont joints sur les clés `Num_Acc` (identifiant d'accident) et `num_veh` (identifiant de véhicule) :

```
usagers ──┐
          ├── join(Num_Acc, num_veh) ──► merged_df
lieux ────┤
          │
caract ───┘
     ↕
vehicules ── join(Num_Acc, num_veh)
```

### Transformations appliquées

#### 1. Variable cible — `grav`

La colonne `grav` (gravité de la blessure) est recodée en binaire :

| Valeur BAAC | Signification | Classe |
|-------------|---------------|--------|
| 1 | Indemne | 0 — non-prioritaire |
| 4 | Blessé léger | 0 — non-prioritaire |
| 2 | Blessé hospitalisé | 1 — prioritaire |
| 3 | Tué | 1 — prioritaire |

```python
df["grav"] = df["grav"].replace({1: 0, 4: 0, 2: 1, 3: 1})
```

#### 2. Heure de l'accident — `hour`

La colonne `hrmn` peut être au format `"HH:MM"` (string) ou entier `HHMM` selon la source.  
Le script détecte automatiquement le format :

```python
hrmn_str = df_caract["hrmn"].astype(str).str.strip()
if hrmn_str.str.contains(":").any():
    df_caract["hour"] = hrmn_str.str.split(":").str[0].astype(int)
else:
    df_caract["hour"] = hrmn_str.astype(int) // 100
```

#### 3. Âge de la victime — `victim_age`

```python
df["victim_age"] = df["year_acc"] - df["an_nais"]
```

#### 4. Codes Corse

Les codes département `2A` et `2B` (Corse) sont convertis en entier pour éviter les erreurs de type :

```python
df["dep"] = df["dep"].replace({"2A": "201", "2B": "202"}).astype(float)
```

#### 5. Coordonnées GPS

La virgule décimale française est remplacée par un point :

```python
df["lat"] = df["lat"].astype(str).str.replace(",", ".").astype(float)
df["long"] = df["long"].astype(str).str.replace(",", ".").astype(float)
```

#### 6. Valeurs manquantes

Les valeurs `-1` encodées dans le BAAC comme "non renseigné" sont remplacées par `NaN` :

```python
df.replace(-1, np.nan, inplace=True)
```

Les colonnes sélectionnées sont ensuite imputées par le **mode** (valeur la plus fréquente) :

```python
for col in FEATURES:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])
```

### Sélection des features

28 variables sont retenues pour l'entraînement :

```python
FEATURES = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age",
    "catv", "obsm", "motor", "catr", "circ", "surf", "situ", "vma",
    "jour", "mois", "lum", "dep", "com", "agg_", "int", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]
```

### Split train/test

| Paramètre | Valeur |
|-----------|--------|
| `test_size` | 0.30 (30%) |
| `random_state` | 42 |
| `stratify` | `y` (pour conserver la distribution des classes) |

### Fichiers produits

Tous les fichiers sont sauvegardés dans `data/processed/` (ignoré par git) :

| Fichier | Contenu | Taille approx. |
|---------|---------|----------------|
| `X_train.csv` | Features d'entraînement | ~37 800 lignes × 28 cols |
| `X_test.csv` | Features de test | ~16 200 lignes × 28 cols |
| `y_train.csv` | Labels d'entraînement | ~37 800 lignes |
| `y_test.csv` | Labels de test | ~16 200 lignes |

---

## Déséquilibre des classes

Le jeu de données présente un déséquilibre modéré :

| Classe | Proportion approx. |
|--------|--------------------|
| 0 — non-prioritaire | ~55% |
| 1 — prioritaire | ~45% |

Le déséquilibre est suffisamment faible pour que le RandomForest l'absorbe sans rééchantillonnage (Phase 1). Des techniques SMOTE ou `class_weight="balanced"` pourront être explorées en Phase 2.

---

## Reproduire le pipeline complet

```bash
# Depuis la racine du projet, environnement virtuel activé
PYTHONPATH=. python3 src/data/import_raw_data.py
PYTHONPATH=. python3 src/data/make_dataset.py

# Vérifier les fichiers produits
ls -lh data/processed/
```
