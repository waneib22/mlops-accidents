# Description du modèle

Ce document détaille le modèle de classification entraîné dans le cadre de la Phase 1.

---

## Objectif

Prédire la **gravité d'un accident de la route** en France en deux classes :

| Classe | Label | Définition |
|--------|-------|------------|
| `1` | prioritaire | Victime hospitalisée ou décédée |
| `0` | non-prioritaire | Victime indemne ou légèrement blessée |

---

## Algorithme

**RandomForestClassifier** (scikit-learn)

Un Random Forest est un ensemble de **N arbres de décision** entraînés sur des sous-échantillons aléatoires du jeu d'entraînement (bagging). La prédiction finale est la **majorité des votes** des arbres individuels.

### Avantages pour ce cas d'usage

- Robuste aux valeurs aberrantes et aux features corrélées.
- Fournit naturellement des probabilités via `predict_proba`.
- Peu sensible à la mise à l'échelle des features.
- Interprétable via l'importance des features.

---

## Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `n_estimators` | 100 | Bon compromis vitesse/variance |
| `random_state` | 42 | Reproductibilité |
| `n_jobs` | -1 | Parallélisation sur tous les cœurs CPU |
| `max_depth` | `None` | Arbres développés jusqu'aux feuilles pures |
| `min_samples_split` | 2 | Valeur par défaut sklearn |
| `min_samples_leaf` | 1 | Valeur par défaut sklearn |

---

## Données d'entraînement

| Jeu | Lignes | Features |
|-----|--------|----------|
| Train (70%) | ~37 800 | 28 |
| Test (30%) | ~16 200 | 28 |

Source : BAAC 2021 (France métropolitaine), après préprocessing complet.  
Voir [docs/data_pipeline.md](data_pipeline.md) pour le détail des transformations.

---

## Performance

| Métrique | Valeur (test set) |
|----------|------------------|
| Accuracy | ~77% |

> En Phase 1, seule l'accuracy est suivie. Des métriques supplémentaires (F1, ROC-AUC, matrice de confusion) seront intégrées en Phase 2 (MLflow).

---

## Features utilisées (28)

| Feature | Source | Type |
|---------|--------|------|
| `place` | usagers | catégorielle |
| `catu` | usagers | catégorielle |
| `sexe` | usagers | catégorielle |
| `secu1` | usagers | catégorielle |
| `year_acc` | caractéristiques | numérique |
| `victim_age` | calculé | numérique |
| `catv` | véhicules | catégorielle |
| `obsm` | véhicules | catégorielle |
| `motor` | véhicules | catégorielle |
| `catr` | lieux | catégorielle |
| `circ` | lieux | catégorielle |
| `surf` | lieux | catégorielle |
| `situ` | lieux | catégorielle |
| `vma` | lieux | numérique |
| `jour` | caractéristiques | catégorielle |
| `mois` | caractéristiques | catégorielle |
| `lum` | caractéristiques | catégorielle |
| `dep` | caractéristiques | catégorielle |
| `com` | caractéristiques | catégorielle |
| `agg_` | caractéristiques | catégorielle |
| `int` | caractéristiques | catégorielle |
| `atm` | caractéristiques | catégorielle |
| `col` | caractéristiques | catégorielle |
| `lat` | caractéristiques | numérique |
| `long` | caractéristiques | numérique |
| `hour` | calculé | numérique |
| `nb_victim` | calculé | numérique |
| `nb_vehicules` | calculé | numérique |

---

## Sérialisation

Le modèle est sauvegardé via **joblib** :

```
src/models/trained_model.joblib
```

Le fichier est ignoré par git (`.gitignore`) — chaque déploiement nécessite un entraînement ou le montage du volume Docker.

### Compatibilité sklearn

Le modèle doit être chargé avec la **même version de scikit-learn** que celle utilisée lors de l'entraînement. Pour éviter tout problème :

- En local : `make train` dans l'environnement virtuel.
- En Docker : `docker exec accidents_api python3 src/models/train_model.py` pour entraîner dans le conteneur.

---

## Entraînement

```bash
# En local
make train

# Dans le conteneur Docker (recommandé pour la cohérence des versions)
docker exec accidents_api python3 src/models/train_model.py
```

Logs attendus :

```
2025-05-10T14:32:01 - INFO - Chargement des données...
2025-05-10T14:32:02 - INFO - Train: (37800, 28), Test: (16200, 28)
2025-05-10T14:32:02 - INFO - Entraînement du RandomForestClassifier...
2025-05-10T14:32:18 - INFO - Accuracy sur le test set : 0.7742
2025-05-10T14:32:18 - INFO - Modèle sauvegardé dans .../trained_model.joblib
```

---

## Ré-entraînement via l'API

L'endpoint `POST /retrain` déclenche un ré-entraînement en arrière-plan (thread séparé) :

```bash
curl -X POST http://localhost:8000/retrain
```

Le modèle est rechargé en mémoire automatiquement à la fin du script, sans redémarrage de l'API.

---

## Évolutions prévues (Phase 2+)

- Tracking des expériences avec **MLflow** (paramètres, métriques, artefacts).
- Comparaison avec d'autres algorithmes (XGBoost, LightGBM, Logistic Regression).
- Optimisation des hyperparamètres (GridSearchCV / Optuna).
- Métriques supplémentaires : F1-score, ROC-AUC, PR-AUC.
- Gestion du déséquilibre de classes : `class_weight="balanced"` ou SMOTE.
- Drift detection et monitoring en production.
