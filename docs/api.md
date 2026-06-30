# A REMETTRE A JOUR 
# MODIFIER DEPUIS 





# Référence API — Accidents Routiers

Base URL : `http://localhost:8000`  
Documentation interactive Swagger : `http://localhost:8000/docs`  
Schéma OpenAPI JSON : `http://localhost:8000/openapi.json`

---

## Authentification

Aucune authentification requise (projet Phase 1).

---

## Endpoints

### `GET /`

Redirige automatiquement vers `/docs` (code HTTP 307/308).

---

### `GET /health`

Retourne le statut opérationnel de l'API et du modèle.

**Réponse 200**

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

| Champ | Type | Description |
|-------|------|-------------|
| `status` | `string` | `"ok"` si le modèle est chargé, `"degraded"` sinon |
| `model_loaded` | `boolean` | `true` si le modèle est disponible |
| `model_type` | `string \| null` | Nom de la classe du modèle sklearn |
| `n_features` | `integer` | Nombre de features attendues (28) |
| `uptime_seconds` | `float` | Temps écoulé depuis le démarrage de l'API |
| `api_version` | `string` | Version sémantique de l'API |

> Le code HTTP est toujours `200` — surveiller le champ `status` pour détecter le mode dégradé.

---

### `GET /stats`

Retourne les compteurs de prédictions depuis le démarrage.

**Réponse 200**

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

| Champ | Type | Description |
|-------|------|-------------|
| `total_predictions` | `integer` | Nombre total de requêtes `/predict` traitées |
| `predictions_by_label.prioritaire` | `integer` | Prédictions de classe 1 |
| `predictions_by_label.non_prioritaire` | `integer` | Prédictions de classe 0 |
| `uptime_seconds` | `float` | Temps écoulé depuis le démarrage |

> Les compteurs sont **en mémoire** et se réinitialisent au redémarrage du service.

---

### `GET /model/info`

Retourne les métadonnées et hyperparamètres du modèle chargé.

**Réponse 200**

```json
{
  "type": "RandomForestClassifier",
  "n_features": 28,
  "features": ["place", "catu", "sexe", "secu1", "year_acc", "victim_age",
               "catv", "obsm", "motor", "catr", "circ", "surf", "situ", "vma",
               "jour", "mois", "lum", "dep", "com", "agg_", "int", "atm", "col",
               "lat", "long", "hour", "nb_victim", "nb_vehicules"],
  "params": {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
    "max_depth": null
  }
}
```

**Réponse 503** — modèle non chargé

```json
{"detail": "Modèle non chargé."}
```

---

### `POST /predict`

Prédit la gravité d'un accident de la route.

**Corps de la requête** (`application/json`)

```json
{
  "place": 10,
  "catu": 3,
  "sexe": 1,
  "secu1": 0.0,
  "year_acc": 2021,
  "victim_age": 60,
  "catv": 2,
  "obsm": 1,
  "motor": 1,
  "catr": 3,
  "circ": 2,
  "surf": 1,
  "situ": 1,
  "vma": 50,
  "jour": 7,
  "mois": 12,
  "lum": 5,
  "dep": 77,
  "com": 77317,
  "agg_": 2,
  "int": 1,
  "atm": 0,
  "col": 6,
  "lat": 48.60,
  "long": 2.89,
  "hour": 17,
  "nb_victim": 2,
  "nb_vehicules": 1
}
```

#### Description des champs

| Champ | Type | Source BAAC | Description |
|-------|------|-------------|-------------|
| `place` | float | usagers | Place occupée dans le véhicule |
| `catu` | float | usagers | Catégorie d'usager (1=conducteur, 2=passager, 3=piéton) |
| `sexe` | float | usagers | Sexe (1=masculin, 2=féminin) |
| `secu1` | float | usagers | Équipement de sécurité 1 |
| `year_acc` | float | caractéristiques | Année de l'accident |
| `victim_age` | float | calculé | Âge de la victime (année acc − année naissance) |
| `catv` | float | véhicules | Catégorie de véhicule |
| `obsm` | float | véhicules | Obstacle mobile heurté |
| `motor` | float | véhicules | Type de motorisation |
| `catr` | float | lieux | Catégorie de route |
| `circ` | float | lieux | Régime de circulation |
| `surf` | float | lieux | État de la surface |
| `situ` | float | lieux | Situation de l'accident |
| `vma` | float | lieux | Vitesse maximale autorisée |
| `jour` | float | caractéristiques | Jour de la semaine |
| `mois` | float | caractéristiques | Mois de l'accident |
| `lum` | float | caractéristiques | Conditions d'éclairage |
| `dep` | float | caractéristiques | Code département |
| `com` | float | caractéristiques | Code commune INSEE |
| `agg_` | float | caractéristiques | Localisation (1=hors agglomération, 2=en agglomération) |
| `int` | float | caractéristiques | Type d'intersection |
| `atm` | float | caractéristiques | Conditions atmosphériques |
| `col` | float | caractéristiques | Type de collision |
| `lat` | float | caractéristiques | Latitude (WGS84) |
| `long` | float | caractéristiques | Longitude (WGS84) |
| `hour` | float | calculé | Heure de l'accident (extrait de `hrmn`) |
| `nb_victim` | float | calculé | Nombre de victimes impliquées |
| `nb_vehicules` | float | calculé | Nombre de véhicules impliqués |

**Réponse 200**

```json
{
  "prediction": 1,
  "label": "prioritaire",
  "probability": 0.8423,
  "confidence": "high"
}
```

| Champ | Type | Valeurs possibles | Description |
|-------|------|-------------------|-------------|
| `prediction` | `integer` | `0`, `1` | Classe prédite |
| `label` | `string` | `"prioritaire"`, `"non-prioritaire"` | Label lisible |
| `probability` | `float` | `[0.0, 1.0]` | Probabilité de la classe prédite |
| `confidence` | `string` | `"high"`, `"medium"`, `"low"` | Niveau de confiance |

Règles de confiance :

| `probability` | `confidence` |
|---------------|-------------|
| ≥ 0.80 | `high` |
| ≥ 0.60 | `medium` |
| < 0.60 | `low` |

**Réponse 422** — champ manquant ou invalide

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "vma"],
      "msg": "Field required"
    }
  ]
}
```

**Réponse 503** — modèle non chargé

```json
{"detail": "Modèle non disponible. Lancez d'abord l'entraînement."}
```

---

### `POST /retrain`

Déclenche un ré-entraînement du modèle en arrière-plan.

**Pas de corps requis.**

**Réponse 202**

```json
{
  "status": "accepted",
  "message": "Retraining started in background."
}
```

Le ré-entraînement exécute `src/models/train_model.py` en sous-processus.  
Le modèle est rechargé automatiquement en mémoire à la fin du script.

> Prérequis : les fichiers `data/processed/X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv` doivent exister.

---

## Codes HTTP utilisés

| Code | Signification |
|------|---------------|
| 200 | Succès |
| 202 | Requête acceptée (traitement asynchrone) |
| 307/308 | Redirection |
| 422 | Erreur de validation Pydantic |
| 503 | Service indisponible (modèle non chargé) |

---

## Exemple complet avec curl

```bash
# Santé
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

# Ré-entraînement
curl -X POST http://localhost:8000/retrain | python3 -m json.tool
```
