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

## Chargement du modèle

Au démarrage, l'API charge le modèle **Random Forest** depuis le **MLflow Model Registry**
(`models:/Modele Random Forest /1`) et charge en mémoire les jeux de test
`X_test.csv` / `y_test.csv` (utilisés par `/metrics` et `/report`).


---

## Endpoints

### `GET /`

Vérifie que l'API répond.

**Réponse 200**

```json
{
  "message": "API active"
}
```

---

### `GET /health`

Endpoint de bonne pratique pour vérifier la disponibilité du service.

**Réponse 200**

```json
{
  "status": "je suis une API qui fonctionne au top de sa forme 😏"
}
```

> Cet endpoint ne vérifie pas réellement l'état du modèle : il renvoie toujours
> ce message tant que le processus tourne. Il n'y a pas de champ `model_loaded`,
> `uptime_seconds` ni `api_version`.

---

### `POST /predict`

Prédit la gravité d'un accident de la route à partir des features fournies.

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

| Champ | Type | Description |
|-------|------|-------------|
| `place` | int | Place occupée dans le véhicule |
| `catu` | int | Catégorie d'usager (1=conducteur, 2=passager, 3=piéton) |
| `sexe` | int | Sexe (1=masculin, 2=féminin) |
| `secu1` | float | Équipement de sécurité 1 |
| `year_acc` | int | Année de l'accident |
| `victim_age` | int | Âge de la victime |
| `catv` | int | Catégorie de véhicule |
| `obsm` | int | Obstacle mobile heurté |
| `motor` | int | Type de motorisation |
| `catr` | int | Catégorie de route |
| `circ` | int | Régime de circulation |
| `surf` | int | État de la surface |
| `situ` | int | Situation de l'accident |
| `vma` | int | Vitesse maximale autorisée |
| `jour` | int | Jour de la semaine |
| `mois` | int | Mois de l'accident |
| `lum` | int | Conditions d'éclairage |
| `dep` | int | Code département |
| `com` | int | Code commune INSEE |
| `agg_` | int | Localisation (1=hors agglomération, 2=en agglomération) |
| `int` | int | Type d'intersection |
| `atm` | int | Conditions atmosphériques |
| `col` | int | Type de collision |
| `lat` | float | Latitude (WGS84) |
| `long` | float | Longitude (WGS84) |
| `hour` | int | Heure de l'accident |
| `nb_victim` | int | Nombre de victimes impliquées |
| `nb_vehicules` | int | Nombre de véhicules impliqués |

Tous les champs sont **requis** (aucune valeur par défaut définie dans le schéma Pydantic).

**Réponse 200**

```json
{
  "prediction": 1
}
```

| Champ | Type | Description |
|-------|------|-------------|
| `prediction` | `integer` | Classe prédite par le modèle |


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

---

### `GET /metrics`

Calcule les métriques de performance du modèle sur le jeu de test (`X_test.csv` / `y_test.csv`
chargé au démarrage).

**Réponse 200**

```json
{
  "accuracy": 0.87,
  "precision": 0.85,
  "recall": 0.86,
  "f1_score": 0.855
}
```

| Champ | Type | Description |
|-------|------|-------------|
| `accuracy` | `float` | Exactitude globale |
| `precision` | `float` | Précision moyenne pondérée (`average="weighted"`) |
| `recall` | `float` | Rappel moyen pondéré (`average="weighted"`) |
| `f1_score` | `float` | F1-score moyen pondéré (`average="weighted"`) |

> Ces métriques sont recalculées à **chaque appel** (pas de mise en cache) : le
> modèle refait une prédiction complète sur tout `X_test` à chaque requête.

---

### `GET /report`

Retourne le rapport de classification complet (`sklearn.metrics.classification_report`,
au format dictionnaire) sur le jeu de test.

**Réponse 200**

```json
{
  "0": {
    "precision": 0.88,
    "recall": 0.90,
    "f1-score": 0.89,
    "support": 1200
  },
  "1": {
    "precision": 0.82,
    "recall": 0.79,
    "f1-score": 0.80,
    "support": 800
  },
  "accuracy": 0.87,
  "macro avg": { "precision": 0.85, "recall": 0.845, "f1-score": 0.845, "support": 2000 },
  "weighted avg": { "precision": 0.87, "recall": 0.87, "f1-score": 0.87, "support": 2000 }
}
```

---

## Codes HTTP utilisés

| Code | Signification |
|------|---------------|
| 200 | Succès |
| 422 | Erreur de validation  |

---

## Exemple complet avec curl

```bash
# Test de disponibilité
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

# Métriques du modèle
curl http://localhost:8000/metrics | python3 -m json.tool

# Rapport de classification complet
curl http://localhost:8000/report | python3 -m json.tool
```

---

## Lancement

```bash
uvicorn api.main_api:app --reload
```

Nécessite qu'un serveur MLflow soit accessible à l'URI configurée dans `main_api.py`
(`http://mlflow:8080` en Docker Compose, `http://localhost:8080` en local hors Docker)
et que le modèle `Modele Random Forest /1` soit bien enregistré dans le Model Registry.
