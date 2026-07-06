# Référence API — Accidents Routiers

Base URL : `http://localhost:8000`
Documentation interactive Swagger : `http://localhost:8000/docs`
Schéma OpenAPI JSON : `http://localhost:8000/openapi.json`

---

## Authentification

Aucune authentification requise .

---

## Chargement du modèle

Au démarrage, l'API charge la **dernière version** du modèle **Random Forest** depuis le **MLflow Model Registry** (`models:/Modele Random Forest/latest`) et charge en mémoire les
jeux de test `X_test.csv` / `y_test.csv` (utilisés par `/metrics` et `/report`).

> Le modèle est également rechargé automatiquement (dernière version) après un ré-entraînement déclenché via `/retrain`.


---

## Endpoints

### `GET /`

Redirige automatiquement vers la documentation interactive Swagger (`/docs`).

---

### `GET /health`

Endpoint de bonne pratique pour vérifier la disponibilité du service.

```json
{
  "api": "ok",
  "model_loaded": true,
  "mlflow_tracking": "http://localhost:8080",
  "service": "accidents-api",
  "version": "1.0.0",
  "message": "je suis une API qui fonctionne au top de sa forme 😏"
}
```

| Champ | Type | Description |
|-------|------|-------------|
| `api` | `string` | Statut de l'API (`"ok"`) |
| `model_loaded` | `boolean` | `true` si le modèle est chargé en mémoire |
| `mlflow_tracking` | `string` | URI du serveur MLflow configuré |
| `service` | `string` | Nom du service |
| `version` | `string` | Version de l'API |
| `message` | `string` | Petit Message  |

> `model_loaded` vaut `true` tant que la variable `model` n'est pas `None` 

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
  "prediction": 0
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

### `POST /retrain`
 
Déclenche un ré-entraînement du modèle **en arrière-plan** (`BackgroundTasks`). La requête
répond immédiatement (202) sans attendre la fin de l'entraînement.
 
**Comportement**
1. L'entraînement (`train()`) s'exécute en tâche de fond et enregistre le nouveau modèle dans le MLflow Model Registry.

2. Une fois terminé, l'API recharge automatiquement la **dernière version** du modèle (`models:/Modele Random Forest/latest`) pour `/predict`, `/metrics` et `/report`.

3. En cas d'erreur pendant l'entraînement, celle-ci est loggée côté serveur ([retrain] Erreur : ...`).


**Réponse 202**
 
```json
{
  "status": "accepted",
  "message": "Retraining lancé."
}
```
 
---

### `GET /metrics`

Calcule les métriques de performance du modèle sur le jeu de test (`X_test.csv` / `y_test.csv` chargé au démarrage).

**Réponse 200**

```json
{
  "accuracy": 0.7442413162705668,
  "precision": 0.7387847064328217,
  "recall": 0.7442413162705668,
  "f1_score": 0.7254516325533181
}
```

| Champ | Type | Description |
|-------|------|-------------|
| `accuracy` | `float` | Exactitude globale |
| `precision` | `float` | Précision moyenne pondérée (`average="weighted"`) |
| `recall` | `float` | Rappel moyen pondéré (`average="weighted"`) |
| `f1_score` | `float` | F1-score moyen pondéré (`average="weighted"`) |

> Ces métriques sont recalculées à **chaque appel** : le modèle refait une prédiction complète sur tout `X_test` à chaque requête.

---

### `GET /report`

Retourne le rapport de classification complet (`sklearn.metrics.classification_report`,
au format dictionnaire) sur le jeu de test.

**Réponse 200**

```json
{
  "0": {
    "precision": 0.7529002320185615,
    "recall": 0.9066778429728974,
    "f1-score": 0.8226644695145139,
    "support": 10737
  },
  "1": {
    "precision": 0.7120689655172414,
    "recall": 0.43680592279217345,
    "f1-score": 0.5414618157980989,
    "support": 5673
  },
  "accuracy": 0.7442413162705668,
  "macro avg": {"precision": 0.7324845987679014,"recall": 0.6717418828825354,"f1-score": 0.6820631426563064,"support": 16410},
  "weighted avg": {"precision": 0.7387847064328217,"recall": 0.7442413162705668,"f1-score": 0.7254516325533181,"support": 16410}
}
```

---

## Codes HTTP utilisés

| Code | Signification |
|------|---------------|
| 200 | Succès |
| 202 | Requête acceptée, traitement en arrière-plan (`/retrain`) |
| 422 | Erreur de validation  |

---

## Exemple complet avec curl

```bash
# Test de disponibilité
curl http://localhost:8000/health 

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
  }' 

# Ré-entraînement 
curl -X POST http://localhost:8000/retrain 

# Métriques du modèle
curl http://localhost:8000/metrics 

# Rapport de classification complet
curl http://localhost:8000/report 
```

---

## Lancement

```bash
uvicorn api.main_api:app --reload
```

Nécessite qu'un serveur MLflow soit accessible à l'URI configurée dans `main_api.py`
(`http://mlflow:8080` en Docker Compose, `http://localhost:8080` en local hors Docker,
actif actuellement) et qu'une version du modèle `Modele Random Forest` soit bien
enregistrée dans le Model Registry.