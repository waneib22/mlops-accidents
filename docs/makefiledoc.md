# Documentation du Makefile

Ce `Makefile` centralise les commandes utilisées dans le projet **Accidents Routiers - MLOps Pipeline**. Il permet d'automatiser les tâches courantes liées à l'installation, la gestion des données, l'entraînement du modèle, l'API, MLflow, Docker et le nettoyage du projet.

-------------------------------------------------

# 1.Installation et qualité du code

## `make install`
Description : Installe toutes les dépendances du projet à l'aide de **uv**

Commande :
```bash
uv sync
```

Intérêt: 
* Installe les dépendances définies dans le projet.
* Garantit un environnement de développement cohérent pour toute l'équipe.

--------------------------------------------------------

## `make lint`
Description: Lance l'analyse statique du code avec **flake8**.
Commande:
```bash
flake8 src/
```
Intérêt:
* Vérifie le respect des conventions Python.
* Détecte certaines erreurs avant l'exécution.
* Améliore la qualité du code.

---------------------------------------------------------
---------------------------------------------------------

# 2. Gestion des données avec DVC

## `make data-pull`
Description : Télécharge les dernières versions des données depuis le stockage distant DVC.

Commande:
```bash
dvc pull data/raw.dvc data/preprocessed.dvc
```
Intérêt: 
* Récupère les données brutes et prétraitées.
* Assure que l'entraînement est effectué sur les données les plus récentes.

-------------------------------------

## `make model-pull`
Description: Télécharge le dernier modèle entraîné depuis le stockage DVC.

Commande :
```bash
dvc pull src/models/trained_model.joblib.dvc
```

Intérêt:
* Permet de récupérer rapidement un modèle déjà entraîné.
* Évite de relancer un entraînement coûteux.

-------------------------------------

## `make pull-all`
Description:Télécharge tous les fichiers suivis par DVC.

Commande :
```bash
dvc pull
```

Intérêt:
* Synchronise complètement le projet avec le stockage distant.
* Recommandé lors d'un premier clonage du dépôt.

------------------------------------

## `make push-data`
Description:
Versionne et publie les données modifiées.

Étapes réalisées:
1. Ajout des données dans DVC.
2. Mise à jour des fichiers `.dvc`.
3. Commit Git.
4. Envoi des données vers le remote DVC.
5. Push Git de la branche courante.

Intérêt:
* Garantit la traçabilité des jeux de données.
* Facilite la collaboration sur les données.

-----------------------------------

## `make push-model`
Description:
Versionne et publie le modèle entraîné.

Étapes réalisées:
1. Ajout du modèle dans DVC.
2. Mise à jour du fichier `.dvc`.
3. Commit Git.
4. Push DVC.
5. Push Git.

Intérêt:
* Conserve l'historique des modèles.
* Permet de retrouver facilement une version de modèle utilisée en production.

-------------------------------------------------------
-------------------------------------------------------

# 3. Machine Learning

## `make train`
Description:
Lance l'entraînement du modèle.

Dépendance:
Avant l'entraînement :
```bash
make data-pull
```
est exécuté automatiquement.

Commande :
```bash
PYTHONPATH=. python src/models/train_model.py
```

Intérêt:
* Utilise les données les plus récentes.
* Génère un nouveau modèle entraîné.

-------------------------------------

## `make evaluate`
Description:
Évalue les performances du modèle.

Commande :
```bash
PYTHONPATH=. python src/models/evaluate_model.py
```

Intérêt:
* Mesure les performances du modèle.
* Permet de comparer plusieurs versions.

--------------------------------------------------
--------------------------------------------------

# 4. API FastAPI

## `make api`
Description:
Démarre l'API FastAPI en mode développement.

Commande: 
```bash
PYTHONPATH=. python -m uvicorn api.main_api:app --reload --host 0.0.0.0 --port 8000
```

Intérêt:
* Expose le modèle via une API REST.
* Recharge automatiquement le serveur lors des modifications du code.

---------------------------------------------

## `make health`
Description:
Teste le point de terminaison `/health`.

Commande :
```bash
curl http://localhost:8000/health
```
Intérêt:
Vérifie rapidement que l'API est opérationnelle.

----------------------------------------------

## `make predict`
Description:
Effectue une prédiction à partir du fichier :
```text
src/models/test_features.json
```

Prérequis:
L'API doit être démarrée :

```bash
make api
```

Intérêt:
* Permet de tester rapidement le modèle.
* Vérifie le bon fonctionnement de la pipeline de prédiction.

-------------------------------------------------
-------------------------------------------------

# 5. MLflow

## `make mlflow`
Description:
Lance le serveur MLflow.

Commande :
```bash
mlflow server \
--host 127.0.0.1 \
--port 8080 \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--serve-artifacts
```
Intérêt:
* Suivi des expériences.
* Comparaison des métriques.
* Gestion des artefacts de modèles.

Accès:
```text
http://localhost:8080
```
--------------------------------------------------

# 6. Docker

## `make docker-up`
Description:
Construit et démarre l'ensemble des services Docker.

### Commande exécutée

```bash
docker compose up --build
```

### Intérêt

* Lance rapidement l'environnement complet.
* Garantit un environnement reproductible.

---

## `make docker-down`
Description:
Arrête les conteneurs Docker.

Commande :
```bash
docker compose down
```

Intérêt:
* Libère les ressources système.
* Nettoie l'environnement d'exécution.

---------------------------------------------

# 7. Nettoyage

## `make clean`
Description:
Supprime les fichiers temporaires générés par Python et les outils de test.

Éléments supprimés :

* `*.pyc`
* `__pycache__`
* `.pytest_cache`
* `htmlcov`

Intérêt:
* Nettoie l'espace de travail.
* Évite les conflits liés aux fichiers générés automatiquement.

------------------------------------------
------------------------------------------
------------------------------------------

# Workflow recommandé

## Première installation

```bash
make install
make pull-all
```

## Entraîner un nouveau modèle

```bash
make train
make evaluate
make push-model
```

## Lancer l'API

```bash
make api
```

Dans un second terminal :

```bash
make predict
```

## Travailler avec Docker

```bash
make docker-up
make docker-down
```
