# Orchestration — Apache Airflow

Airflow orchestre le cycle de vie du modèle : ré-entraînement périodique d'un côté,
supervision active de l'API de l'autre.

- **UI** : http://localhost:8080 — identifiants `admin` / `admin`
- **Version** : Airflow 2.10.5, exécuteur `LocalExecutor`, métadonnées dans PostgreSQL

---

## 1. Démarrage

```bash
make airflow-up      # build + démarrage (le webserver met ~1 min à répondre)
make airflow-dags    # liste les DAGs et les erreurs d'import
make airflow-logs    # logs du scheduler
make airflow-down    # arrêt
```

Les DAGs arrivent **en pause** (`DAGS_ARE_PAUSED_AT_CREATION`) : il faut les activer
depuis l'UI, ce qui évite qu'un pipeline d'entraînement parte tout seul au premier
démarrage.

> Les services Airflow sont derrière le **profil Compose `airflow`**. `docker compose up`
> et `make docker-up` restent strictement identiques à avant : ils ne démarrent que
> l'API, nginx, Prometheus et Grafana.

---

## 2. Architecture

```
┌──────────────────┐        ┌──────────────────┐
│ airflow-webserver│        │ airflow-scheduler│
│    (UI :8080)    │        │  (exécute les    │
└────────┬─────────┘        │     tâches)      │
         │                  └────────┬─────────┘
         └──────────┬────────────────┘
                    ▼
         ┌──────────────────────┐
         │   airflow-postgres   │   métadonnées (runs, XCom, Variables)
         └──────────────────────┘

Le scheduler accède à :
  /opt/airflow/dags      ← ./dags
  /opt/airflow/project   ← ./src, ./data, ./metrics
et joint `api:8000` et `prometheus:9090` par le réseau Compose.
```

L'image `airflow/Dockerfile` ajoute à l'image officielle les dépendances ML
(`pandas`, `scikit-learn`, `mlflow`, `dagshub`…). Elles sont installées **sous les
contraintes officielles Airflow** : les paquets partagés avec l'orchestrateur
(Flask, protobuf, SQLAlchemy) restent sur des versions compatibles. Sans ces
contraintes, MLflow tire une version de Flask qui casse le webserver.

---

## 3. DAG `accidents_training_pipeline`

Ré-entraînement complet. Planifié **tous les lundis à 6h** (`0 6 * * 1`),
`catchup=False`, un seul run à la fois.

```
import_raw_data → preprocess → train_model → evaluate_model → check_model_quality
                                                                    ├─→ promote_model
                                                                    └─→ alert_low_quality
```

| Tâche | Rôle |
|-------|------|
| `import_raw_data` | Télécharge les CSV bruts BAAC 2021 depuis S3 vers `data/raw` |
| `preprocess` | Nettoie et fusionne, écrit `X_train/X_test/y_train/y_test` dans `data/preprocessed` |
| `train_model` | Entraîne le Random Forest, journalise dans MLflow, enregistre dans le Model Registry |
| `evaluate_model` | Évalue sur le jeu de test, écrit `metrics/evaluation_metrics.json` |
| `check_model_quality` | Compare l'accuracy au seuil et aiguille le pipeline |
| `promote_model` | Acte la validation et journalise les métriques |
| `alert_low_quality` | **Fait échouer le run** : modèle sous le seuil |

### Seuil de promotion

Par défaut **0.70** d'accuracy. Modifiable sans redéploiement via la variable Airflow
`accidents_min_accuracy` (*Admin > Variables*, ou en ligne de commande) :

```bash
docker exec airflow_scheduler airflow variables set accidents_min_accuracy 0.75
```

Le modèle actuel est à **0.776** d'accuracy : il passe le seuil par défaut.

### Prérequis : identifiants DagsHub

`train_model.py` **écrit** dans MLflow sur DagsHub, ce qui exige une authentification.
C'est la cause n°1 d'échec du DAG : sans jeton, la tâche `train_model` s'arrête net
et les quatre tâches suivantes restent en `upstream_failed`.

Renseigner le jeton dans le fichier `.env` à la racine du projet :

```bash
cp .env.example .env      # si .env n'existe pas encore
```

```dotenv
DAGSHUB_USER_TOKEN=<votre_token_dagshub>
```

Jeton personnel : https://dagshub.com/user/settings/tokens

Puis recréer les conteneurs pour qu'ils prennent la variable :

```bash
make airflow-down && make airflow-up
```

Docker Compose charge `.env` automatiquement, il n'y a donc rien à exporter dans le
terminal, et le réglage survit à un changement de shell. `.env` est ignoré par git :
le jeton ne part jamais sur GitHub.

Un garde-fou en tête de `train_model` vérifie la présence de la variable et affiche
cette marche à suivre dans les logs de la tâche, plutôt que le traceback dagshub
`ValueError: token can't be empty`. Cette tâche ne fait volontairement **aucun retry** :
un défaut d'authentification ne se répare pas en réessayant deux minutes plus tard.

Les autres tâches n'en ont pas besoin : la **lecture** du modèle depuis le registry
DagsHub fonctionne en anonyme, c'est pourquoi l'API démarre sans identifiants.

---

## 4. DAG `accidents_api_monitoring`

Supervision active, **toutes les 10 minutes** (`*/10 * * * *`). Là où Prometheus et
Grafana observent le trafic réel, ce DAG teste le service même quand personne ne
l'utilise. Les trois vérifications sont indépendantes et tournent en parallèle : une
API en panne et une collecte cassée sont deux problèmes distincts.

| Tâche | Vérifie |
|-------|---------|
| `check_api_health` | `/health` répond 200 **et** `model_loaded` est vrai |
| `smoke_test_prediction` | `/predict` renvoie une prédiction valide (0 ou 1) sur un cas réel |
| `check_prometheus_targets` | `up{job="accidents_api"}` remonte, et aucune instance n'est `down` |

`check_prometheus_targets` détecte précisément la panne qui rendait les dashboards
Grafana vides (voir [monitoring.md](monitoring.md)) : la collecte cassée sans que rien
ne le signale.

---

## 5. Points d'attention

### Les scripts du pipeline sont interactifs

`import_raw_data.py` et `make_dataset.py` posent des questions à l'utilisateur
(`click.prompt`, et `check_structure.check_existing_file` pour chaque fichier à
écraser). Sans stdin, Python lève `EOFError` et la tâche échoue.

Les réponses leur sont fournies par le pipe shell :

```bash
yes y | python src/data/import_raw_data.py
{ echo 'data/raw'; echo 'data/preprocessed'; yes y; } | python src/data/make_dataset.py
```

`yes y` couvre un nombre variable de confirmations — il y en a une par fichier
déjà présent. Ce choix évite de modifier des scripts partagés avec la pipeline DVC
et la CI. La correction de fond serait de rendre `check_structure.py` non
interactif quand `stdin` n'est pas un terminal.

### `import_raw_data` réécrit `data/raw` en UTF-8

Le script fait `response.text.encode('utf-8')` sur des CSV sources encodés en
latin-1. Les fichiers réécrits sont donc **différents octet pour octet** de ceux
récupérés par `dvc pull` (≈ +2 % de taille), alors que le contenu est
sémantiquement identique — vérifié : les fichiers `data/preprocessed` produits
ensuite sont rigoureusement identiques.

Conséquence pratique : après un run du DAG, `dvc status` signale `data/raw` comme
modifié. Ce n'est pas une régression du modèle, mais il ne faut pas faire de
`dvc push` par réflexe sur cette base.

### Le split est déterministe

`make_dataset.py` utilise `train_test_split(..., random_state=42)` : rejouer le
preprocessing sur les mêmes données brutes redonne exactement les mêmes fichiers.

---

## 6. Ce qui a été vérifié en local

| Élément | Résultat |
|---------|----------|
| Image Airflow (Airflow 2.10.5 + MLflow 3.2.0 + Flask 2.2.5) | Cohabitation OK |
| Import des DAGs | Aucune erreur |
| `check_api_health`, `smoke_test_prediction`, `check_prometheus_targets` | Succès (3 instances collectées) |
| `import_raw_data`, `preprocess` | Succès, scripts interactifs pilotés par stdin |
| `evaluate_model` | Succès — accuracy 0.7761 |
| `check_model_quality` → `promote_model` | Branche « promotion » validée |
| `check_model_quality` → `alert_low_quality` | Branche « rejet » validée (seuil forcé à 0.95) |
| `train_model` | **Non validé** : nécessite un jeton DagsHub (échec attendu et net sans jeton) |
