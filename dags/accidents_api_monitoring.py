"""Supervision périodique de l'API de prédiction et de sa collecte Prometheus.

Complète la stack Prometheus/Grafana, qui observe le trafic réel : ce DAG
teste activement le service, y compris quand personne ne l'utilise. Trois
vérifications indépendantes, qui tournent en parallèle :

- l'API répond et son modèle est chargé ;
- une prédiction de bout en bout renvoie un résultat exploitable ;
- Prometheus collecte bien les métriques de tous les replicas.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import pendulum
import requests
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

PROJECT_DIR = "/opt/airflow/project"
API_BASE_URL = os.environ.get("ACCIDENTS_API_URL", "http://api:8000")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
SAMPLE_PAYLOAD = Path(PROJECT_DIR) / "src" / "models" / "test_features.json"

REQUEST_TIMEOUT = 30  # secondes

default_args = {
    "owner": "ibrahima",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "depends_on_past": False,
}


def check_api_health(**context) -> dict:
    """Vérifie que l'API répond et qu'elle a bien un modèle chargé."""
    url = f"{API_BASE_URL}/health"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise AirflowFailException(f"API injoignable sur {url} : {exc}") from exc

    if response.status_code != 200:
        raise AirflowFailException(
            f"{url} a répondu HTTP {response.status_code} au lieu de 200."
        )

    payload = response.json()
    logger.info("Réponse /health : %s", payload)

    # Une API qui répond mais sans modèle chargé renvoie 503 sur /predict :
    # autant le détecter ici plutôt qu'en démonstration.
    if not payload.get("model_loaded"):
        raise AirflowFailException(
            f"L'API répond mais aucun modèle n'est chargé : {payload}"
        )

    return payload


def smoke_test_prediction(**context) -> dict:
    """Envoie une vraie prédiction et contrôle la forme de la réponse."""
    if not SAMPLE_PAYLOAD.exists():
        raise AirflowFailException(f"Jeu d'essai introuvable : {SAMPLE_PAYLOAD}")

    with SAMPLE_PAYLOAD.open() as f:
        features = json.load(f)

    url = f"{API_BASE_URL}/predict"
    try:
        response = requests.post(url, json=features, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise AirflowFailException(f"Appel à {url} en échec : {exc}") from exc

    if response.status_code != 200:
        raise AirflowFailException(
            f"{url} a répondu HTTP {response.status_code} : {response.text[:300]}"
        )

    payload = response.json()
    prediction = payload.get("prediction")

    # Le modèle est un classifieur binaire : toute autre valeur signale une
    # régression, même si l'API a répondu 200.
    if prediction not in (0, 1):
        raise AirflowFailException(
            f"Prédiction inattendue (attendu 0 ou 1) : {payload}"
        )

    logger.info("Prédiction de contrôle : %s", payload)
    return payload


def check_prometheus_targets(**context) -> list:
    """Contrôle que Prometheus collecte bien toutes les instances de l'API."""
    url = f"{PROMETHEUS_URL}/api/v1/query"
    try:
        response = requests.get(
            url, params={"query": 'up{job="accidents_api"}'}, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        raise AirflowFailException(f"Prometheus injoignable sur {url} : {exc}") from exc

    if response.status_code != 200:
        raise AirflowFailException(
            f"{url} a répondu HTTP {response.status_code} : {response.text[:300]}"
        )

    results = response.json().get("data", {}).get("result", [])
    if not results:
        raise AirflowFailException(
            'Aucune target pour le job "accidents_api" : Prometheus ne collecte rien. '
            "Vérifier `metrics_path` et la découverte DNS dans prometheus.yml."
        )

    # up == 1 pour une instance collectée, 0 pour une instance injoignable.
    down = [r["metric"].get("instance") for r in results if r["value"][1] != "1"]
    if down:
        raise AirflowFailException(
            f"{len(down)} instance(s) non collectée(s) par Prometheus : {down}"
        )

    instances = [r["metric"].get("instance") for r in results]
    logger.info("%d instance(s) collectée(s) : %s", len(instances), instances)
    return instances


with DAG(
    dag_id="accidents_api_monitoring",
    description="Supervision active de l'API de prédiction et de sa collecte Prometheus",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    schedule="*/10 * * * *",       # toutes les 10 minutes
    catchup=False,
    max_active_runs=1,
    tags=["accidents", "monitoring"],
) as dag:

    health = PythonOperator(
        task_id="check_api_health",
        python_callable=check_api_health,
        doc_md="Appelle `/health` et vérifie que le modèle est chargé.",
    )

    smoke_test = PythonOperator(
        task_id="smoke_test_prediction",
        python_callable=smoke_test_prediction,
        doc_md="Envoie `src/models/test_features.json` sur `/predict` et contrôle la réponse.",
    )

    prometheus_targets = PythonOperator(
        task_id="check_prometheus_targets",
        python_callable=check_prometheus_targets,
        doc_md="Interroge Prometheus et vérifie que toutes les instances de l'API remontent `up`.",
    )

    # Aucune dépendance entre les trois tâches : elles tournent en parallèle.
    # Une API en panne et une collecte Prometheus cassée sont deux problèmes
    # distincts, on veut voir les deux d'un coup plutôt que le premier seul.
