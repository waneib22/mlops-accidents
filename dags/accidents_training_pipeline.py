"""Pipeline de ré-entraînement du modèle de gravité des accidents.

Orchestre les quatre étapes du pipeline DVC (import → preprocessing →
entraînement → évaluation), puis décide si le modèle produit est
suffisamment bon pour être promu.

Chaque étape réutilise telle quelle les scripts de ``src/`` : Airflow
orchestre, il ne réimplémente pas la logique métier.

Note sur les scripts interactifs
--------------------------------
``import_raw_data.py`` et ``make_dataset.py`` posent des questions à
l'utilisateur (``click.prompt`` et ``check_structure.check_existing_file``).
Sans stdin, Python lèverait ``EOFError`` et la tâche échouerait. Les réponses
leur sont donc fournies par le pipe shell, ce qui évite de modifier des
scripts partagés avec la pipeline DVC et la CI.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

logger = logging.getLogger(__name__)

# Racine du projet montée dans les conteneurs Airflow (voir docker-compose.yml).
PROJECT_DIR = "/opt/airflow/project"
METRICS_FILE = Path(PROJECT_DIR) / "metrics" / "evaluation_metrics.json"

# Seuil de promotion, surchargeable sans redéploiement via la variable
# Airflow "accidents_min_accuracy" (Admin > Variables).
DEFAULT_MIN_ACCURACY = 0.70

default_args = {
    "owner": "ibrahima",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}


def _read_metrics() -> dict:
    """Relit les métriques produites par l'étape d'évaluation."""
    if not METRICS_FILE.exists():
        raise AirflowFailException(
            f"Fichier de métriques introuvable : {METRICS_FILE}. "
            "L'étape d'évaluation a-t-elle bien tourné ?"
        )
    with METRICS_FILE.open() as f:
        return json.load(f)


def _threshold() -> float:
    """Seuil d'accuracy en dessous duquel un modèle n'est pas promu."""
    return float(Variable.get("accidents_min_accuracy", default_var=DEFAULT_MIN_ACCURACY))


def choose_next_step(**context) -> str:
    """Compare l'accuracy au seuil et aiguille vers la promotion ou l'alerte."""
    metrics = _read_metrics()
    accuracy = float(metrics["accuracy"])
    threshold = _threshold()

    # Poussées en XCom pour être lisibles depuis l'UI, pas pour être consommées :
    # les tâches suivantes relisent la source de vérité, ce qui les rend
    # rejouables indépendamment (relance après un `clear` partiel, par exemple).
    context["ti"].xcom_push(key="metrics", value=metrics)
    context["ti"].xcom_push(key="threshold", value=threshold)

    logger.info("Métriques du modèle : %s", metrics)
    logger.info("Seuil de promotion (accuracy) : %.4f", threshold)

    if accuracy >= threshold:
        logger.info("Accuracy %.4f >= %.4f — modèle promu.", accuracy, threshold)
        return "promote_model"

    logger.warning("Accuracy %.4f < %.4f — promotion refusée.", accuracy, threshold)
    return "alert_low_quality"


def promote_model(**context) -> dict:
    """Valide le modèle entraîné.

    Le modèle est déjà enregistré dans le Model Registry MLflow par
    ``train_model.py`` : cette étape acte la validation et journalise le
    résultat, que l'on retrouve dans les logs de la tâche.
    """
    metrics = _read_metrics()
    threshold = _threshold()

    logger.info(
        "Modèle promu — accuracy=%.4f (seuil %.4f), f1=%.4f",
        metrics["accuracy"], threshold, metrics["f1_score"],
    )
    logger.info(
        "Nouvelle version disponible dans le Model Registry MLflow "
        "(modèle « Modele Random Forest »). L'API la chargera à son prochain "
        "démarrage ou via son endpoint /retrain."
    )
    return metrics


def alert_low_quality(**context) -> None:
    """Fait échouer le run quand le modèle est sous le seuil.

    L'échec est volontaire : il rend le problème visible dans l'UI Airflow
    plutôt que de laisser passer silencieusement un modèle dégradé.
    """
    metrics = _read_metrics()
    threshold = _threshold()

    raise AirflowFailException(
        f"Modèle rejeté : accuracy={metrics['accuracy']:.4f} "
        f"< seuil={threshold:.4f}. Métriques complètes : {metrics}"
    )


with DAG(
    dag_id="accidents_training_pipeline",
    description="Ré-entraînement complet du modèle de gravité des accidents",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    schedule="0 6 * * 1",          # tous les lundis à 6h
    catchup=False,                 # pas de rattrapage des runs passés
    max_active_runs=1,             # jamais deux entraînements en parallèle
    tags=["accidents", "ml", "training"],
) as dag:

    # `yes y` répond « oui » à toutes les demandes d'écrasement de fichiers.
    import_raw_data = BashOperator(
        task_id="import_raw_data",
        bash_command="yes y | python src/data/import_raw_data.py",
        cwd=PROJECT_DIR,
        append_env=True,
        env={"PYTHONPATH": PROJECT_DIR},
        doc_md="Télécharge les CSV bruts BAAC 2021 depuis le bucket S3 vers `data/raw`.",
    )

    # Les deux premières lignes répondent aux `click.prompt` (dossier d'entrée,
    # dossier de sortie), `yes y` à toutes les confirmations d'écrasement.
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=(
            "{ echo 'data/raw'; echo 'data/preprocessed'; yes y; } "
            "| python src/data/make_dataset.py"
        ),
        cwd=PROJECT_DIR,
        append_env=True,
        env={"PYTHONPATH": PROJECT_DIR},
        doc_md=(
            "Fusionne et nettoie les CSV bruts, puis écrit "
            "`X_train/X_test/y_train/y_test` dans `data/preprocessed`."
        ),
    )

    # Le garde-fou évite le traceback dagshub « token can't be empty », illisible
    # et émis seulement après plusieurs minutes : sans jeton, autant échouer
    # tout de suite avec la marche à suivre.
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            'if [ -z "${MLFLOW_TRACKING_URI:-}" ] '
            '|| [ -z "${MLFLOW_TRACKING_USERNAME:-}" ]; then\n'
            '  echo "ERREUR : configuration MLflow/DagsHub incomplete."\n'
            '  echo "  MLFLOW_TRACKING_URI      : ${MLFLOW_TRACKING_URI:-<vide>}"\n'
            '  echo "  MLFLOW_TRACKING_USERNAME : ${MLFLOW_TRACKING_USERNAME:+<defini>}"\n'
            '  echo ""\n'
            '  echo "train_model.py n appelle plus dagshub.init() : il lit ces"\n'
            '  echo "variables. Sans elles MLflow ecrit en local, ce que"\n'
            '  echo "l utilisateur airflow n a pas le droit de faire."\n'
            '  echo ""\n'
            '  echo "Marche a suivre :"\n'
            '  echo "  1. Jeton sur https://dagshub.com/user/settings/tokens"\n'
            '  echo "  2. Renseigner .env (voir .env.example) a la racine"\n'
            '  echo "  3. make airflow-down && make airflow-up"\n'
            '  exit 1\n'
            'fi\n'
            'python src/models/train_model.py'
        ),
        cwd=PROJECT_DIR,
        append_env=True,
        env={"PYTHONPATH": PROJECT_DIR},
        # Inutile de réessayer une erreur d'authentification : le jeton ne
        # réapparaîtra pas tout seul deux minutes plus tard.
        retries=0,
        execution_timeout=timedelta(minutes=45),
        doc_md=(
            "Entraîne le Random Forest, journalise params et métriques dans MLflow "
            "et enregistre le modèle dans le Model Registry DagsHub. "
            "**Nécessite `DAGSHUB_USER_TOKEN` dans le fichier `.env`** "
            "(voir `docs/airflow.md`)."
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="python src/models/evaluate_model.py",
        cwd=PROJECT_DIR,
        append_env=True,
        env={"PYTHONPATH": PROJECT_DIR},
        doc_md="Évalue le modèle sur le jeu de test et écrit `metrics/evaluation_metrics.json`.",
    )

    check_model_quality = BranchPythonOperator(
        task_id="check_model_quality",
        python_callable=choose_next_step,
        doc_md="Compare l'accuracy au seuil `accidents_min_accuracy` et aiguille le pipeline.",
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
        doc_md="Acte la validation du modèle et journalise ses métriques.",
    )

    alert = PythonOperator(
        task_id="alert_low_quality",
        python_callable=alert_low_quality,
        doc_md="Fait échouer le run : le modèle est sous le seuil de qualité.",
    )

    import_raw_data >> preprocess >> train_model >> evaluate_model >> check_model_quality
    check_model_quality >> [promote, alert]
