# Image Airflow du projet : l'orchestrateur officiel + les dépendances ML
# du pipeline, pour que les tâches puissent exécuter les scripts de src/.
FROM apache/airflow:2.10.5-python3.11

ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.11

COPY requirements-airflow.txt /tmp/requirements-airflow.txt

# Les contraintes officielles verrouillent les dépendances communes à Airflow
# et à MLflow (flask, protobuf, sqlalchemy…). Sans elles, l'installation de
# MLflow peut remonter une version de Flask incompatible et casser le webserver.
RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
    -r /tmp/requirements-airflow.txt
