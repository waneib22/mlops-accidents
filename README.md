# Projet 3 - MLOps Accidents Routiers

## Contexte

Ce projet vise à mettre en place une architecture MLOps complète appliquée à un cas d’usage de prédiction de la gravité des accidents routiers en France.
Les données utilisées proviennent de bases publiques d’accidents. Un modèle de machine learning simple sera entraîné afin de prédire la gravité d’un accident à partir de caractéristiques disponibles (localisation, conditions, etc.).
Le modèle de machine learning n’est pas l’objectif principal. Il sert plutôt de support pour construire une chaîne MLOps complète et reproductible.
## Les objectifs du projet
-	Construire une chaîne simple de traitement des données et du modèle.
Récupérer des données, les préparer rapidement, entraîner un modèle simple et évaluer ses résultats. 
-	Mettre en place un projet organisé et reproductible.
Créer une structure claire, avec un environnement (venv, requirements.txt) permettant à n’importe qui de relancer le projet facilement.
-	Déployer le modèle sous forme de service utilisable
Créer une API permettant d’utiliser le modèle pour faire des prédictions à partir de nouvelles données. 
-	Mettre en place des outils pour suivre et sauvegarder le travail réalisé.
Suivre les expériences (MLflow), sauvegarder les modèles et garder une trace des résultats. 
-	Préparer le projet pour une utilisation “réelle”.
Ajouter progressivement des éléments comme la conteneurisation (Docker), l’automatisation et le suivi des performances.

## Structure du dossier

-  dossier `data/` : données brutes (raw) et données traitées (processed)
- dossier `src/` : code Python
- dossier `models/` : modèles sauvegardés
- dossier `api/` : API d’inférence
- Dockerfile
- docker-compose.yml
- Makefile
- requirements.txt
- README.md

## Environnement virtuel et dépendances

Environnement virtuel général créé par anacondas: projet_accidents
 Installer les dépendances via:
 pip install -r requirements.txt

## API

L'API est développée avec FastAPI.
Endpoints disponibles :
- GET / : Retourne un message indiquant que l'API est opérationnelle.
- POST /predict : Permet d'effectuer une prédiction de gravité à partir de nouvelles données d'accident.

 ## Conteneurisation

Le projet est conteneurisé avec Docker.
- Construction de l'image : docker build -t accidents .
- Lancement via Docker Compose : docker compose up --build
    Deux services sont définis :
    - api : expose l'API FastAPI
    - trainer : exécute la chaîne collect.py → process.py → train.py

## Makefile

Quelques raccourcis sont disponibles :
- make api
- make train
- make docker-up