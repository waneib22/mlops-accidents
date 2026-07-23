
.PHONY: help install lint test api mlflow train docker-up docker-down clean data-pull model-pull pull-all push-data push-model

PYTHON   := python
UVICORN  := python -m uvicorn
SRC_DIRS := src tests

help:
	@echo ""
	@echo "  Accidents Routiers - MLOps Pipeline"
	@echo "  ======================================="
	@echo "  make install     		Install dependencies (uv)"
	@echo "  make lint        		Run flake8 linter"
	@echo "  make mlflow-local      Start MLFlow tracking server ui"
	@echo "  make mlflow-dagshub    Start MLFlow with dagshub"
	@echo "  make pull-all    		Pull all DVC-tracked data and models (manual execution)" (A faire avant dvc-repro si pas de dossier data)
	@echo "  make dvc-repro   		Run full DVC pipeline (import data + preprocess + train + evaluate if needed)"
	@echo "  make train       		Train model(manual execution)"
	@echo "  make evaluate    		Evaluate model (manual execution)"
	@echo "  make push-all    		Push data and/or models change to DVC remote (dagshub choose by default)"
	@echo "  make pip-status  		Show the status pipeline , changed or not "
	@echo "  make api         		Start API"
	@echo "  make predict     		Predictions du model (lancer make api puis faire la commande dans nouveau terminal)"
	@echo "  make docker-up-full    First installation: DVC pull + build + start"
	@echo "  make docker-up         Start existing Docker stack (pour quotidien : fatsapi -> prometheus -> grafana, ne relance pas l'image à chaque lancement)"
	@echo "  make docker-build      Rebuild Docker images (Si modif du Dockerfile)"
	@echo "  make docker-down       Stop Docker stack" 
	@echo "  make clean       		Clean project"
	@echo "  make airflow-up               "
	@echo "  make tests       		Tests Apis - data: Integration Continue"
	@echo "  make monitoring-up     Start Prometheus + Grafana (Docker)"
	@echo "  make monitoring-down   Stop Prometheus + Grafana (Docker)"
	@echo "  make monitoring-logs   Show monitoring logs (Docker)"

	@echo ""

install:
	uv sync

lint:    #Qualité du code
	flake8 src/

api:
	$(UVICORN) api.main_api:app --reload --host 0.0.0.0 --port 8000

mlflow-local:
	mlflow server \
	--host 127.0.0.1 \
	--port 8080 \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlruns \
	--serve-artifacts

mlflow-dagshub:
	python -c "import dagshub; dagshub.init(repo_owner='Melanie94480', repo_name='mlops-melanie', mlflow=True)"

dvc-repro:   # Recupere la totalité de la pipeline DVC via dvc.yaml
	dvc repro

pull-all: # DVC : récupération des données et du modèle
	dvc pull  
	#dvc pull -r dagshub

push-all: # Envoie sur DVC et Git si nvx changements => juste modifier intituler en fonction des modifs
	#Ajout fichiers modifiés , nouveaux , supprimés
	git add -A 

	# Commit seulement si changements
	git diff --cached --quiet || git commit -m "Sync MLOps pipeline" 

	# push code GitHub
	git push origin $$(git branch --show-current)

	# push code DagsHub
	git push dagshub $$(git branch --show-current)

	# push data DVC (vers DagsHub remote configuré)
	dvc push

pip-status:
	dvc status

train: data-pull #(train: data-pull) — ça garantit qu'on a toujours les données à jour avant d'entraîner)
	PYTHONPATH=. $(PYTHON) src/models/train_model.py

evaluate:
	PYTHONPATH=. $(PYTHON) src/models/evaluate_model.py

health:
	@curl -s http://localhost:8000/health | python -m json.tool


predict:
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d @src/models/test_features.json \
		| python -m json.tool
	


# Docker
docker-up-full: pull-all docker-build #pour la 1ere utilisation
	docker compose up

docker-build:
	docker compose build

docker-up: #utilisation quotidienne
	docker compose up

docker-down:
	docker compose down


clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"


# Monitoring
monitoring-up:
	docker compose up -d prometheus grafana #-d pour tourner en arriere plan et libérer le terminal

monitoring-down:
	docker compose stop prometheus grafana

monitoring-logs:
	docker compose logs -f prometheus grafana