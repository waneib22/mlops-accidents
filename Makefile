
.PHONY: help install lint test api mlflow train docker-up docker-down clean data-pull model-pull pull-all push-data push-model

PYTHON   := python
UVICORN  := python -m uvicorn
SRC_DIRS := src tests

help:
	@echo ""
	@echo "  Accidents Routiers - MLOps Pipeline"
	@echo "  ======================================="
	@echo "  make install     Install dependencies (uv)"
	@echo "  make lint        Run flake8 linter"
	@echo "  make mlflow      Start MLFlow tracking server ui"
	@echo "  make data-pull   Pull latest data from DVC remote"
	@echo "  make model-pull  Pull latest trained model from DVC remote"
	@echo "  make pull-all    Pull all DVC-tracked data and models"
	@echo "  make train       Train model"
	@echo "  make evaluate    Evaluate model"
	@echo "  make push-data   Push data changes to DVC remote"
	@echo "  make push-model  Push trained model to DVC remote"
	@echo "  make api         Start API"
	@echo "  make predict     Predictions du model (lancer make api puis faire la commande dans nouveau terminal)"
	@echo "  make docker-up   Start docker stack"
	@echo "  make docker-down Stop docker stack"
	@echo "  make clean       Clean project"
	@echo ""

install:
	uv sync

lint:    #Qualité du code
	flake8 src/

data-pull:  # DVC : récupération des données 
	dvc pull data/raw.dvc data/preprocessed.dvc

model-pull: # DVC : récupération du modèle entrainé
	dvc pull src/models/trained_model.joblib.dvc

pull-all: # DVC : récupération des données et du modèle
	dvc pull

push-data: # DVC : envoi des données si modification
	dvc add data/raw data/preprocessed
	dvc push

push-model: # DVC : envoi du modèle si modification
	dvc add src/models/trained_model.joblib
	dvc push

api:
	PYTHONPATH=. $(UVICORN) api.main_api:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow server \
	--host 127.0.0.1 \
	--port 8080 \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlruns \
	--serve-artifacts

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
	
docker-up:
	docker compose up --build

docker-down:
	docker compose down





clean:
	python -c "import shutil, pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	rmdir /s /q __pycache__ 2>NUL || true
	rmdir /s /q .pytest_cache 2>NUL || true
	rmdir /s /q htmlcov 2>NUL || true