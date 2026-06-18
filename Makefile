
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
	@echo "  make dvc-repro   Run full DVC pipeline (import data + preprocess + train + evaluate if needed)"
	@echo "  make pull-all    Pull all DVC-tracked data and models (manual execution)"
	@echo "  make train       Train model(manual execution)"
	@echo "  make evaluate    Evaluate model (manual execution)"
	@echo "  make push-data   Push data changes to DVC remote"
	@echo "  make push-model  Push trained model to DVC remote"
	@echo "  make pip-status  Show the status pipeline , changed or not "
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

api:
	$(UVICORN) api.main_api:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow server \
	--host 127.0.0.1 \
	--port 8080 \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlruns \
	--serve-artifacts

dvc-repro:   # Recupere la totalité de la pipeline DVC via dvc.yaml
	dvc repro

pull-all: # DVC : récupération des données et du modèle
	dvc pull

push-all: # Envoie sur DVC et Git si nvx changements
	git add dvc.yaml dvc.lock
	git commit -m "Sync DVC pipeline" || true
	dvc push
	git push origin $$(git branch --show-current)

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
	
docker-up: pull-all #pour être sur que l'on a bien recuperer les données et le modèle 
	docker compose up --build

docker-down:
	docker compose down

clean:
	python -c "import shutil, pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	rmdir /s /q __pycache__ 2>NUL || true
	rmdir /s /q .pytest_cache 2>NUL || true
	rmdir /s /q htmlcov 2>NUL || true