
.PHONY: help install lint test api train docker-up docker-down clean

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
	@echo "  make train       Train model"
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
	PYTHONPATH=. $(UVICORN) api.main_api:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow server \
	--host 127.0.0.1 \
	--port 8080 \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlruns \
	--serve-artifacts

train:
	PYTHONPATH=. $(PYTHON) src/models/train_model.py


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