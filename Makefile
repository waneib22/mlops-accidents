
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
	@echo "  make test        Run test suite with coverage"
	@echo "  make api         Start API"
	@echo "  make train       Train model"
	@echo "  make predict     Predictions du model (lancer make api puis faire la commande dans nouveau terminal)"
	@echo "  make docker-up   Start docker stack"
	@echo "  make docker-down Stop docker stack"
	@echo "  make clean       Clean project"
	@echo ""

install:
	uv sync

lint:
	flake8 $(SRC_DIRS) --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 $(SRC_DIRS) --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test:
	PYTHONPATH=. coverage run -m pytest tests/ -v
	coverage report --min-coverage=60
	coverage html

api:
	PYTHONPATH=. $(UVICORN) api.main_api:app --reload --host 0.0.0.0 --port 8000


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