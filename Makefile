.PHONY: help install lint test api train docker-up docker-down clean

PYTHON   := python3
UVICORN  := uvicorn
SRC_DIRS := src tests

help:
	@echo ""
	@echo "  Accidents Routiers — MLOps Pipeline"
	@echo "  ─────────────────────────────────────"
	@echo "  make install     Install all dependencies"
	@echo "  make lint        Run flake8 linter"
	@echo "  make test        Run test suite with coverage"
	@echo "  make api         Start API in development mode"
	@echo "  make train       Train the model"
	@echo "  make health      Check API health"
	@echo "  make predict     Send an example prediction"
	@echo "  make retrain     Trigger retraining via API"
	@echo "  make docker-up   Build & start full stack (API + Prometheus + Grafana)"
	@echo "  make docker-down Stop Docker stack"
	@echo "  make clean       Remove build artifacts & cache"
	@echo ""

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	flake8 $(SRC_DIRS) --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 $(SRC_DIRS) --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test:
	PYTHONPATH=. coverage run -m pytest tests/ -v
	coverage report --min-coverage=60
	coverage html

api:
	PYTHONPATH=. $(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

train:
	PYTHONPATH=. $(PYTHON) src/models/train_model.py

health:
	@curl -s http://localhost:8000/health | python3 -m json.tool

predict:
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"place":10,"catu":3,"sexe":1,"secu1":0.0,"year_acc":2021,"victim_age":60,"catv":2,"obsm":1,"motor":1,"catr":3,"circ":2,"surf":1,"situ":1,"vma":50,"jour":7,"mois":12,"lum":5,"dep":77,"com":77317,"agg_":2,"int":1,"atm":0,"col":6,"lat":48.60,"long":2.89,"hour":17,"nb_victim":2,"nb_vehicules":1}' \
		| python3 -m json.tool

retrain:
	@curl -s -X POST http://localhost:8000/retrain | python3 -m json.tool

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	rm -rf .coverage htmlcov/ .pytest_cache/
