install:
	pip install -r requirements.txt

collect:
	python src/collect.py

process:
	python src/process.py

train:
	python src/train.py

api:
	uvicorn api.main:app --reload

docker-build:
	docker build -t accidents .

docker-api:
	docker compose up api

docker-train:
	docker compose up trainer

docker-up:
	docker compose up --build