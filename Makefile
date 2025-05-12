.PHONY: build up down logs clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

clean:
	rm -rf __pycache__ mlruns mlflow.db
	docker compose down -v