.PHONY: up down build logs backend-shell worker-shell migrate revision test test-backend test-frontend fmt lint clean

up:
	docker compose up -d --build

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

backend-shell:
	docker compose exec backend bash

worker-shell:
	docker compose exec worker bash

migrate:
	docker compose exec backend alembic upgrade head

revision:
	docker compose exec backend alembic revision --autogenerate -m "$(m)"

test-backend:
	docker compose exec backend pytest -q

test-frontend:
	cd frontend && npm run test -- --run

test: test-backend test-frontend

fmt:
	docker compose exec backend ruff format app tests

lint:
	docker compose exec backend ruff check app tests

clean:
	docker compose down -v
	rm -rf frontend/node_modules frontend/dist backend/.venv
