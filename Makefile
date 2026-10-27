# Makefile for MLOps Fraud Detection Platform
# Usage: make <target>

.PHONY: help install test lint typecheck format train serve monitor dvc-pull dvc-push mlflow ui clean

# Default target
help:
	@echo "MLOps Fraud Detection Platform - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install dependencies with uv"
	@echo "  make pre-commit     - Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run all tests with coverage"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint           - Run ruff linter"
	@echo "  make typecheck      - Run mypy type checker"
	@echo "  make format         - Format code with black"
	@echo "  make check          - Run lint + typecheck + format (dry-run)"
	@echo ""
	@echo "Data & Training:"
	@echo "  make dvc-pull       - Pull data/model artifacts from DVC remote"
	@echo "  make dvc-push       - Push data/model artifacts to DVC remote"
	@echo "  make prepare        - Run data preparation stage"
	@echo "  make train          - Run training pipeline (dvc repro train)"
	@echo "  make train-full     - Run full pipeline (prepare + train)"
	@echo "  make metrics        - Show DVC metrics"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow-up      - Start MLflow server via Docker Compose"
	@echo "  make mlflow-down    - Stop MLflow server"
	@echo "  make mlflow-ui      - Open MLflow UI in browser"
	@echo ""
	@echo "Serving:"
	@echo "  make serve          - Start FastAPI server locally"
	@echo "  make serve-docker   - Start API via Docker Compose"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor        - Start drift detector"
	@echo "  make prometheus-up  - Start Prometheus + Grafana stack"
	@echo "  make grafana-ui     - Open Grafana UI in browser"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build all Docker images"
	@echo "  make docker-up      - Start all services via Docker Compose"
	@echo "  make docker-down    - Stop all services"
	@echo "  make docker-logs    - View Docker Compose logs"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci-local       - Run CI pipeline locally (lint + test + train)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Clean generated files"
	@echo "  make clean-all      - Clean everything including Docker volumes"

# ==========================================
# Setup
# ==========================================
install:
	uv sync --all-extras
	uv pip install -e .

pre-commit:
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

# ==========================================
# Development
# ==========================================
test:
	uv run pytest

test-unit:
	uv run pytest tests/unit -v

test-integration:
	uv run pytest tests/integration -v

test-contract:
	uv run pytest tests/contract -v

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

format:
	uv run black src tests

check: lint typecheck
	uv run black --check src tests

# ==========================================
# Data & Training (DVC)
# ==========================================
dvc-pull:
	dvc pull

dvc-push:
	dvc push

prepare:
	dvc repro prepare

train:
	dvc repro train

train-full:
	dvc repro

metrics:
	dvc metrics show
	dvc metrics diff

# Show params
params:
	dvc params show

# ==========================================
# MLflow
# ==========================================
mlflow-up:
	docker compose -f docker/docker-compose.yml up -d mlflow minio

mlflow-down:
	docker compose -f docker/docker-compose.yml down mlflow minio

mlflow-ui:
	@echo "Opening MLflow UI at http://localhost:5000"
	@start http://localhost:5000 2>/dev/null || xdg-open http://localhost:5000 2>/dev/null || open http://localhost:5000 2>/dev/null || echo "Please open http://localhost:5000 manually"

# ==========================================
# Serving
# ==========================================
serve:
	uv run python -m src.serve.main

serve-dev:
	uv run uvicorn src.serve.main:app --host 0.0.0.0 --port 8000 --reload

serve-docker:
	docker compose -f docker/docker-compose.yml up -d api

# ==========================================
# Monitoring
# ==========================================
monitor:
	uv run python -m src.monitor.drift

prometheus-up:
	docker compose -f docker/docker-compose.yml up -d prometheus grafana node-exporter

grafana-ui:
	@echo "Opening Grafana UI at http://localhost:3000 (admin/admin)"
	@start http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000 manually"

# ==========================================
# Docker
# ==========================================
docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

docker-ps:
	docker compose -f docker/docker-compose.yml ps

# ==========================================
# CI/CD Local Simulation
# ==========================================
ci-local: check test train
	@echo "Local CI pipeline completed successfully!"

# ==========================================
# Cleanup
# ==========================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build *.egg-info 2>/dev/null || true

clean-all: clean
	docker compose -f docker/docker-compose.yml down -v 2>/dev/null || true
	docker system prune -f 2>/dev/null || true
	rm -rf data/processed models/mlflow mlruns .dvc/cache 2>/dev/null || true