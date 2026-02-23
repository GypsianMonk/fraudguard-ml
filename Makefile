# Makefile — FraudGuard ML Developer Workflow
# Usage: make <target>

.PHONY: help setup install generate-data train serve test test-unit test-integration \
        test-coverage lint format type-check clean docker-build docker-up docker-down \
        mlflow-ui smoke-test dvc-repro

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE  := \033[34m
GREEN := \033[32m
RESET := \033[0m

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)FraudGuard ML — Developer Commands$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup: ## Full local dev setup (venv + dependencies + pre-commit)
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	python -m venv .venv
	.venv/bin/pip install --upgrade pip wheel setuptools
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pre-commit install
	cp .env.example .env 2>/dev/null || true
	mkdir -p data/raw data/processed data/features models/artifacts mlruns reports logs
	@echo "$(GREEN)Setup complete! Activate with: source .venv/bin/activate$(RESET)"

install: ## Install production dependencies only
	pip install --upgrade pip
	pip install -e .

# ── Data ──────────────────────────────────────────────────────────────────────
generate-data: ## Generate synthetic training data (500k transactions)
	@echo "$(BLUE)Generating synthetic training data...$(RESET)"
	python scripts/generate_synthetic_data.py \
		--n-samples 500000 \
		--fraud-rate 0.022 \
		--output data/raw/transactions.parquet
	@echo "$(GREEN)Data generated: data/raw/transactions.parquet$(RESET)"

generate-data-small: ## Generate small dataset for testing (10k transactions)
	python scripts/generate_synthetic_data.py \
		--n-samples 10000 \
		--fraud-rate 0.03 \
		--output data/raw/transactions_small.parquet

# ── Training ──────────────────────────────────────────────────────────────────
train: ## Run full training pipeline with MLflow tracking
	@echo "$(BLUE)Starting training pipeline...$(RESET)"
	python scripts/run_training.py \
		--data data/raw/transactions.parquet \
		--experiment fraudguard-production

train-fast: ## Quick training (no TabTransformer, small data)
	python scripts/run_training.py \
		--data data/raw/transactions_small.parquet \
		--skip-tabtransformer \
		--no-register \
		--experiment fraudguard-dev

train-tune: ## Full training with hyperparameter optimization (slow)
	python scripts/run_training.py \
		--data data/raw/transactions.parquet \
		--tune \
		--n-trials 100 \
		--experiment fraudguard-tuning

# ── Serving ───────────────────────────────────────────────────────────────────
serve: ## Start inference API server (local)
	@echo "$(BLUE)Starting FraudGuard API server...$(RESET)"
	uvicorn src.api.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

serve-prod: ## Start server in production mode (no reload)
	uvicorn src.api.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--workers 4 \
		--loop uvloop \
		--no-access-log

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run full test suite
	pytest tests/ -v --timeout=120

test-unit: ## Run unit tests only (fast, no external services)
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest tests/unit/ -v -m "not integration" --timeout=30 -n auto

test-integration: ## Run integration tests (requires Docker services)
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest tests/integration/ -v -m "integration" --timeout=120

test-coverage: ## Run tests with coverage report
	pytest tests/ \
		--cov=src \
		--cov-report=html:reports/coverage \
		--cov-report=term-missing \
		--cov-fail-under=75
	@echo "$(GREEN)Coverage report: reports/coverage/index.html$(RESET)"

smoke-test: ## Run smoke tests against local API
	python scripts/smoke_test.py \
		--base-url http://localhost:8000 \
		--api-key dev-key-local

smoke-test-staging: ## Run smoke tests against staging environment
	python scripts/smoke_test.py \
		--base-url $${STAGING_URL:?STAGING_URL required} \
		--api-key $${STAGING_API_KEY:?STAGING_API_KEY required}

# ── Code Quality ──────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check src/ tests/ scripts/ --output-format=full

format: ## Format code with ruff
	ruff format src/ tests/ scripts/
	ruff check src/ tests/ scripts/ --fix

type-check: ## Run mypy type checker
	mypy src/ --ignore-missing-imports

quality: lint type-check ## Run all code quality checks

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build: ## Build Docker images
	docker build -f docker/Dockerfile.api -t fraudguard-api:latest .
	docker build -f docker/Dockerfile.training -t fraudguard-training:latest .

docker-up: ## Start all services with Docker Compose
	@echo "$(BLUE)Starting FraudGuard stack...$(RESET)"
	docker-compose up -d
	@echo ""
	@echo "$(GREEN)Services started:$(RESET)"
	@echo "  API:       http://localhost:8000"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  Kafka UI:  http://localhost:8080"
	@echo "  Prometheus:http://localhost:9090"
	@echo "  Grafana:   http://localhost:3000 (admin/admin)"
	@echo ""

docker-down: ## Stop all Docker Compose services
	docker-compose down

docker-logs: ## Tail API logs
	docker-compose logs -f api

docker-train: ## Run training pipeline in Docker
	docker-compose run --rm training

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow-ui: ## Open MLflow UI (starts if not running)
	@echo "$(BLUE)Starting MLflow UI at http://localhost:5000$(RESET)"
	mlflow ui --port 5000 --host 0.0.0.0

# ── DVC ───────────────────────────────────────────────────────────────────────
dvc-init: ## Initialize DVC
	dvc init
	dvc remote add -d myremote s3://your-bucket/fraudguard-data

dvc-repro: ## Reproduce full DVC pipeline
	dvc repro

dvc-dag: ## Show DVC pipeline DAG
	dvc dag

dvc-push: ## Push data to DVC remote
	dvc push

dvc-pull: ## Pull data from DVC remote
	dvc pull

# ── Utilities ─────────────────────────────────────────────────────────────────
clean: ## Remove generated files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml reports/coverage/ .mypy_cache/ .ruff_cache/
	@echo "$(GREEN)Cleanup complete$(RESET)"

clean-data: ## Remove generated data (keep raw)
	rm -rf data/processed/* data/features/*

clean-models: ## Remove model artifacts
	rm -rf models/artifacts/* mlruns/

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

env-check: ## Check required environment variables
	@echo "Checking environment variables..."
	@test -n "$${MLFLOW_TRACKING_URI}" && echo "✅ MLFLOW_TRACKING_URI" || echo "⚠️  MLFLOW_TRACKING_URI not set (using default)"
	@test -n "$${REDIS_URL}" && echo "✅ REDIS_URL" || echo "⚠️  REDIS_URL not set (using default)"
	@test -n "$${API_KEYS}" && echo "✅ API_KEYS" || echo "⚠️  API_KEYS not set (using dev default)"

curl-predict: ## Test predict endpoint with curl (requires running API)
	curl -s -X POST http://localhost:8000/api/v1/predict \
		-H "Content-Type: application/json" \
		-H "X-API-Key: dev-key-local" \
		-d '{ \
			"transaction_id": "txn_make_test", \
			"user_id": "usr_12345", \
			"amount": 4999.99, \
			"currency": "USD", \
			"merchant_id": "mrc_electronics_01", \
			"merchant_category": "electronics", \
			"payment_method": "credit_card", \
			"timestamp": "2024-01-15T03:22:00Z", \
			"card_present": false, \
			"location": {"country": "NG"} \
		}' | python -m json.tool
