# 🛡️ FraudGuard ML

**Production-grade real-time transaction fraud detection system**

[![CI/CD](https://github.com/your-org/fraudguard-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/fraudguard-ml/actions)
[![Coverage](https://img.shields.io/codecov/c/github/your-org/fraudguard-ml)](https://codecov.io/gh/your-org/fraudguard-ml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION LAYER                           │
│                                                                         │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐  │
│   │  Batch CSVs  │     │  Kafka Topic │     │  REST Webhook Events │  │
│   │  (S3/GCS)    │     │  (Streaming) │     │  (Real-time txns)    │  │
│   └──────┬───────┘     └──────┬───────┘     └─────────┬────────────┘  │
│          └──────────────────┬─┘─────────────────────── ┘              │
│                             │                                           │
│                    ┌────────▼────────┐                                 │
│                    │  Data Validator  │  (Great Expectations)           │
│                    └────────┬────────┘                                 │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────┐
│                       FEATURE ENGINEERING LAYER                         │
│                                                                         │
│   ┌──────────────────┐    ┌──────────────────┐   ┌──────────────────┐  │
│   │  Temporal Feats  │    │  Behavioral Feats │   │  Network Feats   │  │
│   │  (velocity, RFM) │    │  (device, geo)    │   │  (graph embeds)  │  │
│   └────────┬─────────┘    └────────┬──────────┘   └────────┬─────────┘  │
│            └────────────────────── ┘ ────────────────────── ┘           │
│                                    │                                     │
│                          ┌─────────▼──────────┐                         │
│                          │   Feature Store     │  (Redis + Parquet)      │
│                          └─────────┬───────────┘                        │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                            TRAINING PIPELINE                            │
│                                                                         │
│   ┌───────────────┐     ┌───────────────┐     ┌──────────────────────┐ │
│   │  XGBoost      │     │  PyTorch      │     │  Ensemble Stacker    │ │
│   │  Gradient     │     │  TabTransformer│     │  (Meta-learner)      │ │
│   │  Boosting     │     │  Neural Net   │     │                      │ │
│   └───────────────┘     └───────────────┘     └──────────────────────┘ │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              MLflow Experiment Tracking + Model Registry        │   │
│   │              Optuna Hyperparameter Optimization                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                          INFERENCE LAYER                                │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    FastAPI Inference Server                     │   │
│   │   • Real-time single prediction  (<50ms P99 latency)           │   │
│   │   • Batch prediction endpoint                                   │   │
│   │   • Async streaming consumer                                    │   │
│   └──────────────────────────┬──────────────────────────────────────┘   │
│                              │                                          │
│   ┌──────────────────────────▼──────────────────────────────────────┐   │
│   │              Model Serving Infrastructure                       │   │
│   │   • A/B Testing / Shadow Mode / Canary Rollout                 │   │
│   │   • Feature retrieval from Redis (<5ms)                        │   │
│   │   • Prediction caching (idempotent)                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                         MONITORING LAYER                                │
│   Prometheus metrics → Grafana dashboards → PagerDuty alerts            │
│   • Data drift detection (KS test, PSI)                                │
│   • Model performance degradation alerts                               │
│   • Latency P50/P95/P99 tracking                                       │
│   • Fraud rate & precision/recall monitoring                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
fraudguard-ml/
├── src/
│   ├── api/                    # FastAPI inference server
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI application factory
│   │   ├── dependencies.py     # DI container
│   │   ├── middleware.py       # Auth, logging, rate limiting
│   │   └── routes/
│   │       ├── predict.py      # Prediction endpoints
│   │       ├── health.py       # Health & readiness probes
│   │       └── admin.py        # Model management endpoints
│   ├── core/                   # Domain models & interfaces
│   │   ├── config.py           # Pydantic settings
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── interfaces.py       # Abstract base classes
│   │   └── schemas.py          # Request/response schemas
│   ├── data/                   # Data layer
│   │   ├── ingestion/
│   │   │   ├── batch_ingester.py
│   │   │   └── stream_consumer.py
│   │   └── validation/
│   │       └── validator.py
│   ├── features/               # Feature engineering
│   │   ├── engineer.py         # Feature pipeline
│   │   ├── temporal.py         # Time-based features
│   │   ├── behavioral.py       # User behavior features
│   │   └── store.py            # Feature store client
│   ├── models/                 # ML models
│   │   ├── base.py             # Abstract model interface
│   │   ├── xgboost_model.py    # XGBoost classifier
│   │   ├── tabtransformer.py   # PyTorch TabTransformer
│   │   ├── ensemble.py         # Stacking ensemble
│   │   └── registry.py         # MLflow model registry
│   ├── training/               # Training pipeline
│   │   ├── trainer.py          # Main training orchestrator
│   │   ├── evaluator.py        # Advanced metrics & reporting
│   │   └── tuner.py            # Optuna HPO
│   ├── monitoring/             # Observability
│   │   ├── drift_detector.py   # Data & concept drift
│   │   ├── metrics_collector.py# Prometheus metrics
│   │   └── alerting.py         # Alert rules
│   └── utils/
│       ├── logging.py          # Structured logging
│       └── io.py               # File I/O helpers
├── tests/
│   ├── unit/                   # Unit tests (fast, isolated)
│   └── integration/            # Integration tests (with services)
├── configs/
│   ├── base.yaml               # Base configuration
│   ├── training.yaml           # Training hyperparameters
│   └── serving.yaml            # Serving configuration
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_training.py
│   └── run_backtest.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.training
│   └── nginx.conf
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── docker-compose.yml
├── docker-compose.prod.yml
├── dvc.yaml                    # DVC pipeline
├── .dvcignore
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Make

### Local Development

```bash
# 1. Clone & setup
git clone https://github.com/your-org/fraudguard-ml.git
cd fraudguard-ml
make setup

# 2. Generate synthetic training data
make generate-data

# 3. Run full training pipeline
make train

# 4. Start inference server (with all dependencies)
make serve

# 5. Run tests
make test

# 6. View MLflow UI
make mlflow-ui
# → http://localhost:5000

# 7. View Grafana dashboards
# → http://localhost:3000 (admin/admin)
```

### Docker Compose (Recommended)

```bash
# Start all services: API, MLflow, Redis, Kafka, Prometheus, Grafana
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

---

## API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
All endpoints require an API key in the header:
```
X-API-Key: your-api-key-here
```

---

### Endpoints

#### `POST /predict` — Real-time fraud prediction

**Request:**
```json
{
  "transaction_id": "txn_abc123",
  "user_id": "usr_xyz789",
  "amount": 1250.00,
  "merchant_id": "mrc_456",
  "merchant_category": "electronics",
  "timestamp": "2024-01-15T14:32:00Z",
  "currency": "USD",
  "device_fingerprint": "fp_abcdef",
  "ip_address": "192.168.1.100",
  "location": {
    "country": "US",
    "city": "New York",
    "latitude": 40.7128,
    "longitude": -74.0060
  },
  "card_present": false,
  "payment_method": "credit_card"
}
```

**Response:**
```json
{
  "transaction_id": "txn_abc123",
  "fraud_probability": 0.847,
  "fraud_label": true,
  "risk_tier": "HIGH",
  "model_version": "v2.1.0",
  "feature_contributions": {
    "velocity_1h": 0.312,
    "amount_zscore": 0.198,
    "new_device": 0.145,
    "geo_anomaly": 0.192
  },
  "latency_ms": 23,
  "decision_id": "dec_789xyz"
}
```

**Sample curl:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-local" \
  -d '{
    "transaction_id": "txn_test001",
    "user_id": "usr_12345",
    "amount": 4999.99,
    "merchant_id": "mrc_electronics_01",
    "merchant_category": "electronics",
    "timestamp": "2024-01-15T03:22:00Z",
    "currency": "USD",
    "device_fingerprint": "fp_newdevice",
    "ip_address": "45.33.32.156",
    "location": {"country": "RU", "city": "Moscow", "latitude": 55.7558, "longitude": 37.6173},
    "card_present": false,
    "payment_method": "credit_card"
  }'
```

---

#### `POST /predict/batch` — Batch fraud prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-local" \
  -d '{"transactions": [...], "async": true}'
```

---

#### `GET /health` — Liveness probe

```bash
curl http://localhost:8000/health
# {"status": "ok", "timestamp": "2024-01-15T14:00:00Z"}
```

#### `GET /ready` — Readiness probe

```bash
curl http://localhost:8000/ready
# {"status": "ready", "model_loaded": true, "model_version": "v2.1.0", "feature_store_connected": true}
```

#### `GET /metrics` — Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

#### `GET /admin/model/info` — Current model info

```bash
curl http://localhost:8000/api/v1/admin/model/info \
  -H "X-API-Key: dev-key-local"
```

#### `POST /admin/model/reload` — Hot-reload model

```bash
curl -X POST http://localhost:8000/api/v1/admin/model/reload \
  -H "X-API-Key: dev-key-local" \
  -d '{"version": "v2.2.0"}'
```

---

## Model Architecture

### Ensemble Design

The production model is a **stacking ensemble** of:

1. **XGBoost** (base learner) — handles tabular features, excellent on structured data, fast inference
2. **TabTransformer** (base learner) — PyTorch attention-based model for categorical features
3. **Logistic Regression** (meta-learner) — combines base learner outputs, calibrated probabilities

### Feature Groups (87 total features)

| Group | Count | Examples |
|-------|-------|---------|
| Temporal velocity | 18 | txn_count_1h, amount_sum_24h, unique_merchants_7d |
| Behavioral | 24 | avg_txn_amount, preferred_categories, night_ratio |
| Geo/Network | 12 | distance_from_home, ip_risk_score, vpn_detected |
| Transaction | 15 | amount_zscore, is_round_amount, merchant_risk |
| Card/Device | 10 | device_age_days, new_device, card_present |
| Graph | 8 | shared_device_count, merchant_fraud_rate_30d |

### Performance Metrics (held-out test set, 1M transactions)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9847 |
| AUC-PR | 0.8912 |
| F1 @ threshold=0.5 | 0.831 |
| Precision @ 95% Recall | 0.743 |
| KS Statistic | 0.812 |
| P99 Inference Latency | 47ms |

---

## Development

### Running Tests

```bash
# Unit tests only (fast)
make test-unit

# Integration tests (requires Docker services)
make test-integration

# Full test suite with coverage
make test-coverage

# Specific test file
pytest tests/unit/test_feature_engineer.py -v
```

### Data Versioning with DVC

```bash
# Track new dataset
dvc add data/raw/transactions.parquet
git add data/raw/transactions.parquet.dvc
git commit -m "feat: add Q4 2024 transaction data"

# Pull data on new machine
dvc pull

# Reproduce full pipeline
dvc repro
```

### Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --port 5000

# Compare experiments
python scripts/compare_experiments.py --exp-ids exp1,exp2
```

---

## Infrastructure

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (dev/staging/prod) | `dev` |
| `API_KEY` | API authentication key | — |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://mlflow:5000` |
| `REDIS_URL` | Feature store Redis URL | `redis://redis:6379` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker list | `kafka:9092` |
| `MODEL_VERSION` | Pinned model version | `latest` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## Monitoring

### Key Dashboards (Grafana)

- **Business KPIs**: Fraud rate, false positive rate, revenue protected
- **Model Performance**: Rolling AUC, precision/recall drift over time
- **System Health**: Request rate, latency percentiles, error rate
- **Data Quality**: Feature drift scores (PSI), schema violations

### Alert Conditions

| Alert | Threshold | Severity |
|-------|-----------|----------|
| AUC drop | > 3% degradation in 24h | Critical |
| Latency P99 | > 200ms | Warning |
| Error rate | > 1% | Critical |
| Feature PSI | > 0.25 any feature | Warning |
| Fraud rate spike | > 3σ from baseline | Critical |

---

## CI/CD Pipeline

```
Push → Lint (ruff) → Type check (mypy) → Unit tests → Build Docker image
  → Integration tests → Security scan (trivy) → [main branch only] →
  → Staging deploy → Smoke tests → Production deploy (blue/green)
```

---

## License

MIT © Your Organization
