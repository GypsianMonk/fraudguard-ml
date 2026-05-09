<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a0000,50:7f1d1d,100:1a0000&height=220&section=header&text=FraudGuard+ML&fontSize=70&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Production-Grade%20Real-Time%20Transaction%20Fraud%20Detection%20System&descAlignY=60&descSize=15&descColor=fca5a5" width="100%"/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=19&pause=1000&color=FCA5A5&center=true&vCenter=true&width=850&lines=XGBoost+%2B+TabTransformer+Ensemble+%7C+87+Engineered+Features;AUC-ROC+0.9847+%7C+P99+Latency+47ms+%7C+1M+Transaction+Benchmark;Real-Time+%2B+Batch+%2B+Streaming+%7C+A%2FB+%2B+Shadow+%2B+Canary+%F0%9F%9A%80)](https://git.io/typing-svg)

<br/>

[![CI/CD](https://img.shields.io/badge/CI%2FCD-Passing-22c55e?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/GypsianMonk/fraudguard-ml/actions)
[![Coverage](https://img.shields.io/badge/Coverage-Tracked-0e9de0?style=for-the-badge&logo=codecov&logoColor=white)](https://codecov.io/gh/GypsianMonk/fraudguard-ml)
[![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference_Server-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## ◈ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION LAYER                            │
│                                                                          │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐  │
│   │  Batch CSVs  │      │  Kafka Topic │      │  REST Webhook Events │  │
│   │  (S3 / GCS)  │      │  (Streaming) │      │  (Real-time txns)    │  │
│   └──────┬───────┘      └──────┬───────┘      └──────────┬───────────┘  │
│          └───────────────────┬─┘──────────────────────────┘             │
│                              │                                           │
│                    ┌─────────▼─────────┐                                │
│                    │  Data Validator    │  (Great Expectations)          │
│                    └─────────┬─────────┘                                │
└──────────────────────────────┼───────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│                       FEATURE ENGINEERING LAYER                          │
│                                                                          │
│   ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────┐  │
│   │  Temporal Feats  │   │  Behavioral Feats  │   │  Network Feats   │  │
│   │  velocity · RFM  │   │  device · geo      │   │  graph embeds    │  │
│   └────────┬─────────┘   └─────────┬──────────┘   └────────┬─────────┘  │
│            └─────────────────────┬─┘──────────────────────┘            │
│                                  │                                       │
│                        ┌─────────▼──────────┐                           │
│                        │    Feature Store    │  (Redis + Parquet)        │
│                        └─────────┬───────────┘                          │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                            TRAINING PIPELINE                             │
│                                                                          │
│   ┌───────────────┐      ┌──────────────────┐    ┌────────────────────┐ │
│   │   XGBoost     │      │  TabTransformer   │    │  Ensemble Stacker  │ │
│   │   Gradient    │      │  (PyTorch Attn)   │    │  (Meta-learner)    │ │
│   │   Boosting    │      │  Neural Network   │    │  LogReg calibrated │ │
│   └───────────────┘      └──────────────────┘    └────────────────────┘ │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │        MLflow Experiment Tracking + Model Registry               │  │
│   │        Optuna Hyperparameter Optimization                        │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                           INFERENCE LAYER                                │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                  FastAPI Inference Server                        │  │
│   │   • Real-time single prediction  (< 50ms P99 latency)           │  │
│   │   • Batch prediction endpoint                                    │  │
│   │   • Async streaming consumer                                     │  │
│   └───────────────────────────┬──────────────────────────────────────┘  │
│                               │                                          │
│   ┌───────────────────────────▼──────────────────────────────────────┐  │
│   │               Model Serving Infrastructure                       │  │
│   │   • A/B Testing · Shadow Mode · Canary Rollout                  │  │
│   │   • Feature retrieval from Redis (< 5ms)                        │  │
│   │   • Prediction caching (idempotent)                             │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                          MONITORING LAYER                                │
│   Prometheus metrics → Grafana dashboards → PagerDuty alerts             │
│   • Data drift detection (KS test, PSI)                                  │
│   • Model performance degradation alerts                                 │
│   • Latency P50 / P95 / P99 tracking                                     │
│   • Fraud rate & precision / recall monitoring                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## ◈ Model Performance

### Stacking Ensemble — 1M Transaction Benchmark

<div align="center">

| Metric | Value |
|:---:|:---:|
| AUC-ROC | **0.9847** |
| AUC-PR | **0.8912** |
| F1 @ threshold=0.5 | **0.831** |
| Precision @ 95% Recall | **0.743** |
| KS Statistic | **0.812** |
| P99 Inference Latency | **47ms** |

</div>

### Ensemble Design

The production model is a **3-layer stacking ensemble:**

```
Base Layer 1:  XGBoost          → tabular features, fast inference
Base Layer 2:  TabTransformer   → PyTorch attention for categorical features
Meta-Learner:  Logistic Reg     → calibrated probability combination
```

### Feature Groups — 87 Total Features

<div align="center">

| Group | Count | Examples |
|:---:|:---:|:---|
| ⏱️ Temporal Velocity | 18 | `txn_count_1h` · `amount_sum_24h` · `unique_merchants_7d` |
| 🧠 Behavioral | 24 | `avg_txn_amount` · `preferred_categories` · `night_ratio` |
| 🌍 Geo / Network | 12 | `distance_from_home` · `ip_risk_score` · `vpn_detected` |
| 💳 Transaction | 15 | `amount_zscore` · `is_round_amount` · `merchant_risk` |
| 📱 Card / Device | 10 | `device_age_days` · `new_device` · `card_present` |
| 🕸️ Graph | 8 | `shared_device_count` · `merchant_fraud_rate_30d` |

</div>

---

## ◈ Tech Stack

<div align="center">

### ⟡ ML & Training
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-TabTransformer-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Meta_Learner-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-6C4EFF?style=for-the-badge)](https://optuna.org/)

### ⟡ MLOps & Tracking
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-945DD5?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)

### ⟡ Inference & API
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-Feature_Store-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-Streaming-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)](https://kafka.apache.org/)

### ⟡ Infrastructure & Monitoring
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com/)
[![Great Expectations](https://img.shields.io/badge/Great_Expectations-Data_Validation-FF6B6B?style=for-the-badge)](https://greatexpectations.io/)

### ⟡ Testing & Quality
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/Ruff-Linter-D7FF64?style=for-the-badge)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/mypy-Type_Checker-2A6DB2?style=for-the-badge)](https://mypy.readthedocs.io/)

</div>

---

## ◈ Project Structure

```
fraudguard-ml/
│
├── 🌐 src/
│   ├── api/                       ← FastAPI inference server
│   │   ├── app.py                 ← Application factory
│   │   ├── dependencies.py        ← DI container
│   │   ├── middleware.py          ← Auth · logging · rate limiting
│   │   └── routes/
│   │       ├── predict.py         ← Prediction endpoints
│   │       ├── health.py          ← Liveness & readiness probes
│   │       └── admin.py           ← Model management endpoints
│   │
│   ├── core/                      ← Domain models & interfaces
│   │   ├── config.py              ← Pydantic settings
│   │   ├── exceptions.py          ← Custom exceptions
│   │   ├── interfaces.py          ← Abstract base classes
│   │   └── schemas.py             ← Request / response schemas
│   │
│   ├── data/                      ← Data layer
│   │   ├── ingestion/
│   │   │   ├── batch_ingester.py
│   │   │   └── stream_consumer.py
│   │   └── validation/
│   │       └── validator.py
│   │
│   ├── features/                  ← Feature engineering
│   │   ├── engineer.py            ← Feature pipeline
│   │   ├── temporal.py            ← Time-based features
│   │   ├── behavioral.py          ← User behavior features
│   │   └── store.py               ← Feature store client
│   │
│   ├── models/                    ← ML models
│   │   ├── base.py                ← Abstract model interface
│   │   ├── xgboost_model.py       ← XGBoost classifier
│   │   ├── tabtransformer.py      ← PyTorch TabTransformer
│   │   ├── ensemble.py            ← Stacking ensemble
│   │   └── registry.py            ← MLflow model registry
│   │
│   ├── training/                  ← Training pipeline
│   │   ├── trainer.py             ← Main training orchestrator
│   │   ├── evaluator.py           ← Advanced metrics & reporting
│   │   └── tuner.py               ← Optuna HPO
│   │
│   └── monitoring/                ← Observability
│       ├── drift_detector.py      ← Data & concept drift
│       ├── metrics_collector.py   ← Prometheus metrics
│       └── alerting.py            ← Alert rules
│
├── 🧪 tests/
│   ├── unit/                      ← Fast, isolated unit tests
│   └── integration/               ← Integration tests (with services)
│
├── ⚙️  configs/
│   ├── base.yaml
│   ├── training.yaml
│   └── serving.yaml
│
├── 🐳 docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.training
│   └── nginx.conf
│
├── docker-compose.yml
├── docker-compose.prod.yml
├── dvc.yaml                       ← DVC pipeline definition
├── pyproject.toml
├── Makefile
└── README.md
```

---

## ◈ Quick Start

### Local Development

```bash
# 1. Clone & setup
git clone https://github.com/GypsianMonk/fraudguard-ml.git
cd fraudguard-ml
make setup

# 2. Generate synthetic training data
make generate-data

# 3. Run full training pipeline
make train

# 4. Start inference server
make serve

# 5. Run tests
make test

# 6. MLflow UI → http://localhost:5000
make mlflow-ui
```

### Docker Compose (Recommended)

```bash
# Start all services: API, MLflow, Redis, Kafka, Prometheus, Grafana
docker-compose up -d

# Health check
curl http://localhost:8000/health

# Tail logs
docker-compose logs -f api
```

---

## ◈ API Reference

**Base URL:** `http://localhost:8000/api/v1`  
**Auth:** `X-API-Key: your-api-key-here` on all endpoints.

<div align="center">

| Method | Endpoint | Description |
|:---:|:---|:---|
| `POST` | `/predict` | Real-time single fraud prediction |
| `POST` | `/predict/batch` | Async batch fraud prediction |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe + model status |
| `GET` | `/metrics` | Prometheus metrics scrape |
| `GET` | `/admin/model/info` | Current model version info |
| `POST` | `/admin/model/reload` | Hot-reload model version |

</div>

<details>
<summary><b>POST /predict — Request & Response</b></summary>

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

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-local" \
  -d '{
    "transaction_id": "txn_test001",
    "user_id": "usr_12345",
    "amount": 4999.99,
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

</details>

---

## ◈ Infrastructure & Configuration

### Environment Variables

<div align="center">

| Variable | Description | Default |
|:---:|:---|:---:|
| `ENV` | Environment (`dev` / `staging` / `prod`) | `dev` |
| `API_KEY` | API authentication key | — |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://mlflow:5000` |
| `REDIS_URL` | Feature store Redis URL | `redis://redis:6379` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker list | `kafka:9092` |
| `MODEL_VERSION` | Pinned model version | `latest` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

</div>

---

## ◈ Monitoring

### Grafana Dashboards

```
📊  Business KPIs       →  Fraud rate · false positive rate · revenue protected
🤖  Model Performance   →  Rolling AUC · precision/recall drift over time
⚙️  System Health       →  Request rate · latency percentiles · error rate
🔬  Data Quality        →  Feature drift scores (PSI) · schema violations
```

### Alert Conditions

<div align="center">

| Alert | Threshold | Severity |
|:---:|:---|:---:|
| AUC Drop | > 3% degradation in 24h | 🔴 Critical |
| Latency P99 | > 200ms | 🟡 Warning |
| Error Rate | > 1% | 🔴 Critical |
| Feature PSI | > 0.25 any feature | 🟡 Warning |
| Fraud Rate Spike | > 3σ from baseline | 🔴 Critical |

</div>

---

## ◈ CI/CD Pipeline

```
Push
  → Lint (ruff) → Type Check (mypy) → Unit Tests → Build Docker Image
  → Integration Tests → Security Scan (trivy)
  → [main only] → Staging Deploy → Smoke Tests → Production Deploy (blue/green)
```

---

## ◈ Data Versioning with DVC

```bash
# Track new dataset
dvc add data/raw/transactions.parquet
git add data/raw/transactions.parquet.dvc
git commit -m "feat: add Q4 2024 transaction data"

# Pull data on a new machine
dvc pull

# Reproduce full pipeline
dvc repro
```

---

## ◈ Testing

```bash
# Unit tests only (fast, no services needed)
make test-unit

# Integration tests (requires Docker)
make test-integration

# Full suite with coverage report
make test-coverage

# Single file
pytest tests/unit/test_feature_engineer.py -v
```

---

## ◈ License

Licensed under the **[MIT License](LICENSE)** — use it, fork it, build on it.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a0000,50:7f1d1d,100:1a0000&height=120&section=footer" width="100%"/>

*"Fraud doesn't sleep. Neither does FraudGuard."*

**Built with ❤️ by [GypsianMonk](https://github.com/GypsianMonk)**

</div>
