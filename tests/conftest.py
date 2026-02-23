"""
tests/conftest.py
-----------------
Shared pytest fixtures and configuration.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("ENV", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("API_KEYS", "dev-key-local,test-key")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


@pytest.fixture(scope="session")
def sample_transactions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    n_fraud = 25
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time.replace(hour=int((12 + i * 0.3) % 24)) for i in range(n)]
    labels = [1] * n_fraud + [0] * (n - n_fraud)
    rng.shuffle(labels)
    return pd.DataFrame({
        "transaction_id": [f"txn_{i:06d}" for i in range(n)],
        "user_id": [f"usr_{i % 30:03d}" for i in range(n)],
        "amount": np.round(rng.lognormal(4.2, 1.4, n), 2),
        "currency": rng.choice(["USD", "EUR", "GBP"], n),
        "merchant_id": [f"mrc_{i % 20:03d}" for i in range(n)],
        "merchant_category": rng.choice(
            ["electronics", "groceries", "restaurants", "travel", "other"], n
        ),
        "payment_method": rng.choice(
            ["credit_card", "debit_card", "digital_wallet"], n
        ),
        "timestamp": timestamps,
        "card_present": rng.choice([True, False], n),
        "device_fingerprint": [f"fp_{i % 40:03d}" for i in range(n)],
        "country": rng.choice(["US", "CA", "GB", "RU"], n, p=[0.75, 0.1, 0.1, 0.05]),
        "latitude": rng.uniform(25.0, 50.0, n),
        "longitude": rng.uniform(-125.0, -65.0, n),
        "is_fraud": labels,
    })


@pytest.fixture(scope="session")
def feature_matrix() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 300
    n_fraud = 15
    feature_names = [
        "amount_zscore", "txn_count_1h", "txn_count_24h",
        "amount_sum_1h", "is_night", "is_weekend", "hour_of_day",
        "card_not_present", "is_international", "is_new_merchant",
    ]
    fraud_x = rng.standard_normal((n_fraud, len(feature_names))) + 2.0
    legit_x = rng.standard_normal((n - n_fraud, len(feature_names)))
    features = pd.DataFrame(np.vstack([fraud_x, legit_x]), columns=feature_names)
    labels = pd.Series([1] * n_fraud + [0] * (n - n_fraud), name="is_fraud")
    shuffle = rng.permutation(n)
    return features.iloc[shuffle].reset_index(drop=True), labels.iloc[shuffle].reset_index(drop=True)


@pytest.fixture(autouse=True)
def reset_settings_cache():
    from src.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_redis():
    with patch("redis.asyncio.from_url") as mock:
        mock_client = MagicMock()
        mock_client.ping = MagicMock(return_value=True)
        mock_client.get = MagicMock(return_value=None)
        mock_client.set = MagicMock(return_value=True)
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_mlflow():
    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run") as mock_run,
        patch("mlflow.log_param"),
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_artifacts"),
        patch("mlflow.log_artifact"),
        patch("mlflow.register_model"),
    ):
        mock_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_run.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_run
