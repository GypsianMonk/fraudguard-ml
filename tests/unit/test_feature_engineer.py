"""
tests/unit/test_feature_engineer.py
------------------------------------
Unit tests for the feature engineering pipeline.
Fast, isolated — no external dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FraudFeatureEngineer


def make_sample_df(n: int = 100, fraud_rate: float = 0.05) -> pd.DataFrame:
    """Create a realistic synthetic transaction DataFrame for testing."""
    rng = np.random.default_rng(42)

    n_fraud = max(1, int(n * fraud_rate))
    n_legit = n - n_fraud
    labels = [1] * n_fraud + [0] * n_legit
    rng.shuffle(labels)

    base_time = datetime(2024, 1, 1, 12, 0, 0)
    timestamps = [base_time + timedelta(hours=i * 0.5) for i in range(n)]

    return pd.DataFrame({
        "transaction_id": [f"txn_{i:05d}" for i in range(n)],
        "user_id": [f"usr_{i % 20:03d}" for i in range(n)],
        "amount": rng.lognormal(mean=4.5, sigma=1.2, size=n).round(2),
        "currency": rng.choice(["USD", "EUR", "GBP"], size=n),
        "merchant_id": [f"mrc_{i % 15:03d}" for i in range(n)],
        "merchant_category": rng.choice(
            ["electronics", "groceries", "restaurants", "travel", "other"], size=n
        ),
        "payment_method": rng.choice(["credit_card", "debit_card", "digital_wallet"], size=n),
        "timestamp": timestamps,
        "card_present": rng.choice([True, False], size=n),
        "device_fingerprint": [f"fp_{i % 30:03d}" if rng.random() > 0.1 else None for i in range(n)],
        "country": rng.choice(["US", "CA", "GB", "DE", "RU"], size=n, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
        "latitude": rng.uniform(25.0, 50.0, size=n),
        "longitude": rng.uniform(-125.0, -65.0, size=n),
        "is_fraud": labels,
    })


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return make_sample_df(n=200)


@pytest.fixture
def fitted_engineer(sample_df: pd.DataFrame) -> FraudFeatureEngineer:
    """Return a fitted FraudFeatureEngineer."""
    engineer = FraudFeatureEngineer(mode="training")
    engineer.fit(sample_df)
    return engineer


class TestFraudFeatureEngineer:
    """Unit tests for FraudFeatureEngineer."""

    def test_fit_returns_self(self, sample_df: pd.DataFrame) -> None:
        """fit() should return self for method chaining."""
        engineer = FraudFeatureEngineer()
        result = engineer.fit(sample_df)
        assert result is engineer

    def test_is_fitted_after_fit(self, sample_df: pd.DataFrame) -> None:
        """is_fitted should be True after fit()."""
        engineer = FraudFeatureEngineer()
        assert not engineer.is_fitted
        engineer.fit(sample_df)
        assert engineer.is_fitted

    def test_transform_returns_dataframe(self, fitted_engineer: FraudFeatureEngineer, sample_df: pd.DataFrame) -> None:
        """transform() should return a pandas DataFrame."""
        result = fitted_engineer.transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_consistent_shape(self, fitted_engineer: FraudFeatureEngineer, sample_df: pd.DataFrame) -> None:
        """transform() should produce same number of rows as input."""
        result = fitted_engineer.transform(sample_df)
        assert len(result) == len(sample_df)

    def test_no_nan_in_output(self, fitted_engineer: FraudFeatureEngineer, sample_df: pd.DataFrame) -> None:
        """Feature matrix should have no NaN values."""
        result = fitted_engineer.transform(sample_df)
        assert not result.isna().any().any(), f"NaN found in columns: {result.columns[result.isna().any()].tolist()}"

    def test_target_not_in_features(self, fitted_engineer: FraudFeatureEngineer, sample_df: pd.DataFrame) -> None:
        """is_fraud target should not appear in feature matrix."""
        result = fitted_engineer.transform(sample_df)
        assert "is_fraud" not in result.columns

    def test_transaction_id_not_in_features(self, fitted_engineer: FraudFeatureEngineer, sample_df: pd.DataFrame) -> None:
        """Raw IDs should not appear in feature matrix."""
        result = fitted_engineer.transform(sample_df)
        assert "transaction_id" not in result.columns
        assert "user_id" not in result.columns

    def test_fit_transform_matches_fit_then_transform(self, sample_df: pd.DataFrame) -> None:
        """fit_transform() should produce identical result to fit() then transform()."""
        engineer1 = FraudFeatureEngineer()
        result1 = engineer1.fit_transform(sample_df)

        engineer2 = FraudFeatureEngineer()
        engineer2.fit(sample_df)
        result2 = engineer2.transform(sample_df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_transform_before_fit_raises(self, sample_df: pd.DataFrame) -> None:
        """transform() before fit() should raise RuntimeError."""
        engineer = FraudFeatureEngineer()
        with pytest.raises(RuntimeError, match="Pipeline must be fitted"):
            engineer.transform(sample_df)

    def test_missing_required_columns_raises(self, fitted_engineer: FraudFeatureEngineer) -> None:
        """transform() with missing columns should raise ValueError."""
        incomplete_df = pd.DataFrame({"transaction_id": ["txn_001"], "amount": [100.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            fitted_engineer.transform(incomplete_df)

    def test_empty_dataframe_raises(self, fitted_engineer: FraudFeatureEngineer, sample_df: pd.DataFrame) -> None:
        """transform() with empty DataFrame should raise ValueError."""
        empty_df = sample_df.head(0)
        with pytest.raises(ValueError, match="empty"):
            fitted_engineer.transform(empty_df)

    def test_feature_names_available_after_fit(self, fitted_engineer: FraudFeatureEngineer) -> None:
        """get_feature_names() should return non-empty list after fit."""
        names = fitted_engineer.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_feature_names_not_available_before_fit(self) -> None:
        """get_feature_names() before fit should raise RuntimeError."""
        engineer = FraudFeatureEngineer()
        with pytest.raises(RuntimeError):
            engineer.get_feature_names()

    def test_expected_feature_groups_present(self, fitted_engineer: FraudFeatureEngineer, sample_df: pd.DataFrame) -> None:
        """Result should contain features from all major groups."""
        result = fitted_engineer.transform(sample_df)
        feature_names = list(result.columns)

        # Temporal features
        assert any("txn_count" in f for f in feature_names), "Missing temporal velocity features"
        assert any("hour_of_day" in f for f in feature_names), "Missing hour_of_day feature"

        # Behavioral features
        assert any("amount_vs_user" in f for f in feature_names), "Missing behavioral features"

        # Transaction features
        assert any("amount_zscore" in f for f in feature_names), "Missing amount_zscore"

    def test_unseen_inference_data_handled(self, fitted_engineer: FraudFeatureEngineer) -> None:
        """Transform should handle unseen merchants and users gracefully."""
        new_df = make_sample_df(n=10)
        # Replace with entirely new user/merchant IDs
        new_df["user_id"] = [f"new_user_{i}" for i in range(10)]
        new_df["merchant_id"] = [f"new_merchant_{i}" for i in range(10)]

        # Should not raise — unseen entities handled via fallback
        result = fitted_engineer.transform(new_df)
        assert len(result) == 10
        assert not result.isna().any().any()

    def test_reproducible_with_same_seed(self, sample_df: pd.DataFrame) -> None:
        """Two engineers fit on same data should produce identical features."""
        e1 = FraudFeatureEngineer().fit(sample_df)
        e2 = FraudFeatureEngineer().fit(sample_df)

        r1 = e1.transform(sample_df)
        r2 = e2.transform(sample_df)

        pd.testing.assert_frame_equal(r1, r2)
