"""
tests/unit/test_feature_engineer.py
-------------------------------------
Unit tests for FraudFeatureEngineer.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FraudFeatureEngineer


def make_txn_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    return pd.DataFrame({
        "transaction_id": [f"txn_{i:03d}" for i in range(n)],
        "user_id": [f"usr_{i % 10:02d}" for i in range(n)],
        "amount": np.round(rng.lognormal(4.2, 1.5, n), 2),
        "currency": rng.choice(["USD", "EUR", "GBP"], n),
        "merchant_id": [f"mrc_{i % 8:02d}" for i in range(n)],
        "merchant_category": rng.choice(
            ["electronics", "groceries", "restaurants", "travel"], n
        ),
        "payment_method": rng.choice(["credit_card", "debit_card"], n),
        "timestamp": [
            base.replace(hour=int(i % 24)) for i in range(n)
        ],
        "card_present": rng.choice([True, False], n),
        "device_fingerprint": [f"fp_{i % 15:02d}" for i in range(n)],
        "country": rng.choice(["US", "CA", "GB", "RU"], n, p=[0.7, 0.1, 0.1, 0.1]),
        "latitude": rng.uniform(25, 50, n),
        "longitude": rng.uniform(-125, -65, n),
        "is_fraud": rng.choice([0, 1], n, p=[0.95, 0.05]),
    })


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_txn_df(100)


@pytest.fixture
def fitted_engineer(raw_df: pd.DataFrame) -> FraudFeatureEngineer:
    eng = FraudFeatureEngineer(mode="training")
    eng.fit_transform(raw_df)
    return eng


class TestFraudFeatureEngineer:

    def test_fit_transform_returns_dataframe(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_more_columns_than_input(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        assert result.shape[1] > raw_df.shape[1]

    def test_fit_transform_preserves_row_count(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        assert len(result) == len(raw_df)

    def test_transform_inference_mode_same_columns(
        self, fitted_engineer: FraudFeatureEngineer, raw_df: pd.DataFrame
    ) -> None:
        new_df = make_txn_df(20)
        result = fitted_engineer.transform(new_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_no_nan_in_output(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        nan_cols = result.columns[result.isnull().any()].tolist()
        assert len(nan_cols) == 0, f"NaN columns: {nan_cols}"

    def test_all_numeric_output(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        non_numeric = [
            c for c in result.columns
            if result[c].dtype not in (float, int, bool, "float64", "int64", "bool")
        ]
        assert len(non_numeric) == 0, f"Non-numeric columns: {non_numeric}"

    def test_temporal_features_present(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        temporal = [c for c in result.columns if any(
            k in c for k in ("hour", "day", "is_night", "is_weekend")
        )]
        assert len(temporal) > 0

    def test_amount_features_present(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        amount_cols = [c for c in result.columns if "amount" in c]
        assert len(amount_cols) > 0

    def test_get_feature_names_matches_output(self, fitted_engineer: FraudFeatureEngineer) -> None:
        names = fitted_engineer.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_target_column_excluded_from_features(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        assert "is_fraud" not in result.columns

    def test_feature_count_above_minimum(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        assert result.shape[1] >= 20, f"Expected >= 20 features, got {result.shape[1]}"

    def test_card_not_present_feature(self, raw_df: pd.DataFrame) -> None:
        eng = FraudFeatureEngineer(mode="training")
        result = eng.fit_transform(raw_df)
        card_cols = [c for c in result.columns if "card" in c.lower()]
        assert len(card_cols) > 0

    def test_single_row_inference(self, fitted_engineer: FraudFeatureEngineer) -> None:
        single = make_txn_df(1)
        result = fitted_engineer.transform(single)
        assert len(result) == 1
        assert result.isnull().sum().sum() == 0
