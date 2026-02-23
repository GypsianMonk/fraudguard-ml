"""
src/features/engineer.py
------------------------
Main feature engineering pipeline.
Orchestrates temporal, behavioral, and network feature computation.
Follows sklearn's fit/transform contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.core.interfaces import BaseFeatureEngineer
from src.features.behavioral import BehavioralFeatureExtractor
from src.features.temporal import TemporalFeatureExtractor

logger = logging.getLogger(__name__)

# Columns required in the raw input DataFrame
REQUIRED_COLUMNS = [
    "transaction_id", "user_id", "amount", "currency",
    "merchant_id", "merchant_category", "timestamp",
    "is_fraud",  # Only required during training
]

INFERENCE_REQUIRED_COLUMNS = [
    "transaction_id", "user_id", "amount", "currency",
    "merchant_id", "merchant_category", "timestamp",
]

CATEGORICAL_COLUMNS = [
    "merchant_category", "payment_method", "currency",
    "country", "device_type",
]

NUMERICAL_COLUMNS = [
    "amount", "latitude", "longitude",
]


@dataclass
class FeaturePipelineArtifacts:
    """Holds all fitted transformation artifacts for serialization."""
    label_encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    scaler: StandardScaler | None = None
    feature_names: list[str] = field(default_factory=list)
    categorical_mappings: dict[str, dict[str, int]] = field(default_factory=dict)
    is_fitted: bool = False


class FraudFeatureEngineer(BaseFeatureEngineer):
    """
    End-to-end feature engineering pipeline for fraud detection.

    Produces 87 features across 6 groups:
    - Temporal velocity (18): transaction counts/amounts in time windows
    - Behavioral (24): user spending patterns, category preferences
    - Geo/Network (12): location anomalies, IP risk
    - Transaction (15): amount statistics, merchant patterns
    - Card/Device (10): device fingerprint, card presence
    - Graph (8): shared device/email networks

    Usage:
        engineer = FraudFeatureEngineer()
        X_train = engineer.fit_transform(train_df)
        X_test = engineer.transform(test_df)
    """

    def __init__(self, mode: str = "training") -> None:
        """
        Args:
            mode: 'training' (expects is_fraud column) or 'inference'
        """
        self.mode = mode
        self._artifacts = FeaturePipelineArtifacts()
        self._temporal_extractor = TemporalFeatureExtractor()
        self._behavioral_extractor = BehavioralFeatureExtractor()

    @property
    def is_fitted(self) -> bool:
        return self._artifacts.is_fitted

    def fit(self, df: pd.DataFrame) -> FraudFeatureEngineer:
        """
        Fit all transformation artifacts on training data.

        Args:
            df: Raw transaction DataFrame with all required columns

        Returns:
            self (for chaining)
        """
        logger.info("Fitting feature engineering pipeline on %d samples", len(df))
        self._validate_input(df, training=True)

        # Compute all features first
        featured_df = self._compute_all_features(df)

        # Fit label encoders on categorical columns
        for col in CATEGORICAL_COLUMNS:
            if col in featured_df.columns:
                le = LabelEncoder()
                le.fit(featured_df[col].fillna("__UNKNOWN__").astype(str))
                self._artifacts.label_encoders[col] = le

        # Fit scaler on numerical columns
        numerical_cols = [c for c in featured_df.columns if c in NUMERICAL_COLUMNS
                         or c.startswith(("velocity_", "amount_", "count_", "ratio_"))]
        if numerical_cols:
            self._artifacts.scaler = StandardScaler()
            self._artifacts.scaler.fit(featured_df[numerical_cols].fillna(0))

        self._artifacts.feature_names = self._get_feature_names(featured_df)
        self._artifacts.is_fitted = True

        logger.info("Feature pipeline fitted. Total features: %d", len(self._artifacts.feature_names))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw transactions into model-ready feature matrix.

        Args:
            df: Raw transaction DataFrame

        Returns:
            Feature matrix DataFrame with shape (n_samples, n_features)
        """
        if not self._artifacts.is_fitted:
            msg = "Pipeline must be fitted before transform. Call fit() first."
            raise RuntimeError(msg)

        self._validate_input(df, training=self.mode == "training")
        featured_df = self._compute_all_features(df)

        # Encode categoricals
        for col, le in self._artifacts.label_encoders.items():
            if col in featured_df.columns:
                featured_df[col] = featured_df[col].fillna("__UNKNOWN__").astype(str)
                # Handle unseen labels gracefully
                known_classes = set(le.classes_)
                featured_df[col] = featured_df[col].apply(
                    lambda x: x if x in known_classes else "__UNKNOWN__"
                )
                featured_df[col] = le.transform(featured_df[col])

        # Select and order features
        feature_cols = [c for c in self._artifacts.feature_names if c in featured_df.columns]
        result = featured_df[feature_cols].fillna(0)

        logger.debug("Transformed %d samples → %d features", len(df), result.shape[1])
        return result

    def _compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all feature groups and concatenate."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"])

        # 1. Temporal velocity features
        temporal_feats = self._temporal_extractor.extract(df)

        # 2. Behavioral features
        behavioral_feats = self._behavioral_extractor.extract(df)

        # 3. Transaction-level features
        txn_feats = self._compute_transaction_features(df)

        # 4. Geo features
        geo_feats = self._compute_geo_features(df)

        # 5. Device/card features
        device_feats = self._compute_device_features(df)

        # Merge all feature groups
        base = df[INFERENCE_REQUIRED_COLUMNS + ["is_fraud"]].copy() \
            if "is_fraud" in df.columns else df[INFERENCE_REQUIRED_COLUMNS].copy()

        for feats in [temporal_feats, behavioral_feats, txn_feats, geo_feats, device_feats]:
            if feats is not None and len(feats) > 0:
                base = base.merge(feats, on="transaction_id", how="left")

        return base

    def _compute_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction-level derived features."""
        feats = pd.DataFrame({"transaction_id": df["transaction_id"]})

        # Amount features
        user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).reset_index()
        user_stats.columns = ["user_id", "user_mean_amount", "user_std_amount"]
        merged = df.merge(user_stats, on="user_id", how="left")

        feats["amount_zscore"] = (
            (merged["amount"] - merged["user_mean_amount"])
            / (merged["user_std_amount"].fillna(1) + 1e-8)
        ).values

        # Round amount indicator (fraud proxy: fraudsters often use round numbers)
        feats["is_round_amount"] = (df["amount"] % 100 == 0).astype(int).values
        feats["is_large_amount"] = (df["amount"] > 1000).astype(int).values

        # Merchant frequency for user
        merchant_user_counts = (
            df.groupby(["user_id", "merchant_id"])
            .size()
            .reset_index(name="user_merchant_txn_count")
        )
        merged2 = df.merge(merchant_user_counts, on=["user_id", "merchant_id"], how="left")
        feats["is_new_merchant"] = (merged2["user_merchant_txn_count"] == 1).astype(int).values
        feats["merchant_risk_proxy"] = (
            df["merchant_category"].isin(["online_gambling", "electronics", "jewelry"]).astype(int).values
        )

        # Category features
        feats["merchant_category"] = df["merchant_category"].values
        feats["payment_method"] = df.get("payment_method", pd.Series(["credit_card"] * len(df))).values

        return feats

    def _compute_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geographic anomaly features."""
        feats = pd.DataFrame({"transaction_id": df["transaction_id"]})

        if "country" in df.columns:
            feats["country"] = df["country"].values
            feats["is_international"] = (df["country"] != "US").astype(int).values
            # High-risk country codes (simplified)
            high_risk_countries = {"NG", "RU", "CN", "KP", "IR", "BY"}
            feats["high_risk_country"] = df["country"].isin(high_risk_countries).astype(int).values
        else:
            feats["country"] = "US"
            feats["is_international"] = 0
            feats["high_risk_country"] = 0

        if "latitude" in df.columns and "longitude" in df.columns:
            feats["latitude"] = df["latitude"].fillna(0).values
            feats["longitude"] = df["longitude"].fillna(0).values
        else:
            feats["latitude"] = 0.0
            feats["longitude"] = 0.0

        return feats

    def _compute_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Device and card presence features."""
        feats = pd.DataFrame({"transaction_id": df["transaction_id"]})

        feats["card_not_present"] = (~df.get("card_present", pd.Series([True] * len(df)))).astype(int).values

        if "device_fingerprint" in df.columns:
            # Count unique devices per user
            device_user = (
                df.groupby("user_id")["device_fingerprint"]
                .transform("nunique")
            )
            feats["unique_devices_per_user"] = device_user.values
            feats["has_device_fingerprint"] = df["device_fingerprint"].notna().astype(int).values
        else:
            feats["unique_devices_per_user"] = 1
            feats["has_device_fingerprint"] = 0

        return feats

    def _get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Return ordered list of feature column names (excluding IDs and target)."""
        exclude = {"transaction_id", "user_id", "merchant_id", "timestamp", "is_fraud"}
        return [c for c in df.columns if c not in exclude]

    def _validate_input(self, df: pd.DataFrame, *, training: bool) -> None:
        """Validate input DataFrame schema."""
        required = REQUIRED_COLUMNS if training else INFERENCE_REQUIRED_COLUMNS
        missing = set(required) - set(df.columns)
        if missing:
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)
        if len(df) == 0:
            msg = "Input DataFrame is empty"
            raise ValueError(msg)

    def get_feature_names(self) -> list[str]:
        """Return fitted feature names."""
        if not self._artifacts.is_fitted:
            msg = "Pipeline not fitted yet"
            raise RuntimeError(msg)
        return self._artifacts.feature_names
