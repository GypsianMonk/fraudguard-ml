"""
src/features/behavioral.py
--------------------------
User behavioral profile features.
Captures long-term spending patterns to detect anomalies vs. user's own baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BehavioralFeatureExtractor:
    """
    Extracts behavioral profile features by comparing each transaction
    against the user's historical baseline.

    Features:
    - amount_vs_user_avg: ratio of txn amount to user's avg
    - is_new_category: first time user transacts in this category
    - category_frequency: how often user transacts in this category
    - night_txn_ratio: fraction of user's txns that are night transactions
    - intl_txn_ratio: fraction of user's txns that are international
    - avg_daily_spend: user's average daily spend over history
    - spend_percentile: where this txn falls in user's historical amount distribution
    - preferred_payment_match: does payment method match user's usual method
    """

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute behavioral features.

        Args:
            df: Transaction DataFrame with [transaction_id, user_id, amount,
                timestamp, merchant_category] columns.

        Returns:
            DataFrame with transaction_id + behavioral feature columns.
        """
        logger.debug("Extracting behavioral features")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["is_night"] = ((df["hour"] >= 23) | (df["hour"] < 5)).astype(int)

        features = pd.DataFrame({"transaction_id": df["transaction_id"].values})

        # User-level aggregations
        user_stats = (
            df.groupby("user_id")
            .agg(
                user_avg_amount=("amount", "mean"),
                user_std_amount=("amount", "std"),
                user_median_amount=("amount", "median"),
                user_txn_count=("transaction_id", "count"),
                user_night_ratio=("is_night", "mean"),
                user_total_spend=("amount", "sum"),
            )
            .reset_index()
        )
        user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0)

        # Merge user stats back
        merged = df.merge(user_stats, on="user_id", how="left")

        # Amount relative to user's baseline
        features["amount_vs_user_avg"] = (
            merged["amount"] / (merged["user_avg_amount"] + 1e-8)
        ).values

        features["amount_vs_user_median"] = (
            merged["amount"] / (merged["user_median_amount"] + 1e-8)
        ).values

        features["user_night_txn_ratio"] = merged["user_night_ratio"].fillna(0).values
        features["user_total_spend"] = merged["user_total_spend"].values
        features["user_txn_count"] = merged["user_txn_count"].values

        # Category behavior
        if "merchant_category" in df.columns:
            cat_user_counts = (
                df.groupby(["user_id", "merchant_category"])
                .size()
                .reset_index(name="user_category_count")
            )
            merged2 = df.merge(cat_user_counts, on=["user_id", "merchant_category"], how="left")
            merged2 = merged2.merge(user_stats[["user_id", "user_txn_count"]], on="user_id", how="left")
            features["category_frequency_ratio"] = (
                merged2["user_category_count"] / (merged2["user_txn_count"] + 1e-8)
            ).values
            features["is_new_category"] = (merged2["user_category_count"] == 1).astype(int).values
        else:
            features["category_frequency_ratio"] = 0.5
            features["is_new_category"] = 0

        # Spend percentile within user's own history
        features["spend_percentile"] = self._compute_spend_percentile(df)

        # Average daily spend
        df["date"] = df["timestamp"].dt.date
        daily_spend = df.groupby(["user_id", "date"])["amount"].sum().reset_index()
        avg_daily = daily_spend.groupby("user_id")["amount"].mean().reset_index()
        avg_daily.columns = ["user_id", "avg_daily_spend"]
        merged3 = df.merge(avg_daily, on="user_id", how="left")
        features["avg_daily_spend"] = merged3["avg_daily_spend"].fillna(0).values

        return features

    def _compute_spend_percentile(self, df: pd.DataFrame) -> np.ndarray:
        """Compute each transaction's percentile rank within user's amount distribution."""
        result = np.full(len(df), 0.5)

        for user_id, group in df.groupby("user_id"):
            idx = group.index
            amounts = group["amount"].values
            if len(amounts) > 1:
                # Use expanding window percentile rank
                percentiles = []
                for i, amt in enumerate(amounts):
                    hist = amounts[: i + 1]
                    pct = np.sum(hist <= amt) / len(hist)
                    percentiles.append(pct)
                result[idx] = percentiles

        return result
