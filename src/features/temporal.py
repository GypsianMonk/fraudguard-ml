"""
src/features/temporal.py
------------------------
Time-based velocity and recency features.
Computes transaction counts, amounts, and unique entities
within rolling time windows per user.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Rolling window sizes in hours
VELOCITY_WINDOWS_HOURS = [1, 6, 24, 168]  # 1h, 6h, 24h, 7d


class TemporalFeatureExtractor:
    """
    Extracts velocity features from transaction history.

    Features computed per user for each time window:
    - txn_count_{w}h: Number of transactions in window
    - amount_sum_{w}h: Total spend in window
    - amount_max_{w}h: Maximum single transaction in window
    - unique_merchants_{w}h: Distinct merchants in window
    - unique_countries_{w}h: Distinct countries in window (if available)

    Also computes:
    - hour_of_day: 0-23
    - day_of_week: 0-6
    - is_night: transaction between 23:00-05:00
    - is_weekend: Saturday or Sunday
    - days_since_first_txn: Account age proxy
    """

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all temporal features.

        Args:
            df: Sorted DataFrame (by user_id, timestamp) with at least
                [transaction_id, user_id, amount, timestamp] columns.

        Returns:
            DataFrame with transaction_id + temporal feature columns.
        """
        logger.debug("Extracting temporal features from %d transactions", len(df))

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        features = pd.DataFrame({"transaction_id": df["transaction_id"].values})

        # --- Time-of-day / calendar features ---
        features["hour_of_day"] = df["timestamp"].dt.hour.values
        features["day_of_week"] = df["timestamp"].dt.dayofweek.values
        features["is_night"] = (
            (df["timestamp"].dt.hour >= 23) | (df["timestamp"].dt.hour < 5)
        ).astype(int).values
        features["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int).values
        features["day_of_month"] = df["timestamp"].dt.day.values

        # --- Velocity features via expanding sort + merge ---
        velocity_feats = self._compute_velocity_features(df)
        features = features.merge(velocity_feats, on="transaction_id", how="left")

        # --- Account age ---
        user_first_txn = df.groupby("user_id")["timestamp"].transform("min")
        features["days_since_first_txn"] = (
            (df["timestamp"] - user_first_txn).dt.total_seconds() / 86400
        ).values

        # --- Time between transactions ---
        df_sorted = df.sort_values(["user_id", "timestamp"])
        df_sorted["prev_timestamp"] = df_sorted.groupby("user_id")["timestamp"].shift(1)
        df_sorted["seconds_since_last_txn"] = (
            df_sorted["timestamp"] - df_sorted["prev_timestamp"]
        ).dt.total_seconds().fillna(-1)

        features = features.merge(
            df_sorted[["transaction_id", "seconds_since_last_txn"]],
            on="transaction_id",
            how="left",
        )

        logger.debug("Temporal features shape: %s", features.shape)
        return features

    def _compute_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling window velocity features.
        Uses a merge-asof approach for efficiency on large datasets.
        """
        results = []

        for _, user_df in df.groupby("user_id"):
            user_df = user_df.sort_values("timestamp").copy()
            user_feats = {"transaction_id": user_df["transaction_id"].tolist()}

            for window_h in VELOCITY_WINDOWS_HOURS:
                window_td = pd.Timedelta(hours=window_h)
                counts, sums, maxes, unique_merch = [], [], [], []

                for idx, row in user_df.iterrows():
                    cutoff = row["timestamp"] - window_td
                    window_txns = user_df[
                        (user_df["timestamp"] > cutoff)
                        & (user_df["timestamp"] <= row["timestamp"])
                    ]
                    counts.append(len(window_txns))
                    sums.append(window_txns["amount"].sum())
                    maxes.append(window_txns["amount"].max() if len(window_txns) > 0 else 0)
                    unique_merch.append(window_txns["merchant_id"].nunique())

                user_feats[f"txn_count_{window_h}h"] = counts
                user_feats[f"amount_sum_{window_h}h"] = sums
                user_feats[f"amount_max_{window_h}h"] = maxes
                user_feats[f"unique_merchants_{window_h}h"] = unique_merch

            results.append(pd.DataFrame(user_feats))

        if not results:
            return pd.DataFrame({"transaction_id": df["transaction_id"].values})

        return pd.concat(results, ignore_index=True)
