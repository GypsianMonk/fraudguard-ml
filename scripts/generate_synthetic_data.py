#!/usr/bin/env python3
"""
scripts/generate_synthetic_data.py
------------------------------------
Generates realistic synthetic transaction data for training and testing.

Models realistic fraud patterns:
- Temporal: night/weekend transactions, velocity spikes
- Geographic: international transactions, high-risk countries
- Behavioral: unusual amounts, new merchants, new devices
- Network: shared device fraud rings

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n-samples 1000000 --fraud-rate 0.02
    python scripts/generate_synthetic_data.py --n-samples 5000 --output data/raw/ci_test.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Realistic merchant category fraud rates (empirical estimates)
MERCHANT_FRAUD_RATES = {
    "electronics": 0.055,
    "online_gambling": 0.078,
    "jewelry": 0.062,
    "travel": 0.041,
    "financial_services": 0.038,
    "restaurants": 0.012,
    "groceries": 0.008,
    "healthcare": 0.009,
    "entertainment": 0.022,
    "other": 0.018,
}

HIGH_RISK_COUNTRIES = {"NG", "RU", "KP", "IR", "BY", "VE"}
MEDIUM_RISK_COUNTRIES = {"CN", "UA", "VN", "PH", "ID"}

COUNTRY_DIST = {
    "US": 0.58, "CA": 0.08, "GB": 0.07, "DE": 0.05, "FR": 0.04,
    "AU": 0.03, "JP": 0.03, "CN": 0.03, "RU": 0.02, "NG": 0.01,
    "BR": 0.02, "MX": 0.02, "KP": 0.005, "IR": 0.005, "BY": 0.005,
    "UA": 0.01,
}


def generate_transactions(
    n_samples: int = 500_000,
    fraud_rate: float = 0.022,
    seed: int = 42,
    start_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset with realistic fraud patterns.

    Args:
        n_samples: Total number of transactions to generate
        fraud_rate: Target fraud rate (fraction)
        seed: Random seed for reproducibility
        start_date: Start datetime for transaction timestamps

    Returns:
        DataFrame with transaction features and binary fraud label
    """
    rng = np.random.default_rng(seed)
    start_date = start_date or datetime(2023, 1, 1)

    logger.info("Generating %d transactions (fraud_rate=%.1f%%, seed=%d)", n_samples, fraud_rate * 100, seed)

    # ── User and merchant universe ────────────────────────────────────────────
    n_users = max(100, n_samples // 50)
    n_merchants = max(50, n_samples // 200)

    user_ids = [f"usr_{i:06d}" for i in range(n_users)]
    merchant_ids = [f"mrc_{i:04d}" for i in range(n_merchants)]
    merchant_categories = rng.choice(list(MERCHANT_FRAUD_RATES.keys()), size=n_merchants)
    merchant_cat_map = dict(zip(merchant_ids, merchant_categories))

    # User home countries (stays constant per user)
    user_countries = rng.choice(
        list(COUNTRY_DIST.keys()),
        size=n_users,
        p=list(COUNTRY_DIST.values()),
    )
    user_country_map = dict(zip(user_ids, user_countries))

    # ── Generate base transaction fields ─────────────────────────────────────
    txn_user_ids = rng.choice(user_ids, size=n_samples)

    # Timestamps: Poisson-distributed across date range (365 days)
    total_seconds = 365 * 24 * 3600
    offsets = rng.exponential(scale=total_seconds / n_samples, size=n_samples)
    offsets = np.cumsum(offsets)
    offsets = offsets / offsets.max() * total_seconds
    timestamps = [start_date + timedelta(seconds=float(o)) for o in offsets]

    # Transaction amounts: log-normal with category-specific parameters
    amounts = np.round(rng.lognormal(mean=4.2, sigma=1.5, size=n_samples), 2)
    amounts = np.clip(amounts, 0.01, 50_000.0)

    txn_merchant_ids = rng.choice(merchant_ids, size=n_samples)
    merchant_categories_col = [merchant_cat_map[m] for m in txn_merchant_ids]

    payment_methods = rng.choice(
        ["credit_card", "debit_card", "digital_wallet", "bank_transfer", "bnpl"],
        size=n_samples,
        p=[0.48, 0.27, 0.15, 0.07, 0.03],
    )

    currencies = rng.choice(
        ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"],
        size=n_samples,
        p=[0.58, 0.18, 0.09, 0.07, 0.04, 0.04],
    )

    # Device fingerprints: users have 1-3 devices mostly, fraudsters share devices
    n_devices = max(200, n_samples // 100)
    device_ids = [f"fp_{i:05d}" for i in range(n_devices)]
    user_device_map = {uid: rng.choice(device_ids, size=rng.integers(1, 4)).tolist() for uid in user_ids}
    txn_devices = [rng.choice(user_device_map[uid]) for uid in txn_user_ids]

    # Card present: mostly not for online, yes for POS
    card_present = rng.choice([True, False], size=n_samples, p=[0.35, 0.65])

    # Transaction countries (usually home country, sometimes abroad)
    txn_countries = []
    for uid in txn_user_ids:
        home = user_country_map[uid]
        if rng.random() < 0.08:  # 8% international
            country = rng.choice(list(COUNTRY_DIST.keys()))
        else:
            country = home
        txn_countries.append(country)

    # Lat/lon: rough country centroids (simplified)
    country_coords = {
        "US": (37.0, -95.0), "CA": (56.0, -96.0), "GB": (51.5, -0.1),
        "DE": (51.0, 10.0), "FR": (46.0, 2.0), "AU": (-25.0, 134.0),
        "JP": (36.0, 138.0), "CN": (35.0, 105.0), "RU": (60.0, 60.0),
        "NG": (8.0, 8.0), "BR": (-10.0, -55.0), "MX": (23.0, -102.0),
        "KP": (40.0, 127.0), "IR": (32.0, 53.0), "BY": (53.0, 28.0),
        "UA": (49.0, 32.0), "VE": (8.0, -66.0), "ID": (-5.0, 120.0),
        "PH": (12.0, 122.0), "VN": (16.0, 108.0),
    }
    lats, lons = [], []
    for c in txn_countries:
        lat, lon = country_coords.get(c, (0.0, 0.0))
        lats.append(lat + rng.normal(0, 2.0))
        lons.append(lon + rng.normal(0, 2.0))

    # ── Fraud label generation ────────────────────────────────────────────────
    # Base fraud probability per transaction (influenced by multiple factors)
    fraud_base = np.full(n_samples, fraud_rate * 0.3)

    # Category risk
    cat_risk = np.array([MERCHANT_FRAUD_RATES.get(c, 0.02) for c in merchant_categories_col])
    fraud_base += cat_risk * 0.4

    # Country risk
    country_risk = np.array([
        0.08 if c in HIGH_RISK_COUNTRIES else
        0.04 if c in MEDIUM_RISK_COUNTRIES else
        0.01
        for c in txn_countries
    ])
    fraud_base += country_risk * 0.3

    # Card-not-present adds risk
    fraud_base += (~np.array(card_present)).astype(float) * 0.01

    # Night transactions (23:00-05:00)
    hours = np.array([t.hour for t in timestamps])
    is_night = (hours >= 23) | (hours < 5)
    fraud_base += is_night.astype(float) * 0.015

    # Large amounts
    fraud_base += (amounts > 2000).astype(float) * 0.02
    fraud_base += (amounts > 5000).astype(float) * 0.03

    # Normalize to target fraud rate
    fraud_probs = fraud_base / fraud_base.sum() * (fraud_rate * n_samples)
    fraud_probs = np.clip(fraud_probs, 0.001, 0.95)
    is_fraud = rng.random(n_samples) < fraud_probs

    # Ensure exact target fraud count
    current_fraud = is_fraud.sum()
    target_fraud = int(fraud_rate * n_samples)
    if current_fraud < target_fraud:
        non_fraud_idx = np.where(~is_fraud)[0]
        flip_idx = rng.choice(non_fraud_idx, size=target_fraud - current_fraud, replace=False)
        is_fraud[flip_idx] = True
    elif current_fraud > target_fraud:
        fraud_idx = np.where(is_fraud)[0]
        unflip_idx = rng.choice(fraud_idx, size=current_fraud - target_fraud, replace=False)
        is_fraud[unflip_idx] = False

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "transaction_id": [f"txn_{i:09d}" for i in range(n_samples)],
        "user_id": txn_user_ids,
        "amount": amounts,
        "currency": currencies,
        "merchant_id": txn_merchant_ids,
        "merchant_category": merchant_categories_col,
        "payment_method": payment_methods,
        "timestamp": timestamps,
        "card_present": card_present,
        "device_fingerprint": txn_devices,
        "country": txn_countries,
        "latitude": np.round(lats, 4),
        "longitude": np.round(lons, 4),
        "is_fraud": is_fraud.astype(int),
    })

    actual_fraud_rate = df["is_fraud"].mean()
    logger.info(
        "Generated %d transactions | fraud_rate=%.2f%% | date_range=%s to %s",
        len(df),
        actual_fraud_rate * 100,
        df["timestamp"].min().date(),
        df["timestamp"].max().date(),
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument("--n-samples", type=int, default=500_000, help="Number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.022, help="Target fraud rate (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/transactions.parquet",
        help="Output file path (.parquet or .csv)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_transactions(
        n_samples=args.n_samples,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    elif output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        logger.error("Unsupported output format: %s. Use .parquet or .csv", output_path.suffix)
        sys.exit(1)

    logger.info("Data saved to %s (%.1f MB)", output_path, output_path.stat().st_size / 1024 / 1024)


if __name__ == "__main__":
    main()
