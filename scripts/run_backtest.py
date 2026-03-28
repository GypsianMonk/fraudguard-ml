#!/usr/bin/env python3
"""
scripts/run_backtest.py
------------------------
Temporal walk-forward backtest for the fraud detection model.

Simulates real-world deployment: train on past data, evaluate on future data,
rolling forward month by month. Catches temporal leakage and concept drift.

Usage:
    python scripts/run_backtest.py --data data/raw/transactions.parquet
    python scripts/run_backtest.py --data data/raw/transactions.parquet --windows 6 --step-days 30
    python scripts/run_backtest.py --data data/raw/transactions.parquet --output reports/backtest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineer import FraudFeatureEngineer
from src.models.xgboost_model import XGBoostFraudModel
from src.training.evaluator import ModelEvaluator
from src.utils.logging import configure_logging

configure_logging(level="INFO", format_="console")
logger = logging.getLogger(__name__)


def run_backtest(
    data_path: str,
    train_window_days: int = 180,
    test_window_days: int = 30,
    step_days: int = 30,
    min_fraud_samples: int = 50,
    skip_tabtransformer: bool = True,
) -> list[dict]:
    """
    Walk-forward backtest.

    For each time step:
        Train on [T - train_window, T)
        Evaluate on [T, T + test_window)

    Args:
        data_path: Path to transaction parquet/csv
        train_window_days: Days of history to train on
        test_window_days: Days of future data to evaluate on
        step_days: How many days to advance the window each iteration
        min_fraud_samples: Skip windows with fewer fraud samples than this
        skip_tabtransformer: Use XGBoost only for speed

    Returns:
        List of per-window metric dicts
    """
    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    train_window = timedelta(days=train_window_days)
    test_window = timedelta(days=test_window_days)
    step = timedelta(days=step_days)

    # First evaluation starts after enough training data
    cursor = t_min + train_window
    windows: list[dict] = []
    evaluator = ModelEvaluator()

    window_num = 0
    while cursor + test_window <= t_max:
        window_num += 1
        train_start = cursor - train_window
        train_end = cursor
        test_end = cursor + test_window

        train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)].copy()
        test_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < test_end)].copy()

        n_train_fraud = train_df["is_fraud"].sum() if "is_fraud" in train_df.columns else 0
        n_test_fraud = test_df["is_fraud"].sum() if "is_fraud" in test_df.columns else 0

        logger.info(
            "Window %d | train=%s→%s (%d rows, %d fraud) | test=%s→%s (%d rows, %d fraud)",
            window_num,
            train_start.date(),
            train_end.date(),
            len(train_df),
            n_train_fraud,
            train_end.date(),
            test_end.date(),
            len(test_df),
            n_test_fraud,
        )

        if n_train_fraud < min_fraud_samples or n_test_fraud < 5:
            logger.warning("Window %d skipped — insufficient fraud samples", window_num)
            cursor += step
            continue

        if len(train_df) < 1000:
            logger.warning(
                "Window %d skipped — insufficient training data (%d rows)",
                window_num,
                len(train_df),
            )
            cursor += step
            continue

        try:
            # Feature engineering — fit on train only (no leakage)
            engineer = FraudFeatureEngineer(mode="training")
            X_train = engineer.fit_transform(train_df)
            y_train = (
                X_train.pop("is_fraud") if "is_fraud" in X_train.columns else train_df["is_fraud"]
            )

            X_test = engineer.transform(test_df.assign(is_fraud=test_df.get("is_fraud", 0)))
            y_test = test_df["is_fraud"].values

            # Drop non-feature columns
            drop_cols = ["is_fraud", "transaction_id", "user_id", "merchant_id", "timestamp"]
            for col in drop_cols:
                X_train.drop(columns=[col], errors="ignore", inplace=True)
                X_test.drop(columns=[col], errors="ignore", inplace=True)

            # Train XGBoost
            model = XGBoostFraudModel()
            model.train(X_train, y_train)

            y_proba = model.predict_proba(X_test)
            metrics = evaluator.evaluate(y_test, y_proba, prefix="backtest")

            window_result = {
                "window": window_num,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_train_fraud": int(n_train_fraud),
                "n_test_fraud": int(n_test_fraud),
                "train_fraud_rate": round(float(n_train_fraud / max(len(train_df), 1)), 5),
                **{k: round(float(v), 5) for k, v in metrics.items() if isinstance(v, float)},
            }
            windows.append(window_result)

            logger.info(
                "Window %d → AUC-ROC=%.4f AUC-PR=%.4f Recall=%.4f",
                window_num,
                metrics.get("backtest_auc_roc", 0),
                metrics.get("backtest_auc_pr", 0),
                metrics.get("backtest_recall_optimal", 0),
            )

        except Exception as exc:
            logger.error("Window %d failed: %s", window_num, exc)

        cursor += step

    return windows


def print_summary(windows: list[dict]) -> None:
    """Print backtest summary statistics."""
    if not windows:
        logger.warning("No windows completed")
        return

    auc_rocs = [w.get("backtest_auc_roc", 0) for w in windows]
    auc_prs = [w.get("backtest_auc_pr", 0) for w in windows]
    recalls = [w.get("backtest_recall_optimal", 0) for w in windows]

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Windows completed  : {len(windows)}")
    print(f"Date range         : {windows[0]['train_end']} → {windows[-1]['test_end']}")
    print()
    print(
        f"AUC-ROC  — mean={np.mean(auc_rocs):.4f}  std={np.std(auc_rocs):.4f}  "
        f"min={np.min(auc_rocs):.4f}  max={np.max(auc_rocs):.4f}"
    )
    print(
        f"AUC-PR   — mean={np.mean(auc_prs):.4f}  std={np.std(auc_prs):.4f}  "
        f"min={np.min(auc_prs):.4f}  max={np.max(auc_prs):.4f}"
    )
    print(
        f"Recall   — mean={np.mean(recalls):.4f}  std={np.std(recalls):.4f}  "
        f"min={np.min(recalls):.4f}  max={np.max(recalls):.4f}"
    )
    print("=" * 60)

    # Stability check
    if np.std(auc_prs) > 0.05:
        print("⚠️  High AUC-PR variance across windows — model may be unstable")
    else:
        print("✅ AUC-PR is stable across windows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FraudGuard ML — Walk-forward backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="data/raw/transactions.parquet")
    parser.add_argument("--train-days", type=int, default=180, help="Training window in days")
    parser.add_argument("--test-days", type=int, default=30, help="Test window in days")
    parser.add_argument("--step-days", type=int, default=30, help="Step size in days")
    parser.add_argument(
        "--min-fraud", type=int, default=50, help="Min fraud samples to include a window"
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON path")
    args = parser.parse_args()

    if not Path(args.data).exists():
        logger.error("Data file not found: %s", args.data)
        logger.info("Tip: run 'python scripts/generate_synthetic_data.py' first")
        sys.exit(1)

    logger.info(
        "Starting backtest | train=%dd test=%dd step=%dd",
        args.train_days,
        args.test_days,
        args.step_days,
    )

    windows = run_backtest(
        data_path=args.data,
        train_window_days=args.train_days,
        test_window_days=args.test_days,
        step_days=args.step_days,
        min_fraud_samples=args.min_fraud,
    )

    print_summary(windows)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump({"windows": windows, "n_windows": len(windows)}, f, indent=2, default=str)
        logger.info("Results saved → %s", args.output)


if __name__ == "__main__":
    main()
