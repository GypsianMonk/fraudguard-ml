#!/usr/bin/env python3
"""
scripts/run_training.py
------------------------
CLI entry point for the end-to-end training pipeline.
Can be run locally, in Docker, or as a Kubernetes Job.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --data data/raw/transactions.parquet
    python scripts/run_training.py --tune --n-trials 50
    python scripts/run_training.py --skip-tabtransformer --no-register
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import configure_logging

configure_logging(level="INFO", format_="console")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FraudGuard ML — Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/transactions.parquet",
        help="Path to training data (.parquet or .csv)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (uses config default if not set)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run Optuna hyperparameter optimization before training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (only with --tune)",
    )
    parser.add_argument(
        "--skip-tabtransformer",
        action="store_true",
        default=False,
        help="Skip TabTransformer for faster iteration (XGBoost only)",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        default=False,
        help="Do not register model to MLflow registry",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Path to write metrics JSON (for CI/CD quality gate)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FraudGuard ML Training Pipeline")
    logger.info("=" * 60)
    logger.info("Data:              %s", args.data)
    logger.info("Experiment:        %s", args.experiment or "config default")
    logger.info("HPO:               %s", "enabled" if args.tune else "disabled")
    logger.info("Skip TabTransform: %s", args.skip_tabtransformer)
    logger.info("Register model:    %s", not args.no_register)
    logger.info("=" * 60)

    # Validate data path exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        logger.info("Tip: Run 'python scripts/generate_synthetic_data.py' to create sample data")
        sys.exit(1)

    # ── Optional HPO ─────────────────────────────────────────────────────────
    if args.tune:
        logger.info("Running hyperparameter optimization (%d trials)...", args.n_trials)
        from src.training.tuner import FraudHyperparamTuner
        import pandas as pd
        from src.features.engineer import FraudFeatureEngineer
        from sklearn.model_selection import train_test_split

        df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
        df_sample = df.sample(min(50_000, len(df)), random_state=42)

        engineer = FraudFeatureEngineer(mode="training")
        X_feat = engineer.fit_transform(df_sample)
        for col in ["is_fraud", "transaction_id", "user_id", "merchant_id", "timestamp"]:
            X_feat.drop(columns=[col], errors="ignore", inplace=True)

        y = df_sample["is_fraud"]
        X_train, X_val, y_train, y_val = train_test_split(X_feat, y, test_size=0.2, stratify=y, random_state=42)

        tuner = FraudHyperparamTuner(n_trials=args.n_trials, cv_folds=3)
        best_params = tuner.tune(X_train, y_train)
        logger.info("Best params found: %s", best_params)

    # ── Training ─────────────────────────────────────────────────────────────
    from src.training.trainer import FraudModelTrainer

    trainer = FraudModelTrainer()
    results = trainer.run(
        data_path=args.data,
        experiment_name=args.experiment,
        run_name=args.run_name,
        skip_tabtransformer=args.skip_tabtransformer,
        register_if_better=not args.no_register,
    )

    # ── Output results ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("MLflow Run ID:    %s", results["run_id"])
    logger.info("Model Version:    %s", results.get("model_version", "not registered"))
    logger.info("N Features:       %d", results["n_features"])
    logger.info("Artifacts:        %s", results["artifacts_dir"])
    logger.info("")
    logger.info("Test Metrics:")
    for k, v in sorted(results["metrics"].items()):
        if isinstance(v, float):
            logger.info("  %-40s %.4f", k, v)

    # Save metrics for CI/CD quality gate
    if args.output_metrics:
        metrics_path = Path(args.output_metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as f:
            json.dump(results["metrics"], f, indent=2, default=str)
        logger.info("Metrics written to %s", metrics_path)

    # Quality gate check
    aucpr = results["metrics"].get("test_auc_pr", 0.0)
    if aucpr < 0.70:
        logger.error("Model failed quality gate! AUC-PR=%.4f < 0.70 minimum", aucpr)
        sys.exit(2)

    logger.info("✅ Quality gate passed: AUC-PR=%.4f", aucpr)


if __name__ == "__main__":
    main()
