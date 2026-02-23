"""
src/training/trainer.py
-----------------------
Main training orchestrator.
Wires together: data loading → feature engineering → training → evaluation → model registration.
All experiments are tracked via MLflow.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import get_settings
from src.core.exceptions import InsufficientDataError, TrainingError
from src.features.engineer import FraudFeatureEngineer
from src.models.ensemble import FraudEnsemble
from src.training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 10_000
MIN_FRAUD_SAMPLES = 100


class FraudModelTrainer:
    """
    End-to-end training pipeline orchestrator.

    Responsibilities:
    1. Load and validate training data
    2. Feature engineering (fit on train, transform train/val/test)
    3. Train ensemble model
    4. Comprehensive evaluation with multiple metrics
    5. Log everything to MLflow
    6. Register model in MLflow registry if performance threshold met
    7. Save artifacts locally + remote

    Usage:
        trainer = FraudModelTrainer()
        results = trainer.run(data_path="data/processed/transactions.parquet")
    """

    # Minimum AUC-PR to register model to Production
    PRODUCTION_AUCPR_THRESHOLD = 0.85

    def __init__(self) -> None:
        self._settings = get_settings()
        self._evaluator = ModelEvaluator()

    def run(
        self,
        data_path: str,
        experiment_name: str | None = None,
        run_name: str | None = None,
        skip_tabtransformer: bool = False,
        register_if_better: bool = True,
    ) -> dict[str, Any]:
        """
        Execute full training pipeline.

        Args:
            data_path: Path to processed Parquet training data
            experiment_name: MLflow experiment name (uses config default if None)
            run_name: MLflow run name for this experiment run
            skip_tabtransformer: Skip TabTransformer for faster iteration
            register_if_better: Auto-register to Production if AUC-PR improves

        Returns:
            Dict with model version, metrics, and artifact paths
        """
        start_time = time.time()
        settings = self._settings

        # Configure MLflow
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        exp_name = experiment_name or settings.mlflow.experiment_name
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name=run_name or f"training-{int(time.time())}") as run:
            run_id = run.info.run_id
            logger.info("MLflow run started: %s", run_id)

            # --- Step 1: Load data ---
            logger.info("Loading training data from %s", data_path)
            df = self._load_data(data_path)
            self._validate_data_requirements(df)

            mlflow.log_param("data_path", data_path)
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("fraud_rate", float(df["is_fraud"].mean()))

            # --- Step 2: Train/val/test split ---
            logger.info("Splitting data: train/val/test")
            X, y = df.drop(columns=["is_fraud"]), df["is_fraud"]
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=settings.training.test_size,
                random_state=settings.training.seed,
                stratify=y,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=settings.training.validation_size / (1 - settings.training.test_size),
                random_state=settings.training.seed,
                stratify=y_temp,
            )
            logger.info(
                "Split sizes: train=%d, val=%d, test=%d",
                len(X_train), len(X_val), len(X_test),
            )

            mlflow.log_params({
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size_count": len(X_test),
                "fraud_train_rate": float(y_train.mean()),
                "random_seed": settings.training.seed,
                "skip_tabtransformer": skip_tabtransformer,
            })

            # --- Step 3: Feature engineering ---
            logger.info("Fitting feature engineering pipeline")
            engineer = FraudFeatureEngineer(mode="training")
            X_train_feat = engineer.fit_transform(X_train.assign(is_fraud=y_train))
            X_val_feat = engineer.transform(X_val.assign(is_fraud=y_val)).drop(columns=["is_fraud"], errors="ignore")
            X_test_feat = engineer.transform(X_test.assign(is_fraud=y_test)).drop(columns=["is_fraud"], errors="ignore")

            # Remove target from feature matrices
            for col in ["is_fraud", "transaction_id", "user_id", "merchant_id", "timestamp"]:
                for feat_df in [X_train_feat, X_val_feat, X_test_feat]:
                    if col in feat_df.columns:
                        feat_df.drop(columns=[col], inplace=True)

            mlflow.log_param("n_features", X_train_feat.shape[1])
            mlflow.log_param("feature_names", list(X_train_feat.columns))

            # --- Step 4: Train model ---
            logger.info("Training ensemble model")
            ensemble = FraudEnsemble(skip_tabtransformer=skip_tabtransformer if hasattr(FraudEnsemble, 'skip_tabtransformer') else None)
            train_metrics = ensemble.train(
                X_train_feat, y_train,
                X_val=X_val_feat, y_val=y_val,
                skip_tabtransformer=skip_tabtransformer,
            )
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

            # --- Step 5: Evaluate on held-out test set ---
            logger.info("Evaluating on held-out test set")
            y_proba = ensemble.predict_proba(X_test_feat)
            eval_metrics = self._evaluator.evaluate(y_test.values, y_proba)

            # Log all metrics to MLflow
            mlflow.log_metrics(eval_metrics)
            logger.info("Test metrics: %s", eval_metrics)

            # --- Step 6: Save artifacts ---
            artifacts_dir = Path(settings.training.artifacts_dir) / run_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            ensemble.save(str(artifacts_dir / "model"))

            # Save feature engineer
            import joblib
            joblib.dump(engineer, artifacts_dir / "feature_engineer.joblib")

            # Log artifacts to MLflow
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="model")

            # Generate and log evaluation report
            report_path = artifacts_dir / "evaluation_report.json"
            self._evaluator.save_report(y_test.values, y_proba, str(report_path))
            mlflow.log_artifact(str(report_path))

            # --- Step 7: Register model ---
            model_version = None
            if register_if_better:
                model_version = self._maybe_register_model(
                    run_id=run_id,
                    eval_metrics=eval_metrics,
                    model_name="fraudguard-ensemble",
                )

            elapsed = time.time() - start_time
            mlflow.log_metric("training_duration_seconds", elapsed)

            logger.info("Training pipeline complete in %.1fs", elapsed)

            return {
                "run_id": run_id,
                "model_version": model_version,
                "metrics": eval_metrics,
                "artifacts_dir": str(artifacts_dir),
                "n_features": X_train_feat.shape[1],
            }

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load and basic-validate training data."""
        p = Path(path)
        if not p.exists():
            msg = f"Data file not found: {path}"
            raise FileNotFoundError(msg)

        if p.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif p.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            msg = f"Unsupported data format: {p.suffix}. Use .parquet or .csv"
            raise ValueError(msg)

        logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
        return df

    def _validate_data_requirements(self, df: pd.DataFrame) -> None:
        """Ensure minimum data requirements for meaningful training."""
        if len(df) < MIN_TRAINING_SAMPLES:
            raise InsufficientDataError(required=MIN_TRAINING_SAMPLES, actual=len(df))

        if "is_fraud" not in df.columns:
            msg = "Dataset missing required 'is_fraud' target column"
            raise TrainingError(msg)

        fraud_count = df["is_fraud"].sum()
        if fraud_count < MIN_FRAUD_SAMPLES:
            raise InsufficientDataError(required=MIN_FRAUD_SAMPLES, actual=fraud_count)

        logger.info(
            "Data validation passed: %d samples, %d fraud (%.2f%%)",
            len(df), fraud_count, fraud_count / len(df) * 100,
        )

    def _maybe_register_model(
        self,
        run_id: str,
        eval_metrics: dict[str, float],
        model_name: str,
    ) -> str | None:
        """Register model to MLflow if it meets quality thresholds."""
        aucpr = eval_metrics.get("test_auc_pr", 0.0)

        if aucpr < self.PRODUCTION_AUCPR_THRESHOLD:
            logger.warning(
                "Model AUC-PR %.4f below threshold %.4f. Not registering to Production.",
                aucpr, self.PRODUCTION_AUCPR_THRESHOLD,
            )
            return None

        model_uri = f"runs:/{run_id}/model"
        try:
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            version = result.version
            logger.info("Model registered: %s v%s", model_name, version)

            # Promote to Production
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Model promoted to Production: %s v%s", model_name, version)
            return version
        except Exception as exc:
            logger.error("Model registration failed: %s", exc)
            return None
