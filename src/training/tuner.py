"""
src/training/tuner.py
---------------------
Optuna-based hyperparameter optimization with MLflow integration.
Uses TPE sampler + Hyperband pruner for efficient exploration.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import optuna
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.models.xgboost_model import XGBoostFraudModel

logger = logging.getLogger(__name__)

# Silence Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class FraudHyperparamTuner:
    """
    Hyperparameter optimization using Optuna with MLflow experiment tracking.

    Optimizes XGBoost hyperparameters using cross-validated AUC-PR
    (most appropriate metric for imbalanced fraud detection).

    Usage:
        tuner = FraudHyperparamTuner(n_trials=100, cv_folds=5)
        best_params = tuner.tune(X_train, y_train)
    """

    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 5,
        timeout_seconds: int = 7200,
        metric: str = "aucpr",
        n_jobs: int = 4,
    ) -> None:
        self._n_trials = n_trials
        self._cv_folds = cv_folds
        self._timeout = timeout_seconds
        self._metric = metric
        self._n_jobs = n_jobs
        self._study: optuna.Study | None = None
        self._best_params: dict[str, Any] | None = None

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        mlflow_run_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            X: Training feature matrix
            y: Binary fraud labels
            mlflow_run_id: If provided, log results to existing MLflow run

        Returns:
            Best hyperparameter configuration found
        """
        logger.info(
            "Starting Optuna HPO: %d trials, %d-fold CV, metric=%s",
            self._n_trials, self._cv_folds, self._metric,
        )

        self._study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=100, max_resource=1200, reduction_factor=3
            ),
        )

        objective = self._build_objective(X, y)
        self._study.optimize(
            objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
            n_jobs=self._n_jobs,
            show_progress_bar=True,
        )

        self._best_params = self._study.best_params
        best_value = self._study.best_value

        logger.info("HPO complete | Best %s: %.4f", self._metric, best_value)
        logger.info("Best params: %s", self._best_params)

        # Log to MLflow
        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_params({f"tuned_{k}": v for k, v in self._best_params.items()})
                mlflow.log_metric(f"best_cv_{self._metric}", best_value)
                mlflow.log_metric("n_completed_trials", len(self._study.trials))

        return self._best_params

    def _build_objective(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Any:
        """Build Optuna objective function with cross-validation."""
        cv = StratifiedKFold(n_splits=self._cv_folds, shuffle=True, random_state=42)

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_xgboost_params(trial)
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                # Scale pos weight based on fold imbalance
                neg = (y_fold == 0).sum()
                pos = (y_fold == 1).sum()
                params["scale_pos_weight"] = neg / max(pos, 1)

                model = XGBoostFraudModel(hyperparams=params)
                model.train(X_fold, y_fold, X_val=X_val, y_val=y_val)

                y_proba = model.predict_proba(X_val)
                score = average_precision_score(y_val, y_proba)
                cv_scores.append(score)

                # Report intermediate value for pruning
                trial.report(np.mean(cv_scores), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(cv_scores))

        return objective

    @staticmethod
    def _suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
        """Define the hyperparameter search space for XGBoost."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "tree_method": "hist",
            "eval_metric": ["aucpr", "auc"],
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": 1,  # Parallelism handled by n_jobs in study
        }

    @property
    def best_params(self) -> dict[str, Any] | None:
        return self._best_params

    @property
    def study(self) -> optuna.Study | None:
        return self._study

    def get_importance_plot_data(self) -> dict[str, Any] | None:
        """Return hyperparameter importance data for visualization."""
        if self._study is None:
            return None
        try:
            importance = optuna.importance.get_param_importances(self._study)
            return dict(importance)
        except Exception:
            return None
