"""
src/models/xgboost_model.py
----------------------------
XGBoost fraud detection model with full sklearn compatibility.
Handles class imbalance via scale_pos_weight and SMOTE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.core.exceptions import ModelNotFittedError, TrainingError
from src.core.interfaces import BaseModel

logger = logging.getLogger(__name__)


class XGBoostFraudModel(BaseModel):
    """
    XGBoost gradient boosting model for fraud detection.

    Handles severe class imbalance (typical fraud rate: 1-3%) via:
    1. scale_pos_weight parameter (upweights fraud class)
    2. Optional probability calibration (isotonic regression)
    3. Early stopping on validation set

    Provides SHAP-based feature importance for explainability.
    """

    MODEL_FILENAME = "xgboost_fraud_model.joblib"

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """
        Args:
            hyperparams: XGBoost hyperparameters. Uses production defaults if None.
        """
        self._hyperparams = hyperparams or self._default_hyperparams()
        self._model: xgb.XGBClassifier | None = None
        self._calibrated_model: CalibratedClassifierCV | None = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        calibrate: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train XGBoost classifier.

        Args:
            X: Training feature matrix
            y: Binary fraud labels (0=legit, 1=fraud)
            X_val: Validation features for early stopping (optional)
            y_val: Validation labels (optional)
            calibrate: Whether to apply isotonic calibration

        Returns:
            Training metrics dict
        """
        logger.info("Training XGBoost on %d samples (fraud rate: %.2f%%)", len(X), y.mean() * 100)

        self._feature_names = list(X.columns)

        # Auto-compute scale_pos_weight if not provided
        if "scale_pos_weight" not in self._hyperparams:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            self._hyperparams["scale_pos_weight"] = neg_count / max(pos_count, 1)
            logger.info("Auto scale_pos_weight: %.1f", self._hyperparams["scale_pos_weight"])

        self._model = xgb.XGBClassifier(**self._hyperparams)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        try:
            self._model.fit(
                X,
                y,
                eval_set=eval_set,
                verbose=False,
            )
        except Exception as exc:
            msg = f"XGBoost training failed: {exc}"
            raise TrainingError(msg) from exc

        # Probability calibration
        if calibrate and X_val is not None and y_val is not None:
            logger.info("Calibrating probabilities with isotonic regression")
            self._calibrated_model = CalibratedClassifierCV(
                self._model, method="isotonic", cv="prefit"
            )
            self._calibrated_model.fit(X_val, y_val)

        self._is_fitted = True
        logger.info("XGBoost training complete. Best iteration: %d", self._model.best_iteration)

        return {
            "best_iteration": self._model.best_iteration,
            "best_score": self._model.best_score,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability scores.

        Args:
            X: Feature matrix

        Returns:
            1D array of fraud probabilities, shape (n_samples,)
        """
        if not self._is_fitted:
            raise ModelNotFittedError("XGBoost model has not been trained yet")

        model = self._calibrated_model if self._calibrated_model is not None else self._model
        proba = model.predict_proba(X[self._feature_names])
        return proba[:, 1]  # Fraud class probability

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary fraud predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self) -> dict[str, float]:
        """Return gain-based feature importance scores."""
        if not self._is_fitted or self._model is None:
            raise ModelNotFittedError("Model not fitted")
        importance = self._model.get_booster().get_score(importance_type="gain")
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for explainability."""
        if not self._is_fitted or self._model is None:
            raise ModelNotFittedError("Model not fitted")
        import shap
        explainer = shap.TreeExplainer(self._model)
        return explainer.shap_values(X[self._feature_names])

    def save(self, path: str) -> None:
        """Save model artifacts to directory."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self._model,
            "calibrated_model": self._calibrated_model,
            "hyperparams": self._hyperparams,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(artifact, save_path / self.MODEL_FILENAME)
        logger.info("XGBoost model saved to %s", save_path)

    def load(self, path: str) -> None:
        """Load model artifacts from directory."""
        load_path = Path(path) / self.MODEL_FILENAME
        if not load_path.exists():
            msg = f"Model artifact not found at {load_path}"
            raise FileNotFoundError(msg)

        artifact = joblib.load(load_path)
        self._model = artifact["model"]
        self._calibrated_model = artifact["calibrated_model"]
        self._hyperparams = artifact["hyperparams"]
        self._feature_names = artifact["feature_names"]
        self._is_fitted = artifact["is_fitted"]
        logger.info("XGBoost model loaded from %s", load_path)

    @staticmethod
    def _default_hyperparams() -> dict[str, Any]:
        return {
            "n_estimators": 1200,
            "max_depth": 7,
            "learning_rate": 0.023,
            "subsample": 0.82,
            "colsample_bytree": 0.78,
            "min_child_weight": 8,
            "gamma": 0.15,
            "reg_alpha": 0.12,
            "reg_lambda": 1.8,
            "tree_method": "hist",
            "eval_metric": ["aucpr", "auc"],
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1,
        }
