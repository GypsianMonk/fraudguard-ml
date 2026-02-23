"""
src/models/ensemble.py
-----------------------
Stacking ensemble that combines XGBoost + TabTransformer predictions
using a calibrated Logistic Regression meta-learner.

Training procedure (two-stage stacking):
    Stage 1: Train base learners with cross-validation, collect OOF predictions
    Stage 2: Train meta-learner on OOF predictions → final calibrated probabilities
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.core.exceptions import EnsembleError, ModelNotFittedError
from src.core.interfaces import BaseModel
from src.models.tabtransformer import TabTransformerModel
from src.models.xgboost_model import XGBoostFraudModel

logger = logging.getLogger(__name__)


class FraudEnsemble(BaseModel):
    """
    Two-stage stacking ensemble for fraud detection.

    Architecture:
        Level 0 (Base Learners):
            - XGBoost (trained on full training set)
            - TabTransformer (trained on full training set)

        Level 1 (Meta-Learner):
            - Logistic Regression on [xgb_prob, tab_prob] → final fraud probability

    The meta-learner is trained on out-of-fold (OOF) predictions from the
    base learners to prevent label leakage.

    Inference:
        final_prob = sigmoid(
            w_xgb * logit(xgb_prob) + w_tab * logit(tab_prob) + meta_bias
        )
    """

    ENSEMBLE_FILENAME = "ensemble.joblib"

    def __init__(
        self,
        xgb_hyperparams: dict[str, Any] | None = None,
        tab_hyperparams: dict[str, Any] | None = None,
        ensemble_weights: dict[str, float] | None = None,
        n_folds: int = 5,
    ) -> None:
        self._xgb = XGBoostFraudModel(xgb_hyperparams)
        self._tab = TabTransformerModel(tab_hyperparams)
        self._meta: LogisticRegression | None = None
        self._weights = ensemble_weights or {"xgboost": 0.55, "tabtransformer": 0.45}
        self._n_folds = n_folds
        self._is_fitted = False
        self._training_oof_auc: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        skip_tabtransformer: bool = False,  # For faster iteration
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train the full stacking ensemble.

        Args:
            X: Training features
            y: Fraud labels
            X_val: Held-out validation features
            y_val: Held-out validation labels
            skip_tabtransformer: If True, use only XGBoost (faster debugging)

        Returns:
            Training metrics dict with OOF scores
        """
        logger.info("Training stacking ensemble on %d samples", len(X))
        from sklearn.metrics import average_precision_score, roc_auc_score

        # === Stage 1: Out-of-fold predictions for meta-learner training ===
        oof_xgb = np.zeros(len(X))
        oof_tab = np.zeros(len(X))

        skf = StratifiedKFold(n_splits=self._n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, oof_idx) in enumerate(skf.split(X, y)):
            logger.info("Training fold %d/%d", fold + 1, self._n_folds)

            X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
            X_oof, y_oof = X.iloc[oof_idx], y.iloc[oof_idx]

            # XGBoost fold
            xgb_fold = XGBoostFraudModel()
            xgb_fold.train(X_fold, y_fold, X_val=X_oof, y_val=y_oof)
            oof_xgb[oof_idx] = xgb_fold.predict_proba(X_oof)

            # TabTransformer fold (skip for speed if requested)
            if not skip_tabtransformer:
                tab_fold = TabTransformerModel()
                tab_fold.train(X_fold, y_fold, X_val=X_oof, y_val=y_oof)
                oof_tab[oof_idx] = tab_fold.predict_proba(X_oof)
            else:
                oof_tab[oof_idx] = oof_xgb[oof_idx]  # Fallback

        # === Stage 2: Train meta-learner on OOF predictions ===
        oof_features = np.column_stack([oof_xgb, oof_tab])
        self._meta = LogisticRegression(C=1.2, max_iter=500, class_weight="balanced")
        self._meta.fit(oof_features, y)

        oof_meta = self._meta.predict_proba(oof_features)[:, 1]
        oof_auc = roc_auc_score(y, oof_meta)
        oof_aucpr = average_precision_score(y, oof_meta)
        self._training_oof_auc = oof_auc

        logger.info("OOF AUC-ROC: %.4f | OOF AUC-PR: %.4f", oof_auc, oof_aucpr)

        # === Stage 3: Retrain base learners on FULL training set ===
        logger.info("Retraining base learners on full training set")
        self._xgb.train(X, y, X_val=X_val, y_val=y_val)

        if not skip_tabtransformer:
            self._tab.train(X, y, X_val=X_val, y_val=y_val)

        self._is_fitted = True

        metrics = {
            "oof_auc_roc": oof_auc,
            "oof_auc_pr": oof_aucpr,
        }

        if X_val is not None and y_val is not None:
            val_proba = self.predict_proba(X_val)
            metrics["val_auc_roc"] = roc_auc_score(y_val, val_proba)
            metrics["val_auc_pr"] = average_precision_score(y_val, val_proba)
            logger.info("Val AUC-ROC: %.4f | Val AUC-PR: %.4f", metrics["val_auc_roc"], metrics["val_auc_pr"])

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Ensemble fraud probability via meta-learner.

        Args:
            X: Feature matrix

        Returns:
            1D array of fraud probabilities
        """
        if not self._is_fitted:
            msg = "Ensemble not fitted"
raise ModelNotFittedError(msg)

        try:
            xgb_proba = self._xgb.predict_proba(X)
            tab_proba = self._tab.predict_proba(X) if self._tab.is_fitted else xgb_proba

            if self._meta is not None:
                meta_features = np.column_stack([xgb_proba, tab_proba])
                return self._meta.predict_proba(meta_features)[:, 1]

            # Weighted average fallback
            return (
                self._weights["xgboost"] * xgb_proba
                + self._weights["tabtransformer"] * tab_proba
            )
        except Exception as exc:
            msg = f"Ensemble prediction failed: {exc}"
            raise EnsembleError(msg) from exc

    def get_feature_importance(self) -> dict[str, float]:
        """Return XGBoost feature importance as ensemble-level importance."""
        return self._xgb.get_feature_importance()

    def get_base_learner_probabilities(
        self, X: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Return individual base learner probabilities (for analysis/debugging)."""
        if not self._is_fitted:
            msg = "Ensemble not fitted"
raise ModelNotFittedError(msg)
        return {
            "xgboost": self._xgb.predict_proba(X),
            "tabtransformer": self._tab.predict_proba(X) if self._tab.is_fitted else np.zeros(len(X)),
        }

    def save(self, path: str) -> None:
        """Save all ensemble components."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self._xgb.save(str(save_path / "xgboost"))
        self._tab.save(str(save_path / "tabtransformer"))

        meta_artifact = {
            "meta": self._meta,
            "weights": self._weights,
            "n_folds": self._n_folds,
            "is_fitted": self._is_fitted,
            "training_oof_auc": self._training_oof_auc,
        }
        joblib.dump(meta_artifact, save_path / self.ENSEMBLE_FILENAME)
        logger.info("Ensemble saved to %s", save_path)

    def load(self, path: str) -> None:
        """Load all ensemble components."""
        load_path = Path(path)

        self._xgb.load(str(load_path / "xgboost"))
        self._tab.load(str(load_path / "tabtransformer"))

        meta_artifact = joblib.load(load_path / self.ENSEMBLE_FILENAME)
        self._meta = meta_artifact["meta"]
        self._weights = meta_artifact["weights"]
        self._n_folds = meta_artifact["n_folds"]
        self._is_fitted = meta_artifact["is_fitted"]
        self._training_oof_auc = meta_artifact["training_oof_auc"]

        logger.info("Ensemble loaded from %s (OOF AUC: %.4f)", load_path, self._training_oof_auc or 0)
