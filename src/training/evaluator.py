"""
src/training/evaluator.py
--------------------------
Comprehensive model evaluation for fraud detection.
Goes beyond accuracy — uses business-relevant metrics critical for
imbalanced classification in high-stakes financial applications.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive fraud model evaluation.

    Metrics computed:
    - Standard: AUC-ROC, AUC-PR, F1, Precision, Recall
    - Fraud-specific: KS Statistic, Gini coefficient
    - Threshold analysis: Precision@K recall levels
    - Calibration: Brier score, log-loss
    - Business: Fraud detection rate at various FPR constraints
    - Operational: Optimal threshold via Youden's J / F-beta

    All metrics are prefixed with 'test_' for MLflow logging.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        prefix: str = "test",
    ) -> dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: Ground truth binary labels (0=legit, 1=fraud)
            y_proba: Predicted fraud probabilities [0, 1]
            prefix: Metric name prefix for MLflow logging

        Returns:
            Dict of metric_name → float value
        """
        metrics: dict[str, float] = {}

        # --- Core ranking metrics ---
        metrics[f"{prefix}_auc_roc"] = roc_auc_score(y_true, y_proba)
        metrics[f"{prefix}_auc_pr"] = average_precision_score(y_true, y_proba)
        metrics[f"{prefix}_gini"] = 2 * metrics[f"{prefix}_auc_roc"] - 1

        # --- KS Statistic (key metric in banking/credit) ---
        metrics[f"{prefix}_ks_statistic"] = self._compute_ks_statistic(y_true, y_proba)

        # --- Threshold-based metrics at default 0.5 ---
        y_pred_05 = (y_proba >= 0.5).astype(int)
        metrics[f"{prefix}_precision_t50"] = precision_score(y_true, y_pred_05, zero_division=0)
        metrics[f"{prefix}_recall_t50"] = recall_score(y_true, y_pred_05, zero_division=0)
        metrics[f"{prefix}_f1_t50"] = f1_score(y_true, y_pred_05, zero_division=0)
        metrics[f"{prefix}_f2_t50"] = self._fbeta_score(y_true, y_pred_05, beta=2.0)

        # --- Optimal threshold (Youden's J) ---
        optimal_threshold, optimal_j = self._find_optimal_threshold(y_true, y_proba)
        metrics[f"{prefix}_optimal_threshold"] = optimal_threshold
        metrics[f"{prefix}_optimal_youden_j"] = optimal_j

        y_pred_opt = (y_proba >= optimal_threshold).astype(int)
        metrics[f"{prefix}_precision_optimal"] = precision_score(y_true, y_pred_opt, zero_division=0)
        metrics[f"{prefix}_recall_optimal"] = recall_score(y_true, y_pred_opt, zero_division=0)
        metrics[f"{prefix}_f1_optimal"] = f1_score(y_true, y_pred_opt, zero_division=0)

        # --- Precision at fixed recall levels (critical for fraud ops) ---
        for recall_target in [0.80, 0.85, 0.90, 0.95, 0.99]:
            prec_at_recall = self._precision_at_recall(y_true, y_proba, recall_target)
            metrics[f"{prefix}_prec_at_{int(recall_target*100)}recall"] = prec_at_recall

        # --- Calibration quality ---
        metrics[f"{prefix}_brier_score"] = brier_score_loss(y_true, y_proba)
        metrics[f"{prefix}_log_loss"] = log_loss(y_true, y_proba)

        # --- Class imbalance info ---
        metrics[f"{prefix}_fraud_rate"] = float(y_true.mean())
        metrics[f"{prefix}_n_samples"] = float(len(y_true))
        metrics[f"{prefix}_n_fraud"] = float(y_true.sum())

        # --- Business metric: fraud caught by reviewing top-K% alerts ---
        for pct in [1, 5, 10, 20]:
            caught = self._fraud_catch_rate_at_k_percent(y_true, y_proba, pct)
            metrics[f"{prefix}_fraud_catch_rate_top{pct}pct"] = caught

        logger.info(
            "Evaluation complete | AUC-ROC=%.4f | AUC-PR=%.4f | KS=%.4f | Optimal-F1=%.4f",
            metrics[f"{prefix}_auc_roc"],
            metrics[f"{prefix}_auc_pr"],
            metrics[f"{prefix}_ks_statistic"],
            metrics[f"{prefix}_f1_optimal"],
        )

        return metrics

    def save_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        output_path: str,
    ) -> None:
        """Save detailed evaluation report as JSON."""
        metrics = self.evaluate(y_true, y_proba)

        # Add confusion matrix
        optimal_t = metrics.get("test_optimal_threshold", 0.5)
        y_pred = (y_proba >= optimal_t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        report = {
            "metrics": metrics,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "classification_report": classification_report(
                y_true, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True
            ),
            "threshold_analysis": self._threshold_analysis(y_true, y_proba),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Evaluation report saved to %s", output_path)

    # --- Private metric helpers ---

    @staticmethod
    def _compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Kolmogorov-Smirnov statistic.
        Maximum separation between fraud and non-fraud score distributions.
        Standard metric in credit risk; ranges 0 (random) to 1 (perfect).
        """
        fraud_scores = y_proba[y_true == 1]
        legit_scores = y_proba[y_true == 0]

        if len(fraud_scores) == 0 or len(legit_scores) == 0:
            return 0.0

        from scipy.stats import ks_2samp
        ks_stat, _ = ks_2samp(fraud_scores, legit_scores)
        return float(ks_stat)

    @staticmethod
    def _fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
        """F-beta score. beta=2 weights recall twice as much as precision."""
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        denom = (beta**2 * prec) + rec
        if denom == 0:
            return 0.0
        return (1 + beta**2) * prec * rec / denom

    @staticmethod
    def _find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> tuple[float, float]:
        """Find threshold maximizing Youden's J statistic (sensitivity + specificity - 1)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        return float(thresholds[best_idx]), float(youden_j[best_idx])

    @staticmethod
    def _precision_at_recall(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_recall: float,
    ) -> float:
        """Precision when the model achieves at least target_recall."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        # Find highest precision at or above target recall
        eligible = precision[recall >= target_recall]
        return float(eligible.max()) if len(eligible) > 0 else 0.0

    @staticmethod
    def _fraud_catch_rate_at_k_percent(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        k_percent: int,
    ) -> float:
        """
        What fraction of all fraud is caught if we review top-K% of transactions?
        Critical operational metric — fraud analysts have finite review capacity.
        """
        n = len(y_true)
        k = max(1, int(n * k_percent / 100))
        top_k_idx = np.argsort(y_proba)[-k:]
        fraud_caught = y_true[top_k_idx].sum()
        total_fraud = y_true.sum()
        return float(fraud_caught / max(total_fraud, 1))

    @staticmethod
    def _threshold_analysis(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_thresholds: int = 20,
    ) -> list[dict[str, float]]:
        """Generate precision/recall/F1 at multiple thresholds for threshold tuning."""
        thresholds = np.linspace(0.1, 0.95, n_thresholds)
        analysis = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            analysis.append({
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "n_flagged": int(y_pred.sum()),
                "flag_rate_pct": float(y_pred.mean() * 100),
            })
        return analysis
