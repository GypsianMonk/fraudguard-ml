"""
tests/unit/test_evaluator.py
-----------------------------
Unit tests for ModelEvaluator.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.training.evaluator import ModelEvaluator


@pytest.fixture
def evaluator() -> ModelEvaluator:
    return ModelEvaluator()


@pytest.fixture
def balanced_predictions() -> tuple[np.ndarray, np.ndarray]:
    """50/50 balanced test set with reasonable model."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = np.array([1] * 500 + [0] * 500)
    y_proba = np.where(y_true == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    return y_true, y_proba


@pytest.fixture
def imbalanced_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Realistic fraud ratio (2%) with reasonable model."""
    rng = np.random.default_rng(42)
    n = 10000
    fraud_mask = rng.random(n) < 0.02
    y_true = fraud_mask.astype(int)
    y_proba = np.where(fraud_mask, rng.beta(4, 2, n), rng.beta(1, 8, n))
    return y_true, y_proba


@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Perfect model predictions."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.95, 0.90, 0.85, 0.10, 0.05, 0.02])
    return y_true, y_proba


@pytest.fixture
def random_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Random (useless) model predictions."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = (rng.random(n) > 0.5).astype(int)
    y_proba = rng.random(n)
    return y_true, y_proba


class TestModelEvaluator:
    """Unit tests for ModelEvaluator."""

    def test_evaluate_returns_dict(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_all_expected_metrics_present(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)

        required_keys = [
            "test_auc_roc", "test_auc_pr", "test_gini", "test_ks_statistic",
            "test_precision_t50", "test_recall_t50", "test_f1_t50",
            "test_brier_score", "test_log_loss", "test_optimal_threshold",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_auc_roc_range(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        assert 0.0 <= metrics["test_auc_roc"] <= 1.0

    def test_auc_pr_range(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        assert 0.0 <= metrics["test_auc_pr"] <= 1.0

    def test_brier_score_range(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        assert 0.0 <= metrics["test_brier_score"] <= 1.0

    def test_perfect_model_auc_roc_near_1(self, evaluator: ModelEvaluator, perfect_predictions: tuple) -> None:
        y_true, y_proba = perfect_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        assert metrics["test_auc_roc"] > 0.99

    def test_random_model_auc_roc_near_0_5(self, evaluator: ModelEvaluator, random_predictions: tuple) -> None:
        y_true, y_proba = random_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        # Random model should be near 0.5 AUC
        assert 0.40 <= metrics["test_auc_roc"] <= 0.60

    def test_good_model_better_than_random(
        self,
        evaluator: ModelEvaluator,
        balanced_predictions: tuple,
        random_predictions: tuple,
    ) -> None:
        y_good, p_good = balanced_predictions
        y_rand, p_rand = random_predictions
        good_auc = evaluator.evaluate(y_good, p_good)["test_auc_roc"]
        rand_auc = evaluator.evaluate(y_rand, p_rand)["test_auc_roc"]
        assert good_auc > rand_auc

    def test_ks_statistic_range(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        assert 0.0 <= metrics["test_ks_statistic"] <= 1.0

    def test_gini_equals_2_times_auc_minus_1(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        expected_gini = 2 * metrics["test_auc_roc"] - 1
        assert abs(metrics["test_gini"] - expected_gini) < 1e-9

    def test_precision_at_recall_thresholds_present(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        for recall_target in [80, 85, 90, 95, 99]:
            assert f"test_prec_at_{recall_target}recall" in metrics

    def test_fraud_catch_rate_metrics_present(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        for pct in [1, 5, 10, 20]:
            assert f"test_fraud_catch_rate_top{pct}pct" in metrics

    def test_fraud_catch_rate_values_in_range(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        for pct in [1, 5, 10, 20]:
            rate = metrics[f"test_fraud_catch_rate_top{pct}pct"]
            assert 0.0 <= rate <= 1.0

    def test_save_report_creates_json_file(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = str(Path(tmp_dir) / "report.json")
            evaluator.save_report(y_true, y_proba, report_path)

            assert Path(report_path).exists()
            with open(report_path) as f:
                report = json.load(f)

            assert "metrics" in report
            assert "confusion_matrix" in report
            assert "classification_report" in report
            assert "threshold_analysis" in report

    def test_save_report_confusion_matrix_keys(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = str(Path(tmp_dir) / "report.json")
            evaluator.save_report(y_true, y_proba, report_path)

            with open(report_path) as f:
                report = json.load(f)

            cm = report["confusion_matrix"]
            assert "true_negatives" in cm
            assert "false_positives" in cm
            assert "false_negatives" in cm
            assert "true_positives" in cm

    def test_custom_prefix(self, evaluator: ModelEvaluator, balanced_predictions: tuple) -> None:
        y_true, y_proba = balanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba, prefix="val")
        assert "val_auc_roc" in metrics
        assert "test_auc_roc" not in metrics

    def test_imbalanced_data_auc_pr_reasonable(
        self, evaluator: ModelEvaluator, imbalanced_predictions: tuple
    ) -> None:
        y_true, y_proba = imbalanced_predictions
        metrics = evaluator.evaluate(y_true, y_proba)
        # For 2% fraud rate, a good model should get AUC-PR > 0.3
        assert metrics["test_auc_pr"] > 0.3
