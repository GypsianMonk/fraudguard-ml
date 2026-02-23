"""
tests/unit/test_xgboost_model.py
---------------------------------
Unit tests for XGBoostFraudModel.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import ModelNotFittedError
from src.models.xgboost_model import XGBoostFraudModel


def make_fraud_dataset(n: int = 500, fraud_rate: float = 0.05, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a simple fraud classification dataset."""
    rng = np.random.default_rng(seed)
    n_fraud = max(10, int(n * fraud_rate))
    n_legit = n - n_fraud

    fraud_features = rng.standard_normal((n_fraud, 10)) + np.array([2, -1, 0.5, 1, -0.5, 0.3, 1.2, -0.8, 0.6, -0.3])
    legit_features = rng.standard_normal((n_legit, 10))

    X = pd.DataFrame(
        np.vstack([fraud_features, legit_features]),
        columns=[f"feature_{i}" for i in range(10)],
    )
    y = pd.Series([1] * n_fraud + [0] * n_legit, name="is_fraud")
    shuffle_idx = rng.permutation(len(y))
    return X.iloc[shuffle_idx].reset_index(drop=True), y.iloc[shuffle_idx].reset_index(drop=True)


@pytest.fixture
def dataset():
    X, y = make_fraud_dataset(n=500)
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture
def fitted_model(dataset):
    X_train, X_val, y_train, y_val = dataset
    model = XGBoostFraudModel()
    model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    return model


class TestXGBoostFraudModel:

    def test_not_fitted_initially(self):
        model = XGBoostFraudModel()
        assert not model.is_fitted

    def test_is_fitted_after_train(self, dataset):
        X_train, _, y_train, _ = dataset
        model = XGBoostFraudModel()
        model.train(X_train, y_train)
        assert model.is_fitted

    def test_predict_proba_before_fit_raises(self, dataset):
        X_train, _, _, _ = dataset
        model = XGBoostFraudModel()
        with pytest.raises(ModelNotFittedError):
            model.predict_proba(X_train)

    def test_predict_proba_returns_1d_array(self, fitted_model, dataset):
        _, X_val, _, _ = dataset
        proba = fitted_model.predict_proba(X_val)
        assert isinstance(proba, np.ndarray)
        assert proba.ndim == 1

    def test_predict_proba_shape_matches_input(self, fitted_model, dataset):
        _, X_val, _, _ = dataset
        proba = fitted_model.predict_proba(X_val)
        assert len(proba) == len(X_val)

    def test_predict_proba_in_valid_range(self, fitted_model, dataset):
        _, X_val, _, _ = dataset
        proba = fitted_model.predict_proba(X_val)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_model_learns_signal(self, dataset):
        """Model should achieve better than random AUC on a learnable dataset."""
        from sklearn.metrics import roc_auc_score
        X_train, X_val, y_train, y_val = dataset
        model = XGBoostFraudModel()
        model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        proba = model.predict_proba(X_val)
        auc = roc_auc_score(y_val, proba)
        assert auc > 0.60, f"Expected AUC > 0.60 on learnable dataset, got {auc:.3f}"

    def test_feature_importance_returns_dict(self, fitted_model):
        importance = fitted_model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_feature_importance_all_positive(self, fitted_model):
        importance = fitted_model.get_feature_importance()
        for name, val in importance.items():
            assert val >= 0.0, f"Negative importance for {name}: {val}"

    def test_save_and_load_produces_same_predictions(self, fitted_model, dataset):
        _, X_val, _, _ = dataset
        original_proba = fitted_model.predict_proba(X_val)

        with tempfile.TemporaryDirectory() as tmp_dir:
            fitted_model.save(tmp_dir)

            loaded_model = XGBoostFraudModel()
            loaded_model.load(tmp_dir)

            loaded_proba = loaded_model.predict_proba(X_val)

        np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-5)

    def test_load_from_nonexistent_path_raises(self):
        model = XGBoostFraudModel()
        with pytest.raises(FileNotFoundError):
            model.load("/nonexistent/path/that/does/not/exist")

    def test_scale_pos_weight_auto_computed(self, dataset):
        """scale_pos_weight should be automatically set from class ratio."""
        X_train, _, y_train, _ = dataset
        model = XGBoostFraudModel(hyperparams={"n_estimators": 10, "random_state": 42})
        model.train(X_train, y_train)
        # After training, scale_pos_weight should have been set
        assert "scale_pos_weight" in model._hyperparams

    def test_train_returns_metrics_dict(self, dataset):
        X_train, X_val, y_train, y_val = dataset
        model = XGBoostFraudModel(hyperparams={"n_estimators": 50, "random_state": 42})
        result = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        assert isinstance(result, dict)
        assert "best_iteration" in result

    def test_custom_hyperparams_accepted(self, dataset):
        X_train, _, y_train, _ = dataset
        custom_params = {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        model = XGBoostFraudModel(hyperparams=custom_params)
        model.train(X_train, y_train)
        assert model.is_fitted
