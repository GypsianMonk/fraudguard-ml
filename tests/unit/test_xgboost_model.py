"""
tests/unit/test_xgboost_model.py
---------------------------------
Unit tests for XGBoostFraudModel.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import ModelNotFittedError
from src.models.xgboost_model import XGBoostFraudModel


def make_fraud_dataset(
    n: int = 500, fraud_rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n_fraud = max(10, int(n * fraud_rate))
    n_legit = n - n_fraud
    offset = np.array([2, -1, 0.5, 1, -0.5, 0.3, 1.2, -0.8, 0.6, -0.3])
    fraud_feats = rng.standard_normal((n_fraud, 10)) + offset
    legit_feats = rng.standard_normal((n_legit, 10))
    features = pd.DataFrame(
        np.vstack([fraud_feats, legit_feats]),
        columns=[f"feature_{i}" for i in range(10)],
    )
    labels = pd.Series([1] * n_fraud + [0] * n_legit, name="is_fraud")
    idx = rng.permutation(len(labels))
    return features.iloc[idx].reset_index(drop=True), labels.iloc[idx].reset_index(drop=True)


@pytest.fixture
def dataset():
    features, labels = make_fraud_dataset(n=500)
    split = int(len(features) * 0.8)
    return features[:split], features[split:], labels[:split], labels[split:]


@pytest.fixture
def fitted_model(dataset):
    x_train, x_val, y_train, y_val = dataset
    model = XGBoostFraudModel()
    model.train(x_train, y_train, X_val=x_val, y_val=y_val)
    return model


class TestXGBoostFraudModel:

    def test_not_fitted_initially(self):
        assert not XGBoostFraudModel().is_fitted

    def test_is_fitted_after_train(self, dataset):
        x_train, _, y_train, _ = dataset
        model = XGBoostFraudModel()
        model.train(x_train, y_train)
        assert model.is_fitted

    def test_predict_proba_before_fit_raises(self, dataset):
        x_train, _, _, _ = dataset
        with pytest.raises(ModelNotFittedError):
            XGBoostFraudModel().predict_proba(x_train)

    def test_predict_proba_returns_1d_array(self, fitted_model, dataset):
        _, x_val, _, _ = dataset
        proba = fitted_model.predict_proba(x_val)
        assert isinstance(proba, np.ndarray)
        assert proba.ndim == 1

    def test_predict_proba_shape_matches_input(self, fitted_model, dataset):
        _, x_val, _, _ = dataset
        assert len(fitted_model.predict_proba(x_val)) == len(x_val)

    def test_predict_proba_in_valid_range(self, fitted_model, dataset):
        _, x_val, _, _ = dataset
        proba = fitted_model.predict_proba(x_val)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_model_learns_signal(self, dataset):
        from sklearn.metrics import roc_auc_score
        x_train, x_val, y_train, y_val = dataset
        model = XGBoostFraudModel()
        model.train(x_train, y_train, X_val=x_val, y_val=y_val)
        auc = roc_auc_score(y_val, model.predict_proba(x_val))
        assert auc > 0.60, f"AUC {auc:.3f} < 0.60"

    def test_feature_importance_returns_dict(self, fitted_model):
        importance = fitted_model.get_feature_importance()
        assert isinstance(importance, dict) and len(importance) > 0

    def test_feature_importance_all_positive(self, fitted_model):
        for name, val in fitted_model.get_feature_importance().items():
            assert val >= 0.0, f"Negative importance for {name}"

    def test_save_and_load_same_predictions(self, fitted_model, dataset):
        _, x_val, _, _ = dataset
        original = fitted_model.predict_proba(x_val)
        with tempfile.TemporaryDirectory() as tmp:
            fitted_model.save(tmp)
            loaded = XGBoostFraudModel()
            loaded.load(tmp)
        np.testing.assert_allclose(original, loaded.predict_proba(x_val), rtol=1e-5)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            XGBoostFraudModel().load("/nonexistent/path")

    def test_custom_hyperparams_accepted(self, dataset):
        x_train, _, y_train, _ = dataset
        model = XGBoostFraudModel(hyperparams={"n_estimators": 50, "random_state": 42})
        model.train(x_train, y_train)
        assert model.is_fitted
