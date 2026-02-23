"""
src/core/interfaces.py
----------------------
Abstract base classes defining contracts for all major components.
Follows the Dependency Inversion Principle — high-level modules depend on
abstractions, not concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core.schemas import PredictionResponse, TransactionRequest


class BaseModel(ABC):
    """Abstract interface for all ML models."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        """Train the model on the given dataset."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability scores. Shape: (n_samples,)"""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model artifacts to path."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model artifacts from path."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature name → importance score mapping."""
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if model has been trained."""
        ...


class BaseFeatureEngineer(ABC):
    """Abstract interface for feature engineering pipelines."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseFeatureEngineer:
        """Fit feature transformers on training data."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw transaction data into model features."""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class BaseDataIngester(ABC):
    """Abstract interface for data ingestion strategies."""

    @abstractmethod
    def ingest(self, source: str, **kwargs: Any) -> pd.DataFrame:
        """Ingest data from a source and return a DataFrame."""
        ...

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate the ingested data matches expected schema."""
        ...


class BaseModelRegistry(ABC):
    """Abstract interface for model versioning and registry operations."""

    @abstractmethod
    def register_model(
        self,
        model_name: str,
        model_uri: str,
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> str:
        """Register a trained model and return its version."""
        ...

    @abstractmethod
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load a model from the registry by name and version."""
        ...

    @abstractmethod
    def promote_model(self, model_name: str, version: str, stage: str) -> None:
        """Promote model version to a lifecycle stage (Staging/Production)."""
        ...

    @abstractmethod
    def get_latest_version(self, model_name: str, stage: str = "Production") -> str:
        """Return the latest model version for a given stage."""
        ...


class BaseFeatureStore(ABC):
    """Abstract interface for feature store operations."""

    @abstractmethod
    def get_user_features(self, user_id: str) -> dict[str, Any]:
        """Retrieve precomputed user-level features."""
        ...

    @abstractmethod
    def get_merchant_features(self, merchant_id: str) -> dict[str, Any]:
        """Retrieve precomputed merchant-level features."""
        ...

    @abstractmethod
    def update_user_features(self, user_id: str, features: dict[str, Any]) -> None:
        """Update user features after a transaction."""
        ...

    @abstractmethod
    def get_velocity_features(self, user_id: str, windows: list[int]) -> dict[str, float]:
        """Compute real-time velocity features for given time windows (minutes)."""
        ...


class BaseDriftDetector(ABC):
    """Abstract interface for data/concept drift detection."""

    @abstractmethod
    def fit_reference(self, reference_data: pd.DataFrame) -> None:
        """Set the reference distribution from training/historical data."""
        ...

    @abstractmethod
    def detect_drift(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """Compare current data distribution against reference. Returns drift report."""
        ...

    @abstractmethod
    def is_drifted(self, current_data: pd.DataFrame) -> bool:
        """Return True if statistically significant drift is detected."""
        ...
