"""
src/core/interfaces.py
-----------------------
Abstract base classes (interfaces) for all major components.
Implements Dependency Inversion Principle — high-level modules depend on abstractions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class BaseModel(ABC):
    """Abstract interface for all ML models."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool: ...

    @abstractmethod
    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


class BaseFeatureEngineer(ABC):
    """Abstract interface for feature engineering pipelines."""

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def get_feature_names(self) -> list[str]: ...


class BaseFeatureStore(ABC):
    """Abstract interface for feature stores."""

    @abstractmethod
    async def get_user_features(self, user_id: str) -> dict[str, Any]: ...

    @abstractmethod
    async def update_user_features(
        self, user_id: str, features: dict[str, Any]
    ) -> None: ...

    @abstractmethod
    async def ping(self) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...


class BaseDataIngester(ABC):
    """Abstract interface for data ingestion sources."""

    @abstractmethod
    def ingest(self, source: str, **kwargs: Any) -> pd.DataFrame: ...

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool: ...


class BaseMetricsCollector(ABC):
    """Abstract interface for metrics collection."""

    @abstractmethod
    def record_prediction(
        self, fraud_prob: float, risk_tier: str, latency_ms: int
    ) -> None: ...

    @abstractmethod
    def record_error(self, error_type: str) -> None: ...
