"""
src/core/exceptions.py
-----------------------
Domain-specific exception hierarchy.
All custom exceptions inherit from FraudGuardError for easy top-level catching.
"""

from __future__ import annotations


class FraudGuardError(Exception):
    """Base exception for all FraudGuard ML errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details})"


# --- Data Layer Exceptions ---

class DataIngestionError(FraudGuardError):
    """Raised when data ingestion fails (network, format, schema issues)."""


class DataValidationError(FraudGuardError):
    """Raised when data quality checks fail."""

    def __init__(self, message: str, failed_expectations: list[str] | None = None) -> None:
        super().__init__(message, {"failed_expectations": failed_expectations or []})
        self.failed_expectations = failed_expectations or []


class SchemaValidationError(DataValidationError):
    """Raised when input data doesn't match expected schema."""


# --- Feature Layer Exceptions ---

class FeatureEngineeringError(FraudGuardError):
    """Raised when feature transformation fails."""


class FeatureStoreError(FraudGuardError):
    """Raised when feature store operations fail (connection, timeout, etc.)."""


class FeatureStoreConnectionError(FeatureStoreError):
    """Raised when feature store is unreachable."""


# --- Model Layer Exceptions ---

class ModelNotFoundError(FraudGuardError):
    """Raised when a requested model version doesn't exist."""

    def __init__(self, model_name: str, version: str) -> None:
        super().__init__(
            f"Model '{model_name}' version '{version}' not found in registry",
            {"model_name": model_name, "version": version},
        )


class ModelNotFittedError(FraudGuardError):
    """Raised when attempting inference with an unfitted model."""


class ModelLoadError(FraudGuardError):
    """Raised when model artifact loading fails."""


class InferenceError(FraudGuardError):
    """Raised when model inference fails."""


class EnsembleError(FraudGuardError):
    """Raised when ensemble prediction fails (base learner disagreement, etc.)."""


# --- Training Exceptions ---

class TrainingError(FraudGuardError):
    """Raised when model training fails."""


class HyperparameterTuningError(FraudGuardError):
    """Raised when hyperparameter optimization fails."""


class InsufficientDataError(TrainingError):
    """Raised when not enough data is available for training."""

    def __init__(self, required: int, actual: int) -> None:
        super().__init__(
            f"Insufficient training data: required {required}, got {actual}",
            {"required": required, "actual": actual},
        )


# --- Monitoring Exceptions ---

class DriftDetectionError(FraudGuardError):
    """Raised when drift detection computation fails."""


class AlertingError(FraudGuardError):
    """Raised when alert dispatch fails."""


# --- API / Infrastructure Exceptions ---

class AuthenticationError(FraudGuardError):
    """Raised when API authentication fails."""


class RateLimitError(FraudGuardError):
    """Raised when request rate limit is exceeded."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            {"limit": limit, "window_seconds": window_seconds},
        )


class ConfigurationError(FraudGuardError):
    """Raised when configuration is invalid or missing required values."""


class MLflowError(FraudGuardError):
    """Raised when MLflow operations fail."""
