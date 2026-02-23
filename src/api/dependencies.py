"""
src/api/dependencies.py
-----------------------
Dependency injection container for FastAPI.
Manages singleton instances of model, feature store, and metrics collector.
Provides FastAPI Depends() functions for route injection.
"""

from __future__ import annotations

import logging
from typing import Annotated

import joblib
import mlflow
from fastapi import Depends, Header, HTTPException, Request

from src.core.config import Settings, get_settings
from src.core.exceptions import AuthenticationError, ModelLoadError
from src.features.engineer import FraudFeatureEngineer
from src.features.store import RedisFeatureStore
from src.models.ensemble import FraudEnsemble
from src.monitoring.metrics_collector import FraudMetricsCollector, get_metrics

logger = logging.getLogger(__name__)


class AppContainer:
    """
    Application-level dependency container.
    Single instance shared across all requests (thread-safe reads).

    Manages:
    - Ensemble model (loaded from MLflow registry)
    - Feature engineer (loaded alongside model)
    - Feature store (Redis connection pool)
    - Metrics collector (Prometheus)
    """

    def __init__(self) -> None:
        self._model: FraudEnsemble | None = None
        self._feature_engineer: FraudFeatureEngineer | None = None
        self._feature_store: RedisFeatureStore | None = None
        self._metrics: FraudMetricsCollector = get_metrics()
        self._model_version: str = "unknown"
        self._model_loaded: bool = False
        self._feature_store_connected: bool = False

    async def initialize(self) -> None:
        """Initialize all components on startup."""
        settings = get_settings()

        # Load model from MLflow registry
        await self._load_model(settings)

        # Connect to feature store
        await self._connect_feature_store(settings)

    async def _load_model(self, settings: Settings) -> None:
        """Load ensemble model and feature engineer from MLflow."""
        try:
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            model_name = "fraudguard-ensemble"
            version = settings.model.version

            if version == "latest":
                model_uri = f"models:/{model_name}/Production"
            else:
                model_uri = f"models:/{model_name}/{version}"

            logger.info("Loading model from: %s", model_uri)

            # Load ensemble
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path="/tmp/fraudguard-model",
            )

            self._model = FraudEnsemble()
            self._model.load(f"{local_path}/model")

            # Load feature engineer
            self._feature_engineer = joblib.load(f"{local_path}/feature_engineer.joblib")

            self._model_version = version
            self._model_loaded = True

            logger.info("Model loaded successfully: %s", version)

        except Exception as exc:
            logger.error("Failed to load model from MLflow: %s. Using fallback.", exc)
            # In development, try loading from local artifacts
            self._try_load_local_model()

    def _try_load_local_model(self) -> None:
        """Fallback: load model from local artifacts directory."""
        import os
        local_path = "models/artifacts/latest"

        if os.path.exists(local_path):
            try:
                self._model = FraudEnsemble()
                self._model.load(f"{local_path}/model")
                self._feature_engineer = joblib.load(f"{local_path}/feature_engineer.joblib")
                self._model_version = "local-latest"
                self._model_loaded = True
                logger.info("Loaded local model from %s", local_path)
            except Exception as exc:
                logger.error("Local model load failed: %s", exc)
        else:
            logger.warning("No model available. Predictions will fail until a model is loaded.")

    async def _connect_feature_store(self, settings: Settings) -> None:
        """Initialize Redis feature store connection pool."""
        try:
            self._feature_store = RedisFeatureStore(
                redis_url=settings.feature_store.redis_url,
                db=settings.feature_store.redis_db,
                ttl_seconds=settings.feature_store.ttl_seconds,
                pool_size=settings.feature_store.connection_pool_size,
            )
            await self._feature_store.ping()
            self._feature_store_connected = True
            logger.info("Feature store connected")
        except Exception as exc:
            logger.error("Feature store connection failed: %s", exc)
            self._feature_store_connected = False

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._feature_store:
            await self._feature_store.close()

    # --- Public accessors ---

    @property
    def model(self) -> FraudEnsemble:
        if self._model is None:
            msg = "Model not loaded"
            raise ModelLoadError(msg)
        return self._model

    @property
    def feature_engineer(self) -> FraudFeatureEngineer:
        if self._feature_engineer is None:
            msg = "Feature engineer not loaded"
            raise ModelLoadError(msg)
        return self._feature_engineer

    @property
    def feature_store(self) -> RedisFeatureStore | None:
        return self._feature_store

    @property
    def metrics(self) -> FraudMetricsCollector:
        return self._metrics

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def feature_store_connected(self) -> bool:
        return self._feature_store_connected


# --- FastAPI Dependency Functions ---

def get_container(request: Request) -> AppContainer:
    """Get the app container from request state."""
    return request.app.state.container


def get_model(container: Annotated[AppContainer, Depends(get_container)]) -> FraudEnsemble:
    """Inject the loaded ensemble model."""
    return container.model


def get_feature_engineer(
    container: Annotated[AppContainer, Depends(get_container)],
) -> FraudFeatureEngineer:
    """Inject the fitted feature engineer."""
    return container.feature_engineer


def get_feature_store(
    container: Annotated[AppContainer, Depends(get_container)],
) -> RedisFeatureStore | None:
    """Inject the feature store (may be None if Redis unavailable)."""
    return container.feature_store


async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(get_settings),
) -> str:
    """Validate API key from request header."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    if x_api_key not in settings.security.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key
