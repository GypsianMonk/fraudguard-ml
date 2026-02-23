"""
src/api/dependencies.py
-----------------------
Dependency injection container for FastAPI.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import joblib
import mlflow
from fastapi import Depends, Header, HTTPException, Request

from src.core.config import Settings, get_settings
from src.core.exceptions import ModelLoadError
from src.features.store import RedisFeatureStore
from src.models.ensemble import FraudEnsemble
from src.monitoring.metrics_collector import FraudMetricsCollector, get_metrics

if TYPE_CHECKING:
    from src.features.engineer import FraudFeatureEngineer

logger = logging.getLogger(__name__)


class AppContainer:
    def __init__(self) -> None:
        self._model: FraudEnsemble | None = None
        self._feature_engineer: FraudFeatureEngineer | None = None
        self._feature_store: RedisFeatureStore | None = None
        self._metrics: FraudMetricsCollector = get_metrics()
        self._model_version: str = "unknown"
        self._model_loaded: bool = False
        self._feature_store_connected: bool = False

    async def initialize(self) -> None:
        settings = get_settings()
        await self._load_model(settings)
        await self._connect_feature_store(settings)

    async def _load_model(self, settings: Settings) -> None:
        try:
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            model_name = "fraudguard-ensemble"
            version = settings.model.version
            model_uri = (
                f"models:/{model_name}/Production"
                if version == "latest"
                else f"models:/{model_name}/{version}"
            )
            logger.info("Loading model from: %s", model_uri)
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path="/tmp/fraudguard-model",
            )
            self._model = FraudEnsemble()
            self._model.load(f"{local_path}/model")

            from src.features.engineer import FraudFeatureEngineer
            self._feature_engineer = joblib.load(f"{local_path}/feature_engineer.joblib")
            self._model_version = version
            self._model_loaded = True
            logger.info("Model loaded successfully: %s", version)
        except Exception as exc:
            logger.error("Failed to load model from MLflow: %s. Using fallback.", exc)
            self._try_load_local_model()

    def _try_load_local_model(self) -> None:
        local_path = Path("models/artifacts/latest")
        if local_path.exists():
            try:
                self._model = FraudEnsemble()
                self._model.load(str(local_path / "model"))
                import joblib as jl
                self._feature_engineer = jl.load(local_path / "feature_engineer.joblib")
                self._model_version = "local-latest"
                self._model_loaded = True
                logger.info("Loaded local model from %s", local_path)
            except Exception as exc:
                logger.error("Local model load failed: %s", exc)
        else:
            logger.warning("No model available — predictions will fail.")

    async def _connect_feature_store(self, settings: Settings) -> None:
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
        if self._feature_store:
            await self._feature_store.close()

    @property
    def model(self) -> FraudEnsemble:
        if self._model is None:
            raise ModelLoadError("Model not loaded")
        return self._model

    @property
    def feature_engineer(self) -> FraudFeatureEngineer:
        if self._feature_engineer is None:
            raise ModelLoadError("Feature engineer not loaded")
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


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_model(
    container: Annotated[AppContainer, Depends(get_container)],
) -> FraudEnsemble:
    return container.model


async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    settings = get_settings()
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    if x_api_key not in settings.security.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
