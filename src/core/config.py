"""
src/core/config.py
------------------
Centralised configuration using Pydantic Settings v2.

Rule: only the top-level `Settings` is a BaseSettings.
All nested config blocks are plain BaseModel so they don't
try to read from the environment independently.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        with p.open() as f:
            return yaml.safe_load(f) or {}
    return {}


# ── Nested config blocks (BaseModel, NOT BaseSettings) ────────────────────────


class RateLimitConfig(BaseModel):
    requests_per_minute: int = 1000
    burst: int = 200


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    cors_origins: list[str] = ["*"]
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class SecurityConfig(BaseModel):
    api_keys: str = "dev-key-local"
    jwt_secret: SecretStr = Field(default=SecretStr("change-me-in-production"))

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fraudguard-production"


class ModelConfig(BaseModel):
    version: str = "latest"
    artifacts_dir: str = "models/artifacts"


class FeatureStoreConfig(BaseModel):
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 1
    ttl_seconds: int = 3600
    connection_pool_size: int = 20


class KafkaConfig(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "fraudguard-inference"
    topic_transactions: str = "transactions.raw"
    topic_predictions: str = "fraud.predictions"
    topic_alerts: str = "fraud.alerts"
    auto_offset_reset: str = "latest"


class TrainingConfig(BaseModel):
    """Hyperparameters and paths used by the training pipeline."""

    test_size: float = 0.15
    validation_size: float = 0.15
    seed: int = 42
    artifacts_dir: str = "models/artifacts"
    production_aucpr_threshold: float = 0.85


# ── Top-level Settings (the only BaseSettings) ────────────────────────────────


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="dev", alias="ENV")
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_timeout: int = 30
    api_cors_origins: str = "*"
    api_rate_limit_rpm: int = 1000
    api_rate_limit_burst: int = 200

    # Security — read directly from env vars
    api_keys: str = Field(default="dev-key-local", alias="API_KEYS")
    jwt_secret: SecretStr = Field(default=SecretStr("change-me-in-production"), alias="JWT_SECRET")

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "fraudguard-production"

    # Model
    model_version: str = "latest"
    model_artifacts_dir: str = "models/artifacts"

    # Feature store (Redis)
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 1
    redis_ttl_seconds: int = 3600
    redis_pool_size: int = 20

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "fraudguard-inference"
    kafka_topic_transactions: str = "transactions.raw"
    kafka_topic_predictions: str = "fraud.predictions"
    kafka_topic_alerts: str = "fraud.alerts"

    # Training
    training_test_size: float = 0.15
    training_validation_size: float = 0.15
    training_seed: int = 42

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def log_format(self) -> str:
        return "json" if self.is_production else "console"

    # Convenience property accessors returning typed config blocks
    @property
    def api(self) -> APIConfig:
        return APIConfig(
            host=self.api_host,
            port=self.api_port,
            workers=self.api_workers,
            timeout=self.api_timeout,
            cors_origins=[o.strip() for o in self.api_cors_origins.split(",")],
            rate_limit=RateLimitConfig(
                requests_per_minute=self.api_rate_limit_rpm,
                burst=self.api_rate_limit_burst,
            ),
        )

    @property
    def security(self) -> SecurityConfig:
        return SecurityConfig(api_keys=self.api_keys, jwt_secret=self.jwt_secret)

    @property
    def mlflow(self) -> MLflowConfig:
        return MLflowConfig(
            tracking_uri=self.mlflow_tracking_uri,
            experiment_name=self.mlflow_experiment_name,
        )

    @property
    def model(self) -> ModelConfig:
        return ModelConfig(version=self.model_version, artifacts_dir=self.model_artifacts_dir)

    @property
    def feature_store(self) -> FeatureStoreConfig:
        return FeatureStoreConfig(
            redis_url=self.redis_url,
            redis_db=self.redis_db,
            ttl_seconds=self.redis_ttl_seconds,
            connection_pool_size=self.redis_pool_size,
        )

    @property
    def kafka(self) -> KafkaConfig:
        return KafkaConfig(
            bootstrap_servers=self.kafka_bootstrap_servers,
            consumer_group=self.kafka_consumer_group,
            topic_transactions=self.kafka_topic_transactions,
            topic_predictions=self.kafka_topic_predictions,
            topic_alerts=self.kafka_topic_alerts,
        )

    @property
    def training(self) -> TrainingConfig:
        return TrainingConfig(
            test_size=self.training_test_size,
            validation_size=self.training_validation_size,
            seed=self.training_seed,
            artifacts_dir=self.model_artifacts_dir,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
