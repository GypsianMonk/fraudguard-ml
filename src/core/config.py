"""
src/core/config.py
------------------
Centralised configuration using Pydantic Settings v2.
Reads from YAML files + environment variables (env vars take precedence).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        with p.open() as f:
            return yaml.safe_load(f) or {}
    return {}


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    url: str = "redis://localhost:6379"
    db: int = 0


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_")
    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "fraudguard-inference"
    topic_transactions: str = "transactions.raw"
    topic_predictions: str = "fraud.predictions"
    topic_alerts: str = "fraud.alerts"
    auto_offset_reset: str = "latest"


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fraudguard-production"


class RateLimitSettings(BaseSettings):
    requests_per_minute: int = 1000
    burst: int = 200


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    cors_origins: list[str] = ["*"]
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)


class SecuritySettings(BaseSettings):
    api_keys: str = Field(default="dev-key-local", alias="API_KEYS")
    jwt_secret: SecretStr = Field(
        default=SecretStr("change-me-in-production"), alias="JWT_SECRET"
    )

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_")
    version: str = "latest"
    artifacts_dir: str = "models/artifacts"


class FeatureStoreSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 1
    ttl_seconds: int = 3600
    connection_pool_size: int = 20

    @field_validator("redis_url", mode="before")
    @classmethod
    def from_redis_url(cls, v: str) -> str:
        return os.environ.get("REDIS_URL", v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="dev", alias="ENV")
    log_level: str = "INFO"

    api: APISettings = Field(default_factory=APISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    feature_store: FeatureStoreSettings = Field(default_factory=FeatureStoreSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def log_format(self) -> str:
        return "json" if self.is_production else "console"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
