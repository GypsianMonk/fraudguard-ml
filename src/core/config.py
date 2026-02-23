"""
src/core/config.py
------------------
Config-driven architecture using Pydantic Settings v2.
Loads from YAML base config, then overrides with environment variables.

Environment variables take precedence over YAML config.
YAML config takes precedence over Pydantic field defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_BASE_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "base.yaml"


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML base config, expanding environment variable references."""
    if not _BASE_CONFIG_PATH.exists():
        return {}
    with _BASE_CONFIG_PATH.open() as f:
        content = f.read()
    # Simple env var expansion: ${VAR:default}
    import re
    def replace_env(match: re.Match) -> str:
        var_name, _, default = match.group(1).partition(":")
        return os.environ.get(var_name.strip(), default.strip())
    content = re.sub(r"\$\{([^}]+)\}", replace_env, content)
    return yaml.safe_load(content) or {}


class APIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    debug: bool = False

    model_config = SettingsConfigDict(env_prefix="API_")


class ModelConfig(BaseSettings):
    version: str = "latest"
    artifact_path: str = "models/artifacts"
    xgboost_weight: float = 0.45
    tabtransformer_weight: float = 0.40
    meta_learner_weight: float = 0.15
    low_risk_threshold: float = 0.30
    medium_risk_threshold: float = 0.60
    high_risk_threshold: float = 0.85
    calibration_enabled: bool = True
    calibration_method: str = "isotonic"

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    @field_validator("calibration_method")
    @classmethod
    def validate_calibration(cls, v: str) -> str:
        allowed = {"isotonic", "platt"}
        if v not in allowed:
            msg = f"calibration_method must be one of {allowed}"
            raise ValueError(msg)
        return v


class FeatureStoreConfig(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    ttl_seconds: int = 86400
    batch_size: int = 500
    connection_pool_size: int = 20

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class MLflowConfig(BaseSettings):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fraudguard-production"
    artifact_store: str = "./mlruns"
    registry_uri: str = "http://localhost:5000"

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")


class KafkaConfig(BaseSettings):
    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "fraudguard-consumer"
    topic_transactions: str = "transactions.raw"
    topic_predictions: str = "fraud.predictions"
    topic_alerts: str = "fraud.alerts"
    auto_offset_reset: str = "latest"
    max_poll_records: int = 500

    model_config = SettingsConfigDict(env_prefix="KAFKA_")


class TrainingConfig(BaseSettings):
    seed: int = 42
    test_size: float = 0.15
    validation_size: float = 0.10
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    eval_metric: str = "aucpr"
    target_column: str = "is_fraud"
    artifacts_dir: str = "models/artifacts"
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    data_features_path: str = "data/features"

    model_config = SettingsConfigDict(env_prefix="TRAINING_")


class MonitoringConfig(BaseSettings):
    metrics_port: int = 9090
    drift_detection_enabled: bool = True
    drift_check_interval_minutes: int = 60
    psi_threshold: float = 0.25
    ks_threshold: float = 0.05
    reference_window_days: int = 30
    pagerduty_key: str = ""
    slack_webhook: str = ""

    model_config = SettingsConfigDict(env_prefix="MONITORING_")


class SecurityConfig(BaseSettings):
    api_key_header: str = "X-API-Key"
    api_keys: str = "dev-key-local"  # Comma-separated
    jwt_secret: str = "dev-secret-change-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


class Settings(BaseSettings):
    """Root application settings."""

    environment: str = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = "json"  # json | console

    api: APIConfig = Field(default_factory=APIConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def load_yaml_overrides(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Merge YAML base config into values (env vars override YAML)."""
        yaml_config = _load_yaml_config()
        # Flatten top-level yaml sections
        merged = {**yaml_config.get("app", {}), **values}
        return merged

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"

    @property
    def is_development(self) -> bool:
        return self.environment == "dev"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Reset cache in tests with get_settings.cache_clear()."""
    return Settings()
