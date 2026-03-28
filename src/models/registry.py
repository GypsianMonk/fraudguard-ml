"""
src/models/registry.py
-----------------------
MLflow model registry client.
Handles model versioning, stage transitions, and artifact retrieval.

Stages: None → Staging → Production → Archived
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.core.config import get_settings
from src.core.exceptions import ModelLoadError, ModelNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "fraudguard-ensemble"


@dataclass
class ModelVersion:
    name: str
    version: str
    stage: str
    run_id: str
    creation_timestamp: datetime
    description: str | None
    metrics: dict[str, float]
    tags: dict[str, str]


class ModelRegistry:
    """
    Wrapper around MLflow Model Registry.

    Responsibilities:
    - Register new model versions after training
    - Transition versions between stages (Staging → Production)
    - Load the current Production model artifact
    - List and compare historical versions
    - Archive old versions when promoting new ones

    Usage:
        registry = ModelRegistry()
        version = registry.get_production_version("fraudguard-ensemble")
        artifact_path = registry.download_artifacts(version)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        self._client = MlflowClient()

    # ── Read operations ──────────────────────────────────────────────────────

    def get_production_version(self) -> ModelVersion | None:
        """Return the current Production model version, or None if none promoted."""
        try:
            versions = self._client.get_latest_versions(self._model_name, stages=["Production"])
            if not versions:
                logger.warning("No Production model found for '%s'", self._model_name)
                return None
            v = versions[0]
            return self._to_model_version(v)
        except MlflowException as exc:
            logger.error("Registry error fetching Production version: %s", exc)
            return None

    def get_staging_version(self) -> ModelVersion | None:
        """Return the current Staging model version."""
        try:
            versions = self._client.get_latest_versions(self._model_name, stages=["Staging"])
            return self._to_model_version(versions[0]) if versions else None
        except MlflowException as exc:
            logger.error("Registry error fetching Staging version: %s", exc)
            return None

    def get_version(self, version: str) -> ModelVersion:
        """Return a specific model version by version string."""
        try:
            v = self._client.get_model_version(self._model_name, version)
            return self._to_model_version(v)
        except MlflowException as exc:
            raise ModelNotFoundError(self._model_name, version) from exc

    def list_versions(self, stages: list[str] | None = None) -> list[ModelVersion]:
        """List all model versions, optionally filtered by stage."""
        try:
            filter_str = f"name='{self._model_name}'"
            versions = self._client.search_model_versions(filter_str)
            result = [self._to_model_version(v) for v in versions]
            if stages:
                result = [v for v in result if v.stage in stages]
            return sorted(result, key=lambda v: v.creation_timestamp, reverse=True)
        except MlflowException as exc:
            logger.error("Failed to list model versions: %s", exc)
            return []

    # ── Write operations ──────────────────────────────────────────────────────

    def register(
        self,
        run_id: str,
        artifact_path: str = "model",
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        """
        Register a model from an MLflow run into the registry.

        Args:
            run_id: MLflow run ID containing the model artifact
            artifact_path: Artifact path within the run (default: 'model')
            description: Human-readable description of this version
            tags: Key-value tags for this version

        Returns:
            Registered ModelVersion
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info("Registering model from run '%s' as '%s'", run_id, self._model_name)

        try:
            # Ensure registered model exists
            try:
                self._client.get_registered_model(self._model_name)
            except MlflowException:
                self._client.create_registered_model(
                    self._model_name,
                    description=f"FraudGuard ML ensemble model — {self._model_name}",
                )

            result = mlflow.register_model(model_uri=model_uri, name=self._model_name)
            version = result.version

            # Set description and tags
            if description:
                self._client.update_model_version(
                    name=self._model_name,
                    version=version,
                    description=description,
                )
            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(
                        name=self._model_name, version=version, key=key, value=value
                    )

            logger.info("Registered '%s' version %s", self._model_name, version)
            return self.get_version(version)

        except MlflowException as exc:
            msg = f"Model registration failed: {exc}"
            raise ModelLoadError(msg) from exc

    def promote_to_staging(self, version: str) -> ModelVersion:
        """Transition a version to Staging stage."""
        return self._transition(version, "Staging", archive_existing=True)

    def promote_to_production(self, version: str) -> ModelVersion:
        """
        Promote a version to Production, archiving the previous Production version.
        This is the gating operation — only call after smoke tests pass.
        """
        return self._transition(version, "Production", archive_existing=True)

    def archive(self, version: str) -> ModelVersion:
        """Archive a model version (remove from Staging/Production)."""
        return self._transition(version, "Archived", archive_existing=False)

    def promote_staging_to_production(self) -> ModelVersion | None:
        """
        Convenience: promote the current Staging version to Production.
        Returns None if no Staging version exists.
        """
        staging = self.get_staging_version()
        if staging is None:
            logger.warning("No Staging version to promote")
            return None
        logger.info("Promoting Staging v%s to Production", staging.version)
        return self.promote_to_production(staging.version)

    # ── Artifact retrieval ────────────────────────────────────────────────────

    def download_artifacts(
        self,
        version: ModelVersion | None = None,
        dst_path: str = "/tmp/fraudguard-model",
    ) -> str:
        """
        Download model artifacts to local path.

        Args:
            version: ModelVersion to download (uses Production if None)
            dst_path: Local destination directory

        Returns:
            Path to downloaded artifacts
        """
        if version is None:
            version = self.get_production_version()
            if version is None:
                msg = "No Production model available to download"
                raise ModelLoadError(msg)

        model_uri = f"models:/{self._model_name}/{version.version}"
        logger.info("Downloading artifacts: %s → %s", model_uri, dst_path)

        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path)
        logger.info("Artifacts downloaded to %s", local_path)
        return local_path

    # ── Comparison helpers ────────────────────────────────────────────────────

    def compare_versions(self, versions: list[str]) -> list[dict[str, Any]]:
        """
        Compare metrics across multiple model versions.

        Returns list of dicts sorted by AUC-PR descending.
        """
        rows = []
        for v_str in versions:
            try:
                v = self.get_version(v_str)
                rows.append(
                    {
                        "version": v.version,
                        "stage": v.stage,
                        "auc_pr": v.metrics.get("test_auc_pr", 0.0),
                        "auc_roc": v.metrics.get("test_auc_roc", 0.0),
                        "recall_optimal": v.metrics.get("test_recall_optimal", 0.0),
                        "precision_optimal": v.metrics.get("test_precision_optimal", 0.0),
                        "created": v.creation_timestamp.isoformat(),
                    }
                )
            except Exception as exc:
                logger.warning("Could not load version %s: %s", v_str, exc)

        return sorted(rows, key=lambda r: r["auc_pr"], reverse=True)

    def get_best_version_by_metric(
        self,
        metric: str = "test_auc_pr",
        stages: list[str] | None = None,
    ) -> ModelVersion | None:
        """Return the version with the highest value for a given metric."""
        versions = self.list_versions(stages=stages)
        if not versions:
            return None

        best = max(versions, key=lambda v: v.metrics.get(metric, 0.0))
        logger.info(
            "Best version by %s: v%s (%.4f)",
            metric,
            best.version,
            best.metrics.get(metric, 0.0),
        )
        return best

    # ── Private helpers ───────────────────────────────────────────────────────

    def _transition(
        self,
        version: str,
        stage: str,
        archive_existing: bool,
    ) -> ModelVersion:
        """Transition a model version to a target stage."""
        logger.info(
            "Transitioning '%s' v%s → %s (archive_existing=%s)",
            self._model_name,
            version,
            stage,
            archive_existing,
        )
        try:
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            logger.info("Transition complete: '%s' v%s → %s", self._model_name, version, stage)
            return self.get_version(version)
        except MlflowException as exc:
            msg = f"Stage transition failed for v{version} → {stage}: {exc}"
            raise ModelLoadError(msg) from exc

    def _to_model_version(self, v: Any) -> ModelVersion:
        """Convert MLflow ModelVersion to our typed dataclass."""
        # Fetch run metrics if available
        metrics: dict[str, float] = {}
        try:
            if v.run_id:
                run = self._client.get_run(v.run_id)
                metrics = {k: float(val) for k, val in run.data.metrics.items()}
        except Exception:
            pass

        return ModelVersion(
            name=v.name,
            version=str(v.version),
            stage=v.current_stage,
            run_id=v.run_id or "",
            creation_timestamp=datetime.fromtimestamp(v.creation_timestamp / 1000, tz=UTC),
            description=v.description,
            metrics=metrics,
            tags={t.key: t.value for t in (v.tags or [])},
        )


def get_registry(model_name: str = DEFAULT_MODEL_NAME) -> ModelRegistry:
    """Factory function for DI / testing."""
    return ModelRegistry(model_name)


def save_model_artifact(obj: Any, path: str | Path) -> None:
    """Save any Python object as a joblib artifact."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Artifact saved → %s (%.1f KB)", path, path.stat().st_size / 1024)


def load_model_artifact(path: str | Path) -> Any:
    """Load a joblib artifact from path."""
    import joblib

    path = Path(path)
    if not path.exists():
        msg = f"Artifact not found: {path}"
        raise FileNotFoundError(msg)
    return joblib.load(path)
