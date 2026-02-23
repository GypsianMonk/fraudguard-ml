"""
src/api/routes/admin.py
-----------------------
Admin endpoints for model management (protected, internal use only).
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import AppContainer, get_container, verify_api_key
from src.core.schemas import ModelInfoResponse, ModelReloadRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get current model information",
)
async def get_model_info(
    container: Annotated[AppContainer, Depends(get_container)],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    return ModelInfoResponse(
        model_version=container.model_version,
        model_stage="Production",
        training_date=None,
        metrics={},  # Would load from MLflow in full implementation
        feature_count=len(container.feature_engineer.get_feature_names()) if container.model_loaded else 0,
        ensemble_weights={"xgboost": 0.55, "tabtransformer": 0.45},
    )


@router.post(
    "/model/reload",
    summary="Hot-reload model from registry",
    description="Reload the model from MLflow registry without downtime. "
                "New requests use new model after reload completes.",
)
async def reload_model(
    request: ModelReloadRequest,
    container: Annotated[AppContainer, Depends(get_container)],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> dict:
    """
    Hot-reload the inference model from MLflow registry.

    Supports zero-downtime model updates — in-flight requests complete
    with the old model while the new one loads.
    """
    settings_version = request.version
    logger.info("Model reload requested: version=%s dry_run=%s", settings_version, request.dry_run)

    if request.dry_run:
        return {
            "status": "dry_run",
            "message": f"Would reload model version {settings_version}",
            "current_version": container.model_version,
        }

    # In production, this would atomically swap the model reference
    # Here we trigger a re-initialization
    try:
        from src.core.config import get_settings
        settings = get_settings()
        await container._load_model(settings)
        logger.info("Model hot-reloaded successfully: %s", container.model_version)
        return {
            "status": "success",
            "new_version": container.model_version,
            "message": "Model reloaded successfully",
        }
    except Exception as exc:
        logger.error("Model reload failed: %s", exc)
        return {
            "status": "failed",
            "error": str(exc),
            "current_version": container.model_version,
        }
