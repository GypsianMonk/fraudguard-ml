"""
src/api/routes/health.py
-------------------------
Kubernetes-compatible health and readiness probes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import AppContainer, get_container
from src.core.schemas import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health_check() -> HealthResponse:
    """Returns 200 if the process is alive."""
    return HealthResponse(status="ok", timestamp=datetime.now(tz=timezone.utc))


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness_check(
    container: Annotated[AppContainer, Depends(get_container)],
) -> JSONResponse:
    """Returns 200 only if the service is ready to serve traffic."""
    import mlflow

    is_ready = container.model_loaded
    status_code = 200 if is_ready else 503

    mlflow_connected = True
    try:
        mlflow.search_experiments(max_results=1)
    except Exception:
        mlflow_connected = False

    response = ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        model_loaded=container.model_loaded,
        model_version=container.model_version if container.model_loaded else None,
        feature_store_connected=container.feature_store_connected,
        mlflow_connected=mlflow_connected,
        timestamp=datetime.now(tz=timezone.utc),
    )
    return JSONResponse(content=response.model_dump(mode="json"), status_code=status_code)
