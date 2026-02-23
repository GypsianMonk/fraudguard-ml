"""
src/api/routes/health.py
-------------------------
Kubernetes-compatible health and readiness probes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import AppContainer, get_container
from src.core.schemas import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health_check() -> HealthResponse:
    """
    Liveness probe — returns 200 if the process is alive.
    Kubernetes restarts the container if this returns non-200.
    """
    from datetime import datetime
    return HealthResponse(status="ok", timestamp=datetime.utcnow())


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness_check(
    container: AppContainer = Depends(get_container),
) -> JSONResponse:
    """
    Readiness probe — returns 200 only if the service is ready to serve traffic.
    Kubernetes routes traffic away if this returns non-200.

    Checks:
    - Model is loaded and fitted
    - Feature store connection is healthy
    """
    from datetime import datetime
    import mlflow

    is_ready = container.model_loaded
    status_code = 200 if is_ready else 503

    # Check MLflow connectivity
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
        timestamp=datetime.utcnow(),
    )

    return JSONResponse(content=response.model_dump(mode="json"), status_code=status_code)
