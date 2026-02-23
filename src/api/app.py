"""
src/api/app.py
--------------
FastAPI application factory with all middleware, routers, and lifecycle hooks.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.api.dependencies import AppContainer
from src.api.middleware import LoggingMiddleware, RateLimitMiddleware
from src.api.routes import admin, health, predict
from src.core.config import get_settings
from src.core.exceptions import (
    AuthenticationError,
    FraudGuardError,
    InferenceError,
    RateLimitError,
)
from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, format_=settings.log_format)
    logger.info("FraudGuard ML starting up (env=%s)", settings.environment)

    container = AppContainer()
    await container.initialize()
    app.state.container = container

    logger.info("Application ready. Model version: %s", container.model_version)
    yield

    logger.info("FraudGuard ML shutting down")
    await container.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="FraudGuard ML API",
        description="Real-time transaction fraud detection API.",
        version="2.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.api.rate_limit.requests_per_minute,
        burst=settings.api.rate_limit.burst,
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(_: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=401, content={"error": "Unauthorized", "detail": exc.message}
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(_: Request, exc: RateLimitError) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": exc.message},
            headers={"Retry-After": "60"},
        )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(_: Request, exc: InferenceError) -> JSONResponse:
        logger.error("Inference error: %s | details=%s", exc.message, exc.details)
        return JSONResponse(status_code=500, content={"error": "Inference failed"})

    @app.exception_handler(FraudGuardError)
    async def fraudguard_error_handler(_: Request, exc: FraudGuardError) -> JSONResponse:
        logger.error("Application error: %s", exc.message)
        return JSONResponse(status_code=500, content={"error": exc.message})

    @app.exception_handler(Exception)
    async def generic_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    return app


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, format_=settings.log_format)
    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers if settings.is_production else 1,
        loop="uvloop",
        log_config=None,
        access_log=False,
        timeout_keep_alive=settings.api.timeout,
    )


app = create_app()

if __name__ == "__main__":
    main()
