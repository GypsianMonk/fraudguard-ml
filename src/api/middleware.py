"""
src/api/middleware.py
---------------------
FastAPI middleware for logging, rate limiting, and request tracing.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog.contextvars as cv
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from fastapi import Request, Response
    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        cv.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled exception in %s %s", request.method, request.url.path
            )
            raise
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            cv.bind_contextvars(latency_ms=elapsed_ms)

        logger.info(
            "Request completed",
            extra={"status_code": response.status_code, "latency_ms": elapsed_ms},
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(elapsed_ms)
        cv.clear_contextvars()
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 1000,
        burst: int = 200,
    ) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._burst = burst
        self._counts: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in {"/health", "/ready", "/metrics"}:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = 60.0

        self._counts[client_ip] = [
            t for t in self._counts[client_ip] if now - t < window
        ]

        if len(self._counts[client_ip]) >= self._rpm:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": 60},
                headers={"Retry-After": "60"},
            )

        self._counts[client_ip].append(now)
        return await call_next(request)
