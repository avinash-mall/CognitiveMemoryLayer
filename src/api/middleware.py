"""API middleware: request logging and rate limiting."""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = f"{time.time_ns()}"
        request.state.request_id = request_id

        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
            elapsed = time.time() - start_time
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                elapsed_ms=elapsed * 1000,
            )
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{elapsed * 1000:.2f}ms"
            return response
        except Exception as e:
            logger.error("request_failed", request_id=request_id, error=str(e))
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting per tenant/user."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._buckets: Dict[str, Tuple[int, datetime]] = {}
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID", "default")
        allowed = await self._check_rate_limit(tenant_id)
        if not allowed:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )
        return await call_next(request)

    async def _check_rate_limit(self, key: str) -> bool:
        async with self._lock:
            now = datetime.utcnow()
            if key in self._buckets:
                count, window_start = self._buckets[key]
                if now - window_start > timedelta(minutes=1):
                    self._buckets[key] = (1, now)
                    return True
                if count >= self.requests_per_minute:
                    return False
                self._buckets[key] = (count + 1, window_start)
                return True
            self._buckets[key] = (1, now)
            return True
