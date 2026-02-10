"""API middleware: request logging and rate limiting."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()

_RATE_LIMIT_REDIS_PREFIX = "ratelimit:"
_RATE_LIMIT_WINDOW_SECONDS = 60


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
    """Rate limiting per tenant/user. CON-02: Redis-based when available, else in-process with warning."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._buckets: Dict[str, Tuple[int, datetime]] = {}
        self._lock = asyncio.Lock()
        self._redis_warning_logged = False

    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-Id") or request.headers.get("X-Tenant-ID", "default")
        redis = self._get_redis(request)
        allowed = await self._check_rate_limit(tenant_id, redis)
        if not allowed:
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        return await call_next(request)

    def _get_redis(self, request: Request):
        db = getattr(request.app.state, "db", None)
        return getattr(db, "redis", None) if db else None

    async def _check_rate_limit(self, key: str, redis) -> bool:
        if redis is not None:
            return await self._check_rate_limit_redis(key, redis)
        if not self._redis_warning_logged:
            logger.warning(
                "rate_limit_redis_unavailable",
                msg="Redis not available; rate limit is per-process only",
            )
            self._redis_warning_logged = True
        return await self._check_rate_limit_in_memory(key)

    async def _check_rate_limit_redis(self, key: str, redis) -> bool:
        rkey = f"{_RATE_LIMIT_REDIS_PREFIX}{key}"
        try:
            pipe = redis.pipeline()
            pipe.incr(rkey)
            pipe.expire(rkey, _RATE_LIMIT_WINDOW_SECONDS)
            results = await pipe.execute()
            count = results[0]
            if count > self.requests_per_minute:
                await redis.decr(rkey)
                return False
            return True
        except Exception as e:
            logger.warning("rate_limit_redis_error", key=key, error=str(e))
            return await self._check_rate_limit_in_memory(key)

    async def _check_rate_limit_in_memory(self, key: str) -> bool:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if len(self._buckets) > 10000:
                cutoff = now - timedelta(minutes=2)
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
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
