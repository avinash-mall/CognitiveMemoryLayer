"""API middleware: request logging and rate limiting."""

import asyncio
import hashlib
import time
from datetime import UTC, datetime, timedelta

import structlog
from cachetools import TTLCache  # type: ignore[import-untyped]
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()

_RATE_LIMIT_REDIS_PREFIX = "ratelimit:"
_RATE_LIMIT_WINDOW_SECONDS = 60


_REQUEST_COUNT_PREFIX = "dashboard:reqcount:"
_REQUEST_COUNT_TTL = 48 * 3600  # 48h: covers 2 full day/night dashboard cycles


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and lightweight request counting."""

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

            # Lightweight request counter in Redis (fire-and-forget)
            await self._increment_request_counter(request)

            return response
        except Exception as e:
            logger.error("request_failed", request_id=request_id, error=str(e))
            raise

    @staticmethod
    async def _increment_request_counter(request: Request) -> None:
        """Increment hourly request counter in Redis for dashboard stats."""
        try:
            db = getattr(request.app.state, "db", None)
            redis_client = getattr(db, "redis", None) if db else None
            if redis_client is None:
                return
            hour_key = datetime.now(UTC).strftime("%Y-%m-%d-%H")
            rkey = f"{_REQUEST_COUNT_PREFIX}{hour_key}"
            pipe = redis_client.pipeline()
            pipe.incr(rkey)
            pipe.expire(rkey, _REQUEST_COUNT_TTL)
            await pipe.execute()
        except Exception as e:
            logger.debug("request_counter_increment_failed", error=str(e))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting per tenant/user. CON-02: Redis-based when available, else in-process with warning."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._buckets: TTLCache[str, tuple[int, datetime]] = TTLCache(
            maxsize=10000,
            ttl=120,  # 2 min TTL for in-memory fallback
        )
        self._lock = asyncio.Lock()
        self._redis_warning_logged = False

    async def dispatch(self, request: Request, call_next):
        # Key rate-limit from the API key (authenticated credential) rather than
        # the spoofable X-Tenant-Id header.  Fall back to client IP.
        api_key = request.headers.get("X-API-Key")
        if api_key:
            key = f"apikey:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        else:
            key = f"ip:{request.client.host if request.client else 'unknown'}"
        redis = self._get_redis(request)
        allowed = await self._check_rate_limit(key, redis)
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
            now = datetime.now(UTC)
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
