"""Unit tests for API middleware (request logging, rate limiting)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware


def test_request_logging_middleware_sets_headers():
    """RequestLoggingMiddleware sets X-Request-ID and X-Response-Time on response."""
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    app.add_middleware(RequestLoggingMiddleware)
    with TestClient(app) as client:
        resp = client.get("/ping")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert resp.headers["X-Request-ID"]
    assert "X-Response-Time" in resp.headers
    assert "ms" in resp.headers["X-Response-Time"]


def test_rate_limit_middleware_in_memory_allows_under_limit():
    """RateLimitMiddleware (in-memory) returns 200 when under limit."""
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, requests_per_minute=2)
    # No app.state.db -> Redis None -> in-memory
    with TestClient(app) as client:
        r1 = client.get("/ok", headers={"X-Tenant-Id": "t1"})
        r2 = client.get("/ok", headers={"X-Tenant-Id": "t1"})
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_rate_limit_middleware_in_memory_returns_429_when_exceeded():
    """RateLimitMiddleware (in-memory) returns 429 when exceeding requests_per_minute."""
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, requests_per_minute=2)
    with TestClient(app) as client:
        r1 = client.get("/ok", headers={"X-Tenant-Id": "t2"})
        r2 = client.get("/ok", headers={"X-Tenant-Id": "t2"})
        r3 = client.get("/ok", headers={"X-Tenant-Id": "t2"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert r3.json().get("detail") == "Rate limit exceeded. Try again later."


def test_rate_limit_middleware_isolates_by_tenant():
    """Rate limit is per X-API-Key; different keys have separate buckets."""
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, requests_per_minute=2)
    with TestClient(app) as client:
        # Use different API keys to get separate rate limit buckets
        for i in range(3):
            r = client.get("/ok", headers={"X-API-Key": "key-tenant-a"})
            assert r.status_code == 200 if i < 2 else 429
        for i in range(2):
            r = client.get("/ok", headers={"X-API-Key": "key-tenant-b"})
            assert r.status_code == 200
