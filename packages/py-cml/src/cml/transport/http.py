"""HTTP transport layer using httpx."""

from __future__ import annotations

import contextlib
import time
from typing import Any, cast

import httpx

from cml._version import __version__
from cml.config import CMLConfig
from cml.exceptions import (
    AuthenticationError,
    AuthorizationError,
    CMLError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from cml.transport.retry import retry_async, retry_sync
from cml.utils.logging import logger

API_PREFIX = "/api/v1"


def _raise_for_status(response: httpx.Response) -> None:
    """Map HTTP status codes to CML exceptions with actionable suggestions."""
    if response.is_success:
        return
    body: dict[str, Any] | None = None
    with contextlib.suppress(Exception):
        body = response.json()
    request_id: str | None = response.headers.get("X-Request-ID")
    msg = f"HTTP {response.status_code}"
    if isinstance(body, dict) and "detail" in body:
        msg = str(body["detail"])
    if response.status_code == 401:
        raise AuthenticationError(
            msg,
            status_code=401,
            response_body=body,
            request_id=request_id,
            suggestion="Set CML_API_KEY env var or pass api_key= to constructor",
        )
    if response.status_code == 403:
        raise AuthorizationError(
            msg,
            status_code=403,
            response_body=body,
            request_id=request_id,
            suggestion="This operation may require admin API key. Set CML_ADMIN_API_KEY.",
        )
    if response.status_code == 404:
        raise NotFoundError(
            msg,
            status_code=404,
            response_body=body,
            request_id=request_id,
            suggestion="Check that the CML server version supports this endpoint",
        )
    if response.status_code == 422:
        raise ValidationError(
            msg,
            status_code=422,
            response_body=body,
            request_id=request_id,
            suggestion="Check request parameters match the API schema",
        )
    if response.status_code == 429:
        retry_after_header = response.headers.get("Retry-After")
        retry_after = float(retry_after_header) if retry_after_header else None
        raise RateLimitError(
            msg,
            status_code=429,
            response_body=body,
            retry_after=retry_after,
            request_id=request_id,
            suggestion="Reduce request frequency or wait for Retry-After",
        )
    if response.status_code >= 500:
        suggestion = (
            "Check that backend services (e.g. PostgreSQL, Neo4j, Redis) are running"
            if response.status_code == 503
            else "Server error; retry later or check server logs"
        )
        raise ServerError(
            msg,
            status_code=response.status_code,
            response_body=body,
            request_id=request_id,
            suggestion=suggestion,
        )
    raise CMLError(
        msg,
        status_code=response.status_code,
        response_body=body,
        request_id=request_id,
    )


class HTTPTransport:
    """Synchronous HTTP transport for CML API."""

    def __init__(self, config: CMLConfig) -> None:
        self._config = config
        self._client: httpx.Client | None = None

    def _build_headers(self, use_admin_key: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"py-cml/{__version__}",
        }
        api_key = (
            self._config.admin_api_key
            if (use_admin_key and self._config.admin_api_key)
            else self._config.api_key
        )
        if api_key:
            headers["X-API-Key"] = api_key
        if self._config.tenant_id:
            headers["X-Tenant-ID"] = self._config.tenant_id
        return headers

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                verify=self._config.verify_ssl,
                headers=self._build_headers(),
                http2=True,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    def _do_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        use_admin_key: bool = False,
    ) -> dict[str, Any]:
        url = API_PREFIX + path
        headers = {}
        if use_admin_key and self._config.admin_api_key:
            headers["X-API-Key"] = self._config.admin_api_key
        start = time.perf_counter()
        try:
            response = self.client.request(
                method=method,
                url=url,
                json=json,
                params=params,
                headers=headers or None,
            )
            _raise_for_status(response)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "%s %s%s → %s (%.0fms)",
                method,
                API_PREFIX,
                path,
                response.status_code,
                elapsed_ms,
            )
            return cast("dict[str, Any]", response.json())
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to {self._config.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out after {self._config.timeout}s: {e}") from e

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        use_admin_key: bool = False,
    ) -> dict[str, Any]:
        """Execute an HTTP request with retry and error handling."""
        return retry_sync(
            self._config,
            self._do_request,
            method,
            path,
            json=json,
            params=params,
            use_admin_key=use_admin_key,
        )

    def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            self._client.close()


class AsyncHTTPTransport:
    """Asynchronous HTTP transport for CML API."""

    def __init__(self, config: CMLConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    def _build_headers(self, use_admin_key: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"py-cml/{__version__}",
        }
        api_key = (
            self._config.admin_api_key
            if (use_admin_key and self._config.admin_api_key)
            else self._config.api_key
        )
        if api_key:
            headers["X-API-Key"] = api_key
        if self._config.tenant_id:
            headers["X-Tenant-ID"] = self._config.tenant_id
        return headers

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                verify=self._config.verify_ssl,
                headers=self._build_headers(),
                http2=True,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    async def _do_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        use_admin_key: bool = False,
    ) -> dict[str, Any]:
        url = API_PREFIX + path
        headers = {}
        if use_admin_key and self._config.admin_api_key:
            headers["X-API-Key"] = self._config.admin_api_key
        start = time.perf_counter()
        try:
            response = await self.client.request(
                method=method,
                url=url,
                json=json,
                params=params,
                headers=headers or None,
            )
            _raise_for_status(response)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "%s %s%s → %s (%.0fms)",
                method,
                API_PREFIX,
                path,
                response.status_code,
                elapsed_ms,
            )
            return cast("dict[str, Any]", response.json())
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to {self._config.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out after {self._config.timeout}s: {e}") from e

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        use_admin_key: bool = False,
    ) -> dict[str, Any]:
        """Execute an HTTP request with retry and error handling."""
        return await retry_async(
            self._config,
            self._do_request,
            method,
            path,
            json=json,
            params=params,
            use_admin_key=use_admin_key,
        )

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
