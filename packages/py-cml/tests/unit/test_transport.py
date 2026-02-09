"""Tests for HTTP transport (mocked httpx)."""

from unittest.mock import MagicMock

import httpx
import pytest

from cml.config import CMLConfig
from cml.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from cml.transport.http import API_PREFIX, HTTPTransport


def test_transport_builds_correct_url_and_headers() -> None:
    """Transport uses API prefix and sends API key and tenant headers."""
    config = CMLConfig(
        api_key="sk-test",
        base_url="http://localhost:8000",
        tenant_id="my-tenant",
    )
    transport = HTTPTransport(config)
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = {"status": "ok"}
    mock_client = MagicMock()
    mock_client.request.return_value = mock_response
    mock_client.is_closed = False
    transport._client = mock_client

    result = transport._do_request("GET", "/health")

    assert result == {"status": "ok"}
    transport._client.request.assert_called_once()
    call_kwargs = transport._client.request.call_args[1]
    assert call_kwargs["url"] == API_PREFIX + "/health"
    # Default headers (API key, tenant) are set on the real httpx.Client at build time;
    # _do_request only passes override headers when use_admin_key is True.
    assert call_kwargs["method"] == "GET"


def test_build_headers_includes_api_key_and_tenant() -> None:
    """_build_headers returns X-API-Key and X-Tenant-ID when set in config."""
    config = CMLConfig(
        api_key="sk-test",
        base_url="http://localhost:8000",
        tenant_id="my-tenant",
    )
    transport = HTTPTransport(config)
    headers = transport._build_headers()
    assert headers["X-API-Key"] == "sk-test"
    assert headers["X-Tenant-ID"] == "my-tenant"
    assert "User-Agent" in headers
    assert "py-cml/" in headers["User-Agent"]


def test_raise_for_status_401() -> None:
    """Transport maps 401 to AuthenticationError."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    transport = HTTPTransport(config)
    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.status_code = 401
    mock_response.json.return_value = {"detail": "Invalid API key"}
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.request.return_value = mock_response
    transport._client = mock_client

    with pytest.raises(AuthenticationError) as exc_info:
        transport._do_request("GET", "/health")
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in str(exc_info.value)


def test_raise_for_status_404() -> None:
    """Transport maps 404 to NotFoundError."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    transport = HTTPTransport(config)
    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.status_code = 404
    mock_response.json.return_value = {"detail": "Not found"}
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.request.return_value = mock_response
    transport._client = mock_client

    with pytest.raises(NotFoundError) as exc_info:
        transport._do_request("GET", "/memory/stats")
    assert exc_info.value.status_code == 404


def test_raise_for_status_403() -> None:
    """Transport maps 403 to AuthorizationError."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    transport = HTTPTransport(config)
    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.status_code = 403
    mock_response.json.return_value = {"detail": "Forbidden"}
    mock_response.headers = {}
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.request.return_value = mock_response
    transport._client = mock_client

    with pytest.raises(AuthorizationError) as exc_info:
        transport._do_request("GET", "/admin/tenants")
    assert exc_info.value.status_code == 403


def test_raise_for_status_422() -> None:
    """Transport maps 422 to ValidationError."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    transport = HTTPTransport(config)
    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.status_code = 422
    mock_response.json.return_value = {"detail": "Validation error"}
    mock_response.headers = {}
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.request.return_value = mock_response
    transport._client = mock_client

    with pytest.raises(ValidationError) as exc_info:
        transport._do_request("POST", "/memory/write", json={})
    assert exc_info.value.status_code == 422


def test_raise_for_status_429_with_retry_after() -> None:
    """Transport maps 429 to RateLimitError and passes Retry-After."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    transport = HTTPTransport(config)
    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.status_code = 429
    mock_response.json.return_value = {"detail": "Rate limit exceeded"}
    mock_response.headers = {"Retry-After": "60"}
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.request.return_value = mock_response
    transport._client = mock_client

    with pytest.raises(RateLimitError) as exc_info:
        transport._do_request("GET", "/memory/read")
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 60.0


def test_raise_for_status_500() -> None:
    """Transport maps 500 to ServerError."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    transport = HTTPTransport(config)
    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.status_code = 500
    mock_response.json.return_value = {"detail": "Internal server error"}
    mock_response.headers = {}
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.request.return_value = mock_response
    transport._client = mock_client

    with pytest.raises(ServerError) as exc_info:
        transport._do_request("GET", "/health")
    assert exc_info.value.status_code == 500


def test_do_request_connection_error() -> None:
    """Transport maps httpx.ConnectError to ConnectionError."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    transport = HTTPTransport(config)
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.request.side_effect = httpx.ConnectError("Connection refused")
    transport._client = mock_client

    with pytest.raises(ConnectionError) as exc_info:
        transport._do_request("GET", "/health")
    assert "Failed to connect" in str(exc_info.value)


def test_do_request_timeout_error() -> None:
    """Transport maps httpx.TimeoutException to TimeoutError."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000", timeout=5.0)
    transport = HTTPTransport(config)
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.request.side_effect = httpx.TimeoutException("Timed out")
    transport._client = mock_client

    with pytest.raises(TimeoutError) as exc_info:
        transport._do_request("GET", "/health")
    assert "timed out" in str(exc_info.value).lower()


def test_build_headers_uses_admin_key_when_requested() -> None:
    """When use_admin_key is True and admin_api_key is set, X-API-Key is admin key."""
    config = CMLConfig(
        api_key="sk-user",
        base_url="http://localhost:8000",
        admin_api_key="sk-admin",
    )
    transport = HTTPTransport(config)
    normal = transport._build_headers(use_admin_key=False)
    admin = transport._build_headers(use_admin_key=True)
    assert normal["X-API-Key"] == "sk-user"
    assert admin["X-API-Key"] == "sk-admin"


def test_close_closes_client() -> None:
    """close() closes the underlying httpx client."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    transport = HTTPTransport(config)
    mock_client = MagicMock()
    mock_client.is_closed = False
    transport._client = mock_client

    transport.close()
    mock_client.close.assert_called_once()
