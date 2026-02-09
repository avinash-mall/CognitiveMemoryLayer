"""Tests for CMLConfig."""

import pytest

from cml.config import CMLConfig


def test_config_direct_params() -> None:
    """CMLConfig accepts direct parameters."""
    config = CMLConfig(
        api_key="sk-test",
        base_url="https://api.example.com",
        tenant_id="tenant-1",
        timeout=60.0,
        max_retries=5,
        retry_delay=2.0,
    )
    assert config.api_key == "sk-test"
    assert config.base_url == "https://api.example.com"
    assert config.tenant_id == "tenant-1"
    assert config.timeout == 60.0
    assert config.max_retries == 5
    assert config.retry_delay == 2.0


def test_config_defaults() -> None:
    """CMLConfig has correct defaults."""
    config = CMLConfig()
    assert config.base_url == "http://localhost:8000"
    assert config.tenant_id == "default"
    assert config.timeout == 30.0
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert config.verify_ssl is True


def test_config_strips_trailing_slash() -> None:
    """base_url is normalized to strip trailing slash."""
    config = CMLConfig(base_url="http://localhost:8000/")
    assert config.base_url == "http://localhost:8000"


def test_config_invalid_base_url() -> None:
    """Invalid base_url raises ValueError."""
    with pytest.raises(ValueError, match="http:// or https://"):
        CMLConfig(base_url="ftp://invalid.com")


def test_config_invalid_timeout() -> None:
    """Non-positive timeout raises ValueError."""
    with pytest.raises(ValueError, match="timeout"):
        CMLConfig(timeout=0)
    with pytest.raises(ValueError, match="timeout"):
        CMLConfig(timeout=-1)


def test_config_invalid_max_retries() -> None:
    """Negative max_retries raises ValueError."""
    with pytest.raises(ValueError, match="max_retries"):
        CMLConfig(max_retries=-1)


def test_config_invalid_retry_delay() -> None:
    """Negative retry_delay raises ValueError."""
    with pytest.raises(ValueError, match="retry_delay"):
        CMLConfig(retry_delay=-0.1)


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset fields are loaded from environment."""
    monkeypatch.setenv("CML_API_KEY", "sk-env")
    monkeypatch.setenv("CML_BASE_URL", "http://env:9000")
    monkeypatch.setenv("CML_TENANT_ID", "env-tenant")
    config = CMLConfig()
    assert config.api_key == "sk-env"
    assert config.base_url == "http://env:9000"
    assert config.tenant_id == "env-tenant"


def test_config_direct_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct parameters take precedence over env."""
    monkeypatch.setenv("CML_API_KEY", "sk-env")
    config = CMLConfig(api_key="sk-direct")
    assert config.api_key == "sk-direct"
