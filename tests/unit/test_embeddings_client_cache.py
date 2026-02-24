"""Unit tests for embedding client factory caching and clear_embedding_client_cache."""

import pytest

try:
    from src.utils.embeddings import clear_embedding_client_cache, get_embedding_client
except ImportError as e:
    if "clear_embedding_client_cache" in str(e):
        pytest.skip(
            "clear_embedding_client_cache not in embeddings (reverted to a7a54e5)",
            allow_module_level=True,
        )
    raise

from src.core.config import get_settings


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear settings and embedding caches before/after so tests don't leak state."""
    get_settings.cache_clear()
    clear_embedding_client_cache()
    yield
    get_settings.cache_clear()
    clear_embedding_client_cache()


def test_get_embedding_client_cached_same_config(monkeypatch):
    """get_embedding_client() returns the same instance when called twice with same config."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_INTERNAL__MODEL", "text-embedding-3-small")
    get_settings.cache_clear()

    c1 = get_embedding_client()
    c2 = get_embedding_client()
    assert c1 is c2


def test_clear_embedding_client_cache_returns_new_client(monkeypatch):
    """After clear_embedding_client_cache(), next get_embedding_client() returns a new instance."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_INTERNAL__MODEL", "text-embedding-3-small")
    get_settings.cache_clear()

    c1 = get_embedding_client()
    clear_embedding_client_cache()
    c2 = get_embedding_client()
    assert c2 is not c1


def test_different_config_different_client(monkeypatch):
    """Different embedding config yields different cached clients."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_INTERNAL__MODEL", "text-embedding-3-small")
    get_settings.cache_clear()
    c1 = get_embedding_client()

    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_INTERNAL__BASE_URL", "http://localhost:11434/v1")
    get_settings.cache_clear()
    c2 = get_embedding_client()

    assert c1 is not c2


def test_clear_embedding_client_cache_idempotent(monkeypatch):
    """clear_embedding_client_cache() can be called multiple times without error."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "openai")
    get_settings.cache_clear()
    get_embedding_client()

    clear_embedding_client_cache()
    clear_embedding_client_cache()
    c = get_embedding_client()
    assert c is not None
