"""Unit tests for embedding client factory caching and clear_embedding_client_cache."""

import sys
from types import ModuleType

import pytest

try:
    from src.utils.embeddings import (
        LocalEmbeddings,
        clear_embedding_client_cache,
        get_embedding_client,
    )
except ImportError as e:
    if "clear_embedding_client_cache" in str(e):
        pytest.skip(
            "clear_embedding_client_cache not in embeddings (reverted to a7a54e5)",
            allow_module_level=True,
        )
    raise

from src.core.config import get_settings


class _FakeEncoded:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeSentenceTransformer:
    init_calls: list[dict[str, object]] = []

    def __init__(self, *_args, **kwargs):
        self._dimensions = 5
        self.init_calls.append(dict(kwargs))

    def get_sentence_embedding_dimension(self):
        return self._dimensions

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            return _FakeEncoded([0.1] * self._dimensions)
        return _FakeEncoded([[0.1] * self._dimensions for _ in texts])


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear settings and embedding caches before/after so tests don't leak state."""
    get_settings.cache_clear()
    clear_embedding_client_cache()
    yield
    get_settings.cache_clear()
    clear_embedding_client_cache()


@pytest.fixture
def _fake_local_runtime(monkeypatch):
    _FakeSentenceTransformer.init_calls.clear()

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    torch = ModuleType("torch")
    torch.cuda = type("CudaNamespace", (), {"is_available": staticmethod(lambda: True)})()
    monkeypatch.setitem(sys.modules, "torch", torch)


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


def test_get_embedding_client_cached_same_local_config(monkeypatch, _fake_local_runtime):
    """Local embedding clients are still cached after adding the concurrency guard."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_INTERNAL__LOCAL_MODEL", "nomic-ai/nomic-embed-text-v2-moe")
    get_settings.cache_clear()

    c1 = get_embedding_client()
    c2 = get_embedding_client()

    assert isinstance(c1, LocalEmbeddings)
    assert c1 is c2


def test_local_embedding_device_changes_cache_key(monkeypatch, _fake_local_runtime):
    """Changing EMBEDDING_INTERNAL__DEVICE should produce a distinct cached local client."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_INTERNAL__LOCAL_MODEL", "fake/nomic")
    monkeypatch.setenv("EMBEDDING_INTERNAL__DEVICE", "cpu")
    get_settings.cache_clear()
    c1 = get_embedding_client()

    monkeypatch.setenv("EMBEDDING_INTERNAL__DEVICE", "cuda")
    get_settings.cache_clear()
    c2 = get_embedding_client()

    assert isinstance(c1, LocalEmbeddings)
    assert isinstance(c2, LocalEmbeddings)
    assert c1 is not c2
    assert c1.device == "cpu"
    assert c2.device == "cuda"
