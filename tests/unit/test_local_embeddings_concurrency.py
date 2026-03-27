"""Regression tests for LocalEmbeddings concurrency safety."""

import asyncio
import sys
import threading
import time
from types import ModuleType

import pytest

from src.core.config import get_settings
from src.utils.embeddings import LocalEmbeddings, clear_embedding_client_cache


class _FakeEncoded:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _RaceySentenceTransformer:
    """Fake encoder that fails when concurrent calls overlap with different shapes."""

    _state_lock = threading.Lock()
    _in_flight = 0
    _active_signature = None

    def __init__(self, *_args, **_kwargs):
        self._dimensions = 6

    def get_sentence_embedding_dimension(self):
        return self._dimensions

    def encode(self, texts, batch_size=None):
        items = [texts] if isinstance(texts, str) else list(texts)
        signature = tuple(len(item.split()) for item in items)

        with self._state_lock:
            if self._in_flight and self._active_signature != signature:
                raise RuntimeError(
                    "The size of tensor a (33) must match the size of tensor b (16) "
                    "at non-singleton dimension 1"
                )
            self._in_flight += 1
            self._active_signature = signature

        try:
            time.sleep(0.05)
            embeddings = []
            for index, item in enumerate(items):
                base = float(len(item.split()) + index)
                embeddings.append([base + offset for offset in range(self._dimensions)])
            return _FakeEncoded(embeddings[0] if isinstance(texts, str) else embeddings)
        finally:
            with self._state_lock:
                self._in_flight -= 1
                if self._in_flight == 0:
                    self._active_signature = None


class _RecordingSentenceTransformer:
    init_calls: list[dict[str, object]] = []
    encode_calls: list[dict[str, object]] = []

    def __init__(self, *_args, **kwargs):
        self._dimensions = 4
        self.init_calls.append(dict(kwargs))

    def get_sentence_embedding_dimension(self):
        return self._dimensions

    def encode(self, texts, **kwargs):
        self.encode_calls.append(dict(kwargs))
        if "batch_size" in kwargs and kwargs["batch_size"] is None:
            raise TypeError("batch_size=None should not be forwarded")
        if isinstance(texts, str):
            return _FakeEncoded([1.0] * self._dimensions)
        return _FakeEncoded([[1.0] * self._dimensions for _ in texts])


@pytest.fixture(autouse=True)
def _clear_settings_and_embedding_cache():
    get_settings.cache_clear()
    clear_embedding_client_cache()
    yield
    get_settings.cache_clear()
    clear_embedding_client_cache()


@pytest.fixture
def _fake_sentence_transformers(monkeypatch):
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _RaceySentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


@pytest.fixture
def _recording_sentence_transformers(monkeypatch):
    _RecordingSentenceTransformer.init_calls.clear()
    _RecordingSentenceTransformer.encode_calls.clear()
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _RecordingSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


@pytest.fixture
def _fake_cuda_torch(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.cuda = type("CudaNamespace", (), {"is_available": staticmethod(lambda: True)})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


@pytest.mark.asyncio
async def test_local_embeddings_serializes_concurrent_batch_calls(
    monkeypatch, _fake_sentence_transformers
):
    monkeypatch.setenv("EMBEDDING_INTERNAL__LOCAL_BATCH_SIZE", "4")
    client = LocalEmbeddings(model_name="fake/nomic")

    batch_a = [
        "alpha",
        "one two three four five six seven eight",
        "another sentence with several words for padding",
    ]
    batch_b = [
        "beta gamma",
        "tiny batch",
        "shape differs from batch a",
        "final item",
    ]

    result_a, result_b = await asyncio.gather(
        client.embed_batch(batch_a),
        client.embed_batch(batch_b),
    )

    assert len(result_a) == len(batch_a)
    assert len(result_b) == len(batch_b)
    assert all(item.model == "fake/nomic" for item in result_a + result_b)


@pytest.mark.asyncio
async def test_local_embeddings_serializes_mixed_single_and_batch_calls(
    monkeypatch, _fake_sentence_transformers
):
    monkeypatch.setenv("EMBEDDING_INTERNAL__LOCAL_BATCH_SIZE", "2")
    client = LocalEmbeddings(model_name="fake/nomic")

    single_result, batch_result = await asyncio.gather(
        client.embed("short text"),
        client.embed_batch(
            [
                "this batch has several tokens",
                "and a second sentence with different length",
            ]
        ),
    )

    assert single_result.model == "fake/nomic"
    assert len(single_result.embedding) == client.dimensions
    assert len(batch_result) == 2
    assert all(len(item.embedding) == client.dimensions for item in batch_result)


@pytest.mark.asyncio
async def test_local_embeddings_embed_omits_none_batch_size_and_respects_cpu_override(
    monkeypatch, _recording_sentence_transformers, _fake_cuda_torch
):
    monkeypatch.setenv("EMBEDDING_INTERNAL__DEVICE", "cpu")
    client = LocalEmbeddings(model_name="fake/nomic")

    result = await client.embed("short text")

    assert result.model == "fake/nomic"
    assert client.device == "cpu"
    assert _RecordingSentenceTransformer.init_calls[-1]["device"] == "cpu"
    assert _RecordingSentenceTransformer.encode_calls[-1] == {}
