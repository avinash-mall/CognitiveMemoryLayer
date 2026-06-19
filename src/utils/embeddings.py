"""Embedding service for memory content."""

import asyncio
import hashlib
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.config import EmbeddingInternalSettings, get_embedding_dimensions, get_settings

# Retryable exceptions for embedding API calls (transient network/rate-limit).
# The OpenAI SDK raises APITimeoutError/APIConnectionError/RateLimitError/
# InternalServerError — none subclass the builtins below, so they must be listed
# explicitly or the @retry decorators never actually fire on a transient error.
_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, OSError)
try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    _RETRY_EXCEPTIONS = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        *_RETRY_EXCEPTIONS,
    )
except ImportError:
    pass

# Default when EMBEDDING_INTERNAL__* not provided: Sentence Transformer
# nomic-ai/nomic-embed-text-v2-moe (768 dims, 512 max sequence length)
_DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
_DEFAULT_EMBEDDING_DIMENSIONS = 768

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

_logger = structlog.get_logger(__name__)
_EMBEDDING_CLIENT_CACHE: dict[
    tuple[str, str, int, str, str, str, str, str, float], "EmbeddingClient"
] = {}


@dataclass
class EmbeddingResult:
    embedding: list[float]
    model: str
    dimensions: int
    tokens_used: int


class EmbeddingClient(ABC):
    """Abstract embedding client."""

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]: ...


class OpenAIEmbeddings(EmbeddingClient):
    """OpenAI embedding client (supports any OpenAI-compatible embedding endpoint via base_url).

    When ``pass_dimensions`` is *False* the ``dimensions`` parameter is omitted
    from API requests.  This is needed for providers like Ollama whose
    ``/v1/embeddings`` endpoint does not accept this OpenAI-specific parameter.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
        pass_dimensions: bool = True,
    ) -> None:
        import os

        if AsyncOpenAI is None:
            raise ImportError("openai package is required for OpenAIEmbeddings")
        settings = get_settings()
        ei = getattr(settings, "embedding_internal", None) or EmbeddingInternalSettings()
        key = api_key or ei.api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or ei.model or _DEFAULT_EMBEDDING_MODEL
        self._dimensions = (
            dimensions
            if dimensions is not None
            else (ei.dimensions if ei.dimensions is not None else _DEFAULT_EMBEDDING_DIMENSIONS)
        )
        self._pass_dimensions = pass_dimensions
        url = base_url or ei.base_url
        if url:
            self.client = AsyncOpenAI(base_url=url, api_key=key)
        else:
            self.client = AsyncOpenAI(api_key=key)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=5),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    )
    async def embed(self, text: str) -> EmbeddingResult:
        kwargs: dict = dict(model=self.model, input=text)
        if self._pass_dimensions:
            kwargs["dimensions"] = self._dimensions
        response = await self.client.embeddings.create(**kwargs)
        return EmbeddingResult(
            embedding=response.data[0].embedding,
            model=self.model,
            dimensions=self._dimensions,
            tokens_used=response.usage.total_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=5),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    )
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        kwargs: dict = dict(model=self.model, input=texts)
        if self._pass_dimensions:
            kwargs["dimensions"] = self._dimensions
        response = await self.client.embeddings.create(**kwargs)
        per_token = response.usage.total_tokens // len(texts) if texts else 0
        return [
            EmbeddingResult(
                embedding=item.embedding,
                model=self.model,
                dimensions=self._dimensions,
                tokens_used=per_token,
            )
            for item in response.data
        ]


class LocalEmbeddings(EmbeddingClient):
    """Local sentence-transformers embeddings (optional dependency)."""

    def __init__(
        self,
        model_name: str | None = None,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required for LocalEmbeddings")
        torch: Any = None
        try:
            import torch as _torch

            torch = _torch
        except ImportError:
            pass
        settings = get_settings()
        ei = getattr(settings, "embedding_internal", None) or EmbeddingInternalSettings()
        name = model_name or ei.local_model or _DEFAULT_EMBEDDING_MODEL
        model_revision = revision or ei.revision
        # HuggingFace repo id cannot contain ':' (e.g. :latest); strip tag for download
        if ":" in name:
            name = name.split(":")[0]
        device_preference = device or ei.device
        resolved_device = "cpu"
        if device_preference in {"auto", "cuda"}:
            if torch is not None and getattr(torch, "cuda", None) is not None:
                try:
                    if bool(torch.cuda.is_available()):
                        resolved_device = "cuda"
                except Exception:
                    resolved_device = "cpu"
        sentence_transformer_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device": resolved_device,
            "model_kwargs": {"trust_remote_code": True},
            "tokenizer_kwargs": {"trust_remote_code": True},
            "config_kwargs": {"trust_remote_code": True},
        }
        if model_revision:
            sentence_transformer_kwargs["revision"] = model_revision
        self.model = SentenceTransformer(name, **sentence_transformer_kwargs)
        self.model_name = name
        self.revision = model_revision
        self.device = resolved_device
        self.device_preference = device_preference
        self._batch_size = (
            ei.local_batch_size
            if ei.local_batch_size > 0
            else (64 if resolved_device == "cuda" else 8)
        )
        self._encode_lock = threading.Lock()
        self._dimensions = self.model.get_sentence_embedding_dimension()
        _logger.info(
            "local_embeddings_loaded",
            model=self.model_name,
            revision=self.revision,
            device=self.device,
            device_preference=self.device_preference,
            batch_size=self._batch_size,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions or 0

    def _encode_sync(self, texts: str | list[str], *, batch_size: int | None = None) -> Any:
        kwargs: dict[str, Any] = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        with self._encode_lock:
            return self.model.encode(texts, **kwargs)

    async def _encode_async(self, texts: str | list[str], *, batch_size: int | None = None) -> Any:
        import asyncio

        loop = asyncio.get_running_loop()
        try:
            encoded = await loop.run_in_executor(
                None,
                lambda: self._encode_sync(texts, batch_size=batch_size),
            )
        except Exception:
            _logger.exception(
                "local_embeddings_encode_failed",
                model=self.model_name,
                revision=self.revision,
                device=self.device,
                device_preference=self.device_preference,
                batch_size=batch_size,
                input_count=1 if isinstance(texts, str) else len(texts),
            )
            raise
        return encoded.tolist() if hasattr(encoded, "tolist") else encoded

    async def embed(self, text: str) -> EmbeddingResult:
        embedding = await self._encode_async(text)
        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            dimensions=self._dimensions or 0,
            tokens_used=len(text.split()),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        bs = self._batch_size
        embeddings = await self._encode_async(texts, batch_size=bs)
        return [
            EmbeddingResult(
                embedding=emb,
                model=self.model_name,
                dimensions=self._dimensions or 0,
                tokens_used=len(t.split()),
            )
            for emb, t in zip(embeddings, texts, strict=False)
        ]


class MockEmbeddingClient(EmbeddingClient):
    """Deterministic mock for tests; no API calls."""

    def __init__(self, dimensions: int | None = None) -> None:
        if dimensions is not None:
            self._dimensions = dimensions
        else:
            self._dimensions = get_embedding_dimensions()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        # Deterministic hashed-token embedding so texts with lexical overlap
        # produce higher cosine similarity in tests and offline environments.
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        if not tokens:
            tokens = ["__empty__"]
        embedding = [0.0] * self._dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            for offset in (0, 8):
                idx = int.from_bytes(digest[offset : offset + 4], "little") % self._dimensions
                sign = 1.0 if digest[offset + 4] % 2 == 0 else -1.0
                weight = 1.0 + (digest[offset + 5] / 255.0) * 0.25
                embedding[idx] += sign * weight
        text_digest = hashlib.sha256(text.encode()).digest()
        embedding[int.from_bytes(text_digest[:4], "little") % self._dimensions] += 0.01
        # L2-normalize for reliable cosine similarity
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return EmbeddingResult(
            embedding=embedding,
            model="mock",
            dimensions=self._dimensions,
            tokens_used=len(tokens),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [await self.embed(t) for t in texts]


class CachedEmbeddings(EmbeddingClient):
    """Wrapper that caches embeddings in Redis."""

    def __init__(
        self,
        client: EmbeddingClient,
        redis_client: Any,
        ttl_seconds: int = 86400,
    ) -> None:
        self.client = client
        self.redis = redis_client
        self.ttl = ttl_seconds

    @property
    def dimensions(self) -> int:
        return self.client.dimensions

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:32]
        # Namespace by resolved model + dimensions so changing the embedding
        # model never returns a stale cached vector of the wrong length. The
        # client may be wrapped (e.g. BatchingEmbeddingClient) — unwrap it, and
        # fall back to "default" only when no string model id is available.
        inner = getattr(self.client, "_inner", self.client)
        model_id = getattr(inner, "model_name", None) or getattr(inner, "model", None)
        if not isinstance(model_id, str):
            model_id = "default"
        return f"emb:{model_id}:{self.dimensions}:{h}"

    async def embed(self, text: str) -> EmbeddingResult:
        import json

        cache_key = self._cache_key(text)
        cached = await self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return EmbeddingResult(**data)
        result = await self.client.embed(text)
        await self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(
                {
                    "embedding": result.embedding,
                    "model": result.model,
                    "dimensions": result.dimensions,
                    "tokens_used": result.tokens_used,
                }
            ),
        )
        return result

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Batch embed with Redis MGET/pipeline for efficient cache access."""
        import json

        cache_keys = [self._cache_key(t) for t in texts]

        # Batch cache lookup (MGET instead of NxGET)
        try:
            cached_values = await self.redis.mget(*cache_keys)
        except Exception:
            # Fallback: no cache
            cached_values = [None] * len(texts)

        results: list[tuple[int, EmbeddingResult | None]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, cached in enumerate(cached_values):
            if cached:
                try:
                    results.append((i, EmbeddingResult(**json.loads(cached))))
                    continue
                except Exception:
                    pass  # Corrupted cache entry
            results.append((i, None))
            uncached_texts.append(texts[i])
            uncached_indices.append(i)

        # Batch compute uncached
        if uncached_texts:
            computed = await self.client.embed_batch(uncached_texts)
            if len(computed) != len(uncached_texts):
                raise ValueError(
                    f"embed_batch returned {len(computed)} results for {len(uncached_texts)} inputs; "
                    "counts must match"
                )

            # Batch cache store (pipeline instead of NxSETEX)
            try:
                pipe = self.redis.pipeline()
                for idx, result in zip(uncached_indices, computed, strict=False):
                    results[idx] = (idx, result)
                    pipe.setex(
                        cache_keys[idx],
                        self.ttl,
                        json.dumps(
                            {
                                "embedding": result.embedding,
                                "model": result.model,
                                "dimensions": result.dimensions,
                                "tokens_used": result.tokens_used,
                            }
                        ),
                    )
                await pipe.execute()
            except Exception:
                # Cache write failure — results are still valid
                for idx, result in zip(uncached_indices, computed, strict=False):
                    results[idx] = (idx, result)

        results.sort(key=lambda x: x[0])
        return [r for _, r in results if r is not None]


class BatchingEmbeddingClient(EmbeddingClient):
    """Wraps any EmbeddingClient to coalesce concurrent embed calls into larger GPU batches.

    Multiple concurrent ``embed`` / ``embed_batch`` calls that arrive within
    ``max_wait_ms`` are merged into a single ``inner.embed_batch()`` call,
    amortising per-request overhead and saturating GPU throughput.  Works for
    all API paths (write, retrieval, evaluation) — not just bulk ingestion.

    Thread-safety: asyncio-only (each uvicorn worker has its own event loop and
    its own ``BatchingEmbeddingClient`` instance via the singleton cache).
    """

    def __init__(
        self,
        inner: EmbeddingClient,
        max_wait_ms: float = 10.0,
        max_batch_size: int = 512,
    ) -> None:
        self._inner = inner
        self._max_wait = max_wait_ms / 1000.0
        self._max_batch = max_batch_size
        # Per-event-loop state — created lazily so the object is safe to
        # construct outside of a running loop (e.g. at module import time).
        self._lock: asyncio.Lock | None = None
        self._pending: list[tuple[str, asyncio.Future]] = []
        self._dispatch_task: asyncio.Task | None = None

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def embed(self, text: str) -> EmbeddingResult:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future] = [loop.create_future() for _ in texts]
        lock = self._get_lock()
        async with lock:
            for text, fut in zip(texts, futures, strict=True):
                self._pending.append((text, fut))
            if self._dispatch_task is None or self._dispatch_task.done():
                self._dispatch_task = loop.create_task(self._dispatch_after_wait())
        return list(await asyncio.gather(*futures))

    async def _dispatch_after_wait(self) -> None:
        await asyncio.sleep(self._max_wait)
        await self._drain()

    async def _drain(self) -> None:
        lock = self._get_lock()
        async with lock:
            if not self._pending:
                return
            batch = self._pending[: self._max_batch]
            self._pending = self._pending[self._max_batch :]
            if self._pending:
                loop = asyncio.get_running_loop()
                self._dispatch_task = loop.create_task(self._drain())
            else:
                self._dispatch_task = None

        texts = [t for t, _ in batch]
        try:
            results = await self._inner.embed_batch(texts)
        except Exception as exc:
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return

        if len(results) != len(texts):
            exc_val = ValueError(
                f"embed_batch returned {len(results)} results for {len(texts)} texts"
            )
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc_val)
            return

        for (_, fut), result in zip(batch, results, strict=True):
            if not fut.done():
                fut.set_result(result)


def get_embedding_client() -> EmbeddingClient:
    """Factory function to get configured embedding client. Reads EMBEDDING_INTERNAL__*.
    When not provided, defaults to LocalEmbeddings with nomic-ai/nomic-embed-text-v2-moe (768d)."""
    import os

    settings = get_settings()
    ei = getattr(settings, "embedding_internal", None) or EmbeddingInternalSettings()
    provider = ei.provider if ei.provider is not None else "local"
    dims = ei.dimensions if ei.dimensions is not None else _DEFAULT_EMBEDDING_DIMENSIONS
    model = ei.model or _DEFAULT_EMBEDDING_MODEL
    local_model = ei.local_model or _DEFAULT_EMBEDDING_MODEL
    revision = ei.revision or ""
    api_key = ei.api_key or os.environ.get("OPENAI_API_KEY") or ""
    base_url = ei.base_url or ""
    device = ei.device
    batch_wait_ms = getattr(ei, "batch_wait_ms", 10.0)

    cache_key = (
        provider,
        model,
        dims,
        local_model,
        revision,
        api_key,
        base_url,
        device,
        batch_wait_ms,
    )
    cached = _EMBEDDING_CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    client: EmbeddingClient

    if provider == "openai":
        if not api_key and not base_url:
            _logger.warning(
                "embedding_openai_missing_key_fallback_to_mock",
                provider=provider,
                model=model,
            )
            client = MockEmbeddingClient(dimensions=dims)
        else:
            client = OpenAIEmbeddings(
                api_key=api_key,
                model=model,
                dimensions=dims,
                base_url=ei.base_url,
            )
    elif provider in ("openai_compatible", "vllm"):
        resolved_base_url = base_url or "http://localhost:8000/v1"
        resolved_api_key = api_key or "dummy"
        client = OpenAIEmbeddings(
            api_key=resolved_api_key,
            model=model,
            dimensions=dims,
            base_url=resolved_base_url,
        )
    elif provider == "ollama":
        resolved_base_url = base_url or "http://localhost:11434/v1"
        resolved_api_key = api_key or "ollama"
        client = OpenAIEmbeddings(
            api_key=resolved_api_key,
            model=model,
            dimensions=dims,
            base_url=resolved_base_url,
            pass_dimensions=False,
        )
    elif provider == "mock":
        client = MockEmbeddingClient(dimensions=dims)
    else:
        # default: local (nomic-embed-text-v2-moe) when provider is local or unset
        client = LocalEmbeddings(model_name=local_model, revision=ei.revision, device=device)

    # Wrap with dynamic batcher for all non-mock clients when enabled
    if batch_wait_ms > 0 and not isinstance(client, MockEmbeddingClient):
        client = BatchingEmbeddingClient(client, max_wait_ms=batch_wait_ms)

    _EMBEDDING_CLIENT_CACHE[cache_key] = client
    return client


def clear_embedding_client_cache() -> None:
    """Clear cached embedding clients. Useful in tests that mutate env config."""
    _EMBEDDING_CLIENT_CACHE.clear()
