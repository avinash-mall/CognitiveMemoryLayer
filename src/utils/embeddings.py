"""Embedding service for memory content."""

import functools
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.config import EmbeddingInternalSettings, get_embedding_dimensions, get_settings

# Retryable exceptions for embedding API calls (transient network/rate-limit)
_RETRY_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)

# Default when EMBEDDING_INTERNAL__* not provided: Sentence Transformer
# nomic-ai/nomic-embed-text-v2-moe (768 dims, 512 max sequence length)
_DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
_DEFAULT_EMBEDDING_DIMENSIONS = 768

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]


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

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required for LocalEmbeddings")
        import threading

        import torch

        self._lock = threading.Lock()
        settings = get_settings()
        ei = getattr(settings, "embedding_internal", None) or EmbeddingInternalSettings()
        name = model_name or ei.local_model or _DEFAULT_EMBEDDING_MODEL
        # HuggingFace repo id cannot contain ':' (e.g. :latest); strip tag for download
        if ":" in name:
            name = name.split(":")[0]
        # Auto-select device: CUDA > MPS (Apple) > CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.model = SentenceTransformer(name, device=device, trust_remote_code=True)
        self.model_name = name
        self._dimensions = self.model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dimensions or 0

    async def embed(self, text: str) -> EmbeddingResult:
        import asyncio

        loop = asyncio.get_running_loop()
        
        def _encode() -> list[float]:
            with self._lock:
                return self.model.encode(text).tolist()
                
        embedding = await loop.run_in_executor(None, _encode)
        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            dimensions=self._dimensions or 0,
            tokens_used=len(text.split()),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        import asyncio

        loop = asyncio.get_running_loop()
        
        def _encode() -> list[list[float]]:
            with self._lock:
                return self.model.encode(texts).tolist()
                
        embeddings = await loop.run_in_executor(None, _encode)
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
        import random

        # Use the hash to seed a deterministic PRNG, generating all dimensions
        # instead of padding most with 0.0 (LOW-15)
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = random.Random(seed)
        embedding = [rng.gauss(0.0, 0.3) for _ in range(self._dimensions)]
        # L2-normalize for reliable cosine similarity
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return EmbeddingResult(
            embedding=embedding,
            model="mock",
            dimensions=self._dimensions,
            tokens_used=len(text.split()),
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
        return f"emb:{getattr(self.client, 'model', 'default')}:{h}"

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


def _embedding_client_cache_key(
    provider: str,
    model: str,
    local_model: str,
    dimensions: int,
    base_url: str,
    api_key_present: str,
    pass_dimensions: bool,
) -> tuple[str, str, str, int, str, str, bool]:
    """Build a hashable cache key from embedding config. Used by _get_embedding_client_cached."""
    return (provider, model, local_model, dimensions, base_url or "", api_key_present, pass_dimensions)


@functools.lru_cache(maxsize=8)
def _get_embedding_client_cached(
    provider: str,
    model: str,
    local_model: str,
    dimensions: int,
    base_url: str,
    api_key_present: str,
    pass_dimensions: bool,
) -> EmbeddingClient:
    """Create an embedding client from a config key. Cached per key to avoid repeated model loads."""
    import os

    if provider == "openai":
        ei = getattr(get_settings(), "embedding_internal", None) or EmbeddingInternalSettings()
        return OpenAIEmbeddings(
            api_key=ei.api_key,
            model=model,
            dimensions=dimensions,
            base_url=base_url or None,
        )
    if provider in ("openai_compatible", "vllm"):
        ei = getattr(get_settings(), "embedding_internal", None) or EmbeddingInternalSettings()
        base = base_url or "http://localhost:8000/v1"
        api_key = ei.api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
        return OpenAIEmbeddings(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            base_url=base,
        )
    if provider == "ollama":
        ei = getattr(get_settings(), "embedding_internal", None) or EmbeddingInternalSettings()
        base = base_url or "http://localhost:11434/v1"
        api_key = ei.api_key or os.environ.get("OPENAI_API_KEY") or "ollama"
        return OpenAIEmbeddings(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            base_url=base,
            pass_dimensions=pass_dimensions,
        )
    # default: local
    return LocalEmbeddings(model_name=local_model)


def clear_embedding_client_cache() -> None:
    """Clear the embedding client cache. Call after changing embedding config (e.g. env) to get a new client."""
    _get_embedding_client_cached.cache_clear()


def get_embedding_client() -> EmbeddingClient:
    """Factory function to get configured embedding client. Reads EMBEDDING_INTERNAL__*.
    When not provided, defaults to LocalEmbeddings with nomic-ai/nomic-embed-text-v2-moe (768d).
    The returned client is cached per config; call clear_embedding_client_cache() after changing
    embedding settings if you need a new client (and typically get_settings.cache_clear() first)."""
    settings = get_settings()
    ei = getattr(settings, "embedding_internal", None) or EmbeddingInternalSettings()
    provider = ei.provider if ei.provider is not None else "local"
    dims = ei.dimensions if ei.dimensions is not None else _DEFAULT_EMBEDDING_DIMENSIONS
    model = ei.model or _DEFAULT_EMBEDDING_MODEL
    local_model = ei.local_model or _DEFAULT_EMBEDDING_MODEL
    base_url = ei.base_url or ""
    api_key_present = "ak" if ei.api_key else "no_ak"
    pass_dimensions = provider != "ollama"
    key = _embedding_client_cache_key(
        provider, model, local_model, dims, base_url, api_key_present, pass_dimensions
    )
    return _get_embedding_client_cached(*key)
