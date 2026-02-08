"""Embedding service for memory content."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import hashlib

from ..core.config import get_settings

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


@dataclass
class EmbeddingResult:
    embedding: List[float]
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
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]: ...


class OpenAIEmbeddings(EmbeddingClient):
    """OpenAI embedding client (supports any OpenAI-compatible embedding endpoint via base_url).

    When ``pass_dimensions`` is *False* the ``dimensions`` parameter is omitted
    from API requests.  This is needed for providers like Ollama whose
    ``/v1/embeddings`` endpoint does not accept this OpenAI-specific parameter.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        pass_dimensions: bool = True,
    ) -> None:
        import os

        if AsyncOpenAI is None:
            raise ImportError("openai package is required for OpenAIEmbeddings")
        settings = get_settings()
        key = api_key or settings.embedding.api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or settings.embedding.model
        self._dimensions = dimensions if dimensions is not None else settings.embedding.dimensions
        self._pass_dimensions = pass_dimensions
        url = base_url or settings.embedding.base_url
        if url:
            self.client = AsyncOpenAI(base_url=url, api_key=key)
        else:
            self.client = AsyncOpenAI(api_key=key)

    @property
    def dimensions(self) -> int:
        return self._dimensions

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

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
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

    def __init__(self, model_name: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required for LocalEmbeddings")
        settings = get_settings()
        name = model_name or settings.embedding.local_model
        self.model = SentenceTransformer(name)
        self.model_name = name
        self._dimensions = self.model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        import asyncio

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, lambda: self.model.encode(text).tolist())
        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            dimensions=self._dimensions,
            tokens_used=len(text.split()),
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        import asyncio

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, lambda: self.model.encode(texts).tolist())
        return [
            EmbeddingResult(
                embedding=emb,
                model=self.model_name,
                dimensions=self._dimensions,
                tokens_used=len(t.split()),
            )
            for emb, t in zip(embeddings, texts)
        ]


class MockEmbeddingClient(EmbeddingClient):
    """Deterministic mock for tests; no API calls."""

    def __init__(self, dimensions: Optional[int] = None) -> None:
        self._dimensions = dimensions if dimensions is not None else get_settings().embedding.dimensions

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

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
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

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        import json

        results: List[tuple] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            cached = await self.redis.get(cache_key)
            if cached:
                results.append((i, EmbeddingResult(**json.loads(cached))))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        if uncached_texts:
            computed = await self.client.embed_batch(uncached_texts)
            for idx, result in zip(uncached_indices, computed):
                results.append((idx, result))
                cache_key = self._cache_key(texts[idx])
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
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]


def get_embedding_client() -> EmbeddingClient:
    """Factory function to get configured embedding client."""
    import os

    settings = get_settings()
    provider = settings.embedding.provider
    if provider == "openai":
        return OpenAIEmbeddings(
            api_key=settings.embedding.api_key,
            model=settings.embedding.model,
            dimensions=settings.embedding.dimensions,
            base_url=settings.embedding.base_url,
        )
    if provider == "vllm":
        # OpenAI-compatible embedding endpoint (e.g. local vLLM with embedding model)
        base_url = settings.embedding.base_url or "http://localhost:8000/v1"
        api_key = settings.embedding.api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
        return OpenAIEmbeddings(
            api_key=api_key,
            model=settings.embedding.model,
            dimensions=settings.embedding.dimensions,
            base_url=base_url,
        )
    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible /v1/embeddings endpoint but does
        # NOT accept the ``dimensions`` parameter.
        base_url = settings.embedding.base_url or "http://localhost:11434/v1"
        api_key = settings.embedding.api_key or os.environ.get("OPENAI_API_KEY") or "ollama"
        return OpenAIEmbeddings(
            api_key=api_key,
            model=settings.embedding.model,
            dimensions=settings.embedding.dimensions,
            base_url=base_url,
            pass_dimensions=False,
        )
    if provider == "local":
        return LocalEmbeddings(model_name=settings.embedding.local_model)
    raise ValueError(f"Unknown embedding provider: {provider}")
