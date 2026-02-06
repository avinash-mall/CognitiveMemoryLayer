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
    """OpenAI embedding client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: int = 1536,
    ) -> None:
        import os

        if AsyncOpenAI is None:
            raise ImportError("openai package is required for OpenAIEmbeddings")
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=api_key or settings.embedding.api_key or os.environ.get("OPENAI_API_KEY", "")
        )
        self.model = model or settings.embedding.model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self._dimensions,
        )
        return EmbeddingResult(
            embedding=response.data[0].embedding,
            model=self.model,
            dimensions=self._dimensions,
            tokens_used=response.usage.total_tokens,
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self._dimensions,
        )
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

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, lambda: self.model.encode(text).tolist())
        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            dimensions=self._dimensions,
            tokens_used=len(text.split()),
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        import asyncio

        loop = asyncio.get_event_loop()
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

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        h = hashlib.sha256(text.encode()).digest()
        embedding = [(b / 255.0) - 0.5 for b in h[: self._dimensions]]
        if len(embedding) < self._dimensions:
            embedding.extend([0.0] * (self._dimensions - len(embedding)))
        return EmbeddingResult(
            embedding=embedding[: self._dimensions],
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
