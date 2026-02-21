"""Working memory manager: per-scope state and chunk processing."""

import asyncio
from datetime import datetime

from ...utils.bounded_state import BoundedStateMap
from .chunker import SemchunkChunker
from .models import SemanticChunk, WorkingMemoryState


class WorkingMemoryManager:
    """
    Manages working memory states per scope.

    Responsibilities:
    1. Process sensory buffer into chunks
    2. Maintain limited-capacity working memory
    3. Decide what needs long-term encoding

    Uses :class:`BoundedStateMap` with LRU eviction and TTL to prevent
    unbounded memory growth in long-running servers.
    """

    def __init__(
        self,
        max_chunks_per_user: int = 10,
        max_scopes: int = 1000,
        scope_ttl_seconds: float = 1800.0,
    ) -> None:
        self._states: BoundedStateMap[WorkingMemoryState] = BoundedStateMap(
            max_size=max_scopes,
            ttl_seconds=scope_ttl_seconds,
        )
        self._lock = asyncio.Lock()
        self.max_chunks = max_chunks_per_user

        from ...core.config import get_settings

        cfg = get_settings().chunker
        self.chunker = SemchunkChunker(
            tokenizer_id=cfg.tokenizer,
            chunk_size=cfg.chunk_size,
            overlap_percent=cfg.overlap_percent,
        )

    def _get_key(self, tenant_id: str, scope_id: str) -> str:
        return f"{tenant_id}:{scope_id}"

    async def get_state(self, tenant_id: str, scope_id: str) -> WorkingMemoryState:
        """Get or create working memory state for scope."""
        key = self._get_key(tenant_id, scope_id)
        return await self._states.get_or_create(
            key,
            factory=lambda: WorkingMemoryState(
                tenant_id=tenant_id,
                user_id=scope_id,
                max_chunks=self.max_chunks,
            ),
        )

    async def process_input(
        self,
        tenant_id: str,
        scope_id: str,
        text: str,
        turn_id: str | None = None,
        role: str = "user",
        timestamp: datetime | None = None,
    ) -> list[SemanticChunk]:
        """
        Process new input into working memory.

        Returns:
            New chunks added to working memory
        """
        state = await self.get_state(tenant_id, scope_id)

        new_chunks = self.chunker.chunk(text, turn_id=turn_id, role=role, timestamp=timestamp)
        for chunk in new_chunks:
            state.add_chunk(chunk)
        state.turn_count += 1
        return new_chunks

    async def get_chunks_for_encoding(
        self,
        tenant_id: str,
        scope_id: str,
        min_salience: float = 0.4,
    ) -> list[SemanticChunk]:
        """Get chunks that should be encoded into long-term memory."""
        state = await self.get_state(tenant_id, scope_id)
        return [c for c in state.chunks if c.salience >= min_salience]

    async def get_current_context(
        self,
        tenant_id: str,
        scope_id: str,
        max_chunks: int = 5,
    ) -> str:
        """Get formatted current context for LLM prompts."""
        state = await self.get_state(tenant_id, scope_id)
        recent = sorted(
            state.chunks,
            key=lambda c: c.timestamp,
            reverse=True,
        )[:max_chunks]
        lines = [f"- [{c.chunk_type.value}] {c.text}" for c in reversed(recent)]
        return "\n".join(lines)

    async def clear_user(self, tenant_id: str, scope_id: str) -> None:
        """Clear working memory for scope."""
        key = self._get_key(tenant_id, scope_id)
        await self._states.delete(key)

    async def get_stats(self, tenant_id: str, scope_id: str) -> dict[str, object]:
        """Get working memory statistics."""
        state = await self.get_state(tenant_id, scope_id)
        avg_salience = (
            sum(c.salience for c in state.chunks) / len(state.chunks) if state.chunks else 0.0
        )
        return {
            "chunk_count": len(state.chunks),
            "max_chunks": state.max_chunks,
            "turn_count": state.turn_count,
            "current_topic": state.current_topic,
            "last_updated": state.last_updated.isoformat(),
            "avg_salience": avg_salience,
        }
