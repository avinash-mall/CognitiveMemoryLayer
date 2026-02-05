"""Working memory manager: per-scope state and chunk processing."""
from typing import Dict, List, Optional

import asyncio

from ...utils.llm import LLMClient
from .chunker import RuleBasedChunker, SemanticChunker
from .models import SemanticChunk, WorkingMemoryState


class WorkingMemoryManager:
    """
    Manages working memory states per scope.

    Responsibilities:
    1. Process sensory buffer into chunks
    2. Maintain limited-capacity working memory
    3. Decide what needs long-term encoding
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_chunks_per_user: int = 10,
        use_fast_chunker: bool = False,
    ) -> None:
        self._states: Dict[str, WorkingMemoryState] = {}
        self._lock = asyncio.Lock()
        self.max_chunks = max_chunks_per_user

        if llm_client and not use_fast_chunker:
            self.chunker = SemanticChunker(llm_client)
            self._use_llm = True
        else:
            self.chunker = RuleBasedChunker()
            self._use_llm = False

    def _get_key(self, tenant_id: str, scope_id: str) -> str:
        return f"{tenant_id}:{scope_id}"

    async def get_state(self, tenant_id: str, scope_id: str) -> WorkingMemoryState:
        """Get or create working memory state for scope."""
        key = self._get_key(tenant_id, scope_id)
        async with self._lock:
            if key not in self._states:
                self._states[key] = WorkingMemoryState(
                    tenant_id=tenant_id,
                    user_id=scope_id,  # Internal model still uses user_id field name
                    max_chunks=self.max_chunks,
                )
            return self._states[key]

    async def process_input(
        self,
        tenant_id: str,
        scope_id: str,
        text: str,
        turn_id: Optional[str] = None,
        role: str = "user",
    ) -> List[SemanticChunk]:
        """
        Process new input into working memory.

        Returns:
            New chunks added to working memory
        """
        state = await self.get_state(tenant_id, scope_id)
        context = state.chunks[-5:] if state.chunks else None

        if self._use_llm:
            new_chunks = await self.chunker.chunk(
                text,
                context_chunks=context,
                turn_id=turn_id,
                role=role,
            )
        else:
            new_chunks = self.chunker.chunk(text, turn_id=turn_id, role=role)

        for chunk in new_chunks:
            state.add_chunk(chunk)
        state.turn_count += 1
        return new_chunks

    async def get_chunks_for_encoding(
        self,
        tenant_id: str,
        scope_id: str,
        min_salience: float = 0.4,
    ) -> List[SemanticChunk]:
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
        async with self._lock:
            if key in self._states:
                del self._states[key]

    async def get_stats(
        self, tenant_id: str, scope_id: str
    ) -> Dict[str, object]:
        """Get working memory statistics."""
        state = await self.get_state(tenant_id, scope_id)
        avg_salience = (
            sum(c.salience for c in state.chunks) / len(state.chunks)
            if state.chunks
            else 0.0
        )
        return {
            "chunk_count": len(state.chunks),
            "max_chunks": state.max_chunks,
            "turn_count": state.turn_count,
            "current_topic": state.current_topic,
            "last_updated": state.last_updated.isoformat(),
            "avg_salience": avg_salience,
        }
