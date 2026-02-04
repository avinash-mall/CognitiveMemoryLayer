"""Short-term memory facade: sensory buffer + working memory."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .sensory.buffer import SensoryBufferConfig
from .sensory.manager import SensoryBufferManager
from .working.manager import WorkingMemoryManager
from .working.models import SemanticChunk
from ..utils.llm import LLMClient


@dataclass
class ShortTermMemoryConfig:
    """Configuration for short-term memory."""

    sensory_max_tokens: int = 500
    sensory_decay_seconds: float = 30.0
    working_max_chunks: int = 10
    use_fast_chunker: bool = False
    min_salience_for_encoding: float = 0.4


class ShortTermMemory:
    """
    Unified interface for sensory buffer + working memory.

    Entry point for all new information before long-term encoding.
    """

    def __init__(
        self,
        config: Optional[ShortTermMemoryConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.config = config or ShortTermMemoryConfig()
        sensory_config = SensoryBufferConfig(
            max_tokens=self.config.sensory_max_tokens,
            decay_seconds=self.config.sensory_decay_seconds,
        )
        self.sensory = SensoryBufferManager(sensory_config)
        self.working = WorkingMemoryManager(
            llm_client=llm_client,
            max_chunks_per_user=self.config.working_max_chunks,
            use_fast_chunker=self.config.use_fast_chunker,
        )

    async def ingest_turn(
        self,
        tenant_id: str,
        user_id: str,
        text: str,
        turn_id: Optional[str] = None,
        role: str = "user",
    ) -> Dict[str, Any]:
        """
        Ingest a new conversation turn.

        Flow:
        1. Add to sensory buffer
        2. Process into working memory chunks
        3. Return chunks ready for potential encoding
        """
        tokens_added = await self.sensory.ingest(
            tenant_id, user_id, text, turn_id, role
        )
        new_chunks = await self.working.process_input(
            tenant_id, user_id, text, turn_id, role
        )
        chunks_for_encoding = [
            c for c in new_chunks
            if c.salience >= self.config.min_salience_for_encoding
        ]
        return {
            "tokens_buffered": tokens_added,
            "chunks_created": len(new_chunks),
            "chunks_for_encoding": chunks_for_encoding,
            "all_chunks": new_chunks,
        }

    async def get_immediate_context(
        self,
        tenant_id: str,
        user_id: str,
        include_sensory: bool = True,
        max_working_chunks: int = 5,
    ) -> Dict[str, Any]:
        """Get immediate context for the current conversation."""
        result = {
            "working_memory": await self.working.get_current_context(
                tenant_id, user_id, max_working_chunks
            ),
        }
        if include_sensory:
            result["recent_text"] = await self.sensory.get_recent_text(
                tenant_id, user_id, max_tokens=200
            )
        return result

    async def get_encodable_chunks(
        self,
        tenant_id: str,
        user_id: str,
    ) -> List[SemanticChunk]:
        """Get all chunks that should be encoded into long-term memory."""
        return await self.working.get_chunks_for_encoding(
            tenant_id,
            user_id,
            min_salience=self.config.min_salience_for_encoding,
        )

    async def clear(self, tenant_id: str, user_id: str) -> None:
        """Clear all short-term memory for user."""
        await self.sensory.clear_user(tenant_id, user_id)
        await self.working.clear_user(tenant_id, user_id)
