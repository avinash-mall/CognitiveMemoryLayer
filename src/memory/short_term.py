"""Short-term memory facade: sensory buffer + working memory."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .sensory.buffer import SensoryBufferConfig
from .sensory.manager import SensoryBufferManager
from .working.manager import WorkingMemoryManager


@dataclass
class ShortTermMemoryConfig:
    """Configuration for short-term memory."""

    sensory_max_tokens: int = 500
    sensory_decay_seconds: float = 30.0
    working_max_chunks: int = 10
    min_salience_for_encoding: float = 0.4


class ShortTermMemory:
    """
    Unified interface for sensory buffer + working memory.

    Entry point for all new information before long-term encoding.
    """

    def __init__(
        self,
        config: ShortTermMemoryConfig | None = None,
        llm_client: Any = None,
    ) -> None:
        self.config = config or ShortTermMemoryConfig()
        sensory_config = SensoryBufferConfig(
            max_tokens=self.config.sensory_max_tokens,
            decay_seconds=self.config.sensory_decay_seconds,
        )
        self.sensory = SensoryBufferManager(sensory_config)
        self.working = WorkingMemoryManager(
            max_chunks_per_user=self.config.working_max_chunks,
        )

    async def ingest_turn(
        self,
        tenant_id: str,
        scope_id: str,
        text: str,
        turn_id: str | None = None,
        role: str = "user",
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a new conversation turn.

        Flow:
        1. Add to sensory buffer
        2. Process into working memory chunks
        3. Return chunks ready for potential encoding
        """
        tokens_added = await self.sensory.ingest(tenant_id, scope_id, text, turn_id, role)
        new_chunks = await self.working.process_input(
            tenant_id, scope_id, text, turn_id, role, timestamp=timestamp
        )
        chunks_for_encoding = [
            c for c in new_chunks if c.salience >= self.config.min_salience_for_encoding
        ]
        return {
            "tokens_buffered": tokens_added,
            "chunks_created": len(new_chunks),
            "chunks_for_encoding": chunks_for_encoding,
            "all_chunks": new_chunks,
        }

    async def clear(self, tenant_id: str, scope_id: str) -> None:
        """Clear short-term memory for scope."""
        await self.sensory.clear_user(tenant_id, scope_id)
        await self.working.clear_user(tenant_id, scope_id)
