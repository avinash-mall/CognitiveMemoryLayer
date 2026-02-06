"""
Seamless memory provider: automatic retrieval and storage per turn.
Makes memory recall unconscious, like human association.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..core.schemas import MemoryPacket, MemoryRecord, RetrievedMemory

try:
    from ..memory.orchestrator import MemoryOrchestrator
except ImportError:
    MemoryOrchestrator = None  # type: ignore


@dataclass
class SeamlessTurnResult:
    """Result of processing a conversation turn with seamless memory."""

    memory_context: str  # Formatted string for LLM injection
    injected_memories: List[RetrievedMemory]
    stored_count: int
    reconsolidation_applied: bool


class SeamlessMemoryProvider:
    """
    Automatically retrieves relevant memories for each interaction.
    Makes memory recall unconscious - like human association.
    """

    def __init__(
        self,
        orchestrator: "MemoryOrchestrator",
        max_context_tokens: int = 1500,
        auto_store: bool = True,
        relevance_threshold: float = 0.3,
    ):
        self.orchestrator = orchestrator
        self.max_context_tokens = max_context_tokens
        self.auto_store = auto_store
        self.relevance_threshold = relevance_threshold

    async def process_turn(
        self,
        tenant_id: str,
        user_message: str,
        assistant_response: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> SeamlessTurnResult:
        """
        Process a conversation turn:
        1. Auto-retrieve relevant memories for user message
        2. Optionally store salient information
        3. Run reconsolidation if assistant responded

        Returns context to inject into LLM prompt.
        """
        # Step 1: Retrieve relevant context BEFORE response
        memory_context, injected_memories = await self._retrieve_context(tenant_id, user_message)

        stored_count = 0
        reconsolidation_applied = False

        # Step 2: Store user message if salient (auto)
        if self.auto_store:
            write_result = await self.orchestrator.write(
                tenant_id=tenant_id,
                content=user_message,
                session_id=session_id,
                context_tags=["conversation", "user_input"],
            )
            stored_count += write_result.get("chunks_created", 0) or (
                1 if write_result.get("memory_id") else 0
            )

        # Step 3: If assistant responded, store and reconsolidate
        if assistant_response and self.auto_store:
            resp_result = await self._process_response(
                tenant_id=tenant_id,
                user_message=user_message,
                assistant_response=assistant_response,
                session_id=session_id,
                turn_id=turn_id,
                retrieved_memories=[m.record for m in injected_memories],
            )
            stored_count += resp_result.get("chunks_created", 0) or (
                1 if resp_result.get("memory_id") else 0
            )
            reconsolidation_applied = resp_result.get("reconsolidation_applied", False)

        return SeamlessTurnResult(
            memory_context=memory_context,
            injected_memories=injected_memories,
            stored_count=stored_count,
            reconsolidation_applied=reconsolidation_applied,
        )

    async def _retrieve_context(
        self,
        tenant_id: str,
        message: str,
    ) -> tuple[str, List[RetrievedMemory]]:
        """Retrieve and format memories for injection."""
        packet = await self.orchestrator.read(
            tenant_id=tenant_id,
            query=message,
            max_results=10,
        )
        # Filter by relevance and format for LLM
        filtered = [m for m in packet.all_memories if m.relevance_score >= self.relevance_threshold]
        context_str = self._format_for_injection(packet, filtered)
        return context_str, filtered

    def _format_for_injection(
        self,
        packet: MemoryPacket,
        memories: List[RetrievedMemory],
    ) -> str:
        """Build a context string from filtered memories (respects max_context_tokens)."""
        max_chars = self.max_context_tokens * 4  # rough: 4 chars per token
        if not memories:
            return ""

        # Build a minimal packet with only filtered memories for consistent formatting
        from ..core.enums import MemoryType

        facts = [m for m in memories if m.record.type == MemoryType.SEMANTIC_FACT]
        preferences = [m for m in memories if m.record.type == MemoryType.PREFERENCE]
        procedures = [m for m in memories if m.record.type == MemoryType.PROCEDURE]
        constraints = [m for m in memories if m.record.type == MemoryType.CONSTRAINT]
        recent = [
            m
            for m in memories
            if m.record.type
            not in (
                MemoryType.SEMANTIC_FACT,
                MemoryType.PREFERENCE,
                MemoryType.PROCEDURE,
                MemoryType.CONSTRAINT,
            )
        ]

        filtered_packet = MemoryPacket(
            query=packet.query,
            retrieved_at=packet.retrieved_at,
            facts=facts,
            recent_episodes=recent,
            preferences=preferences,
            procedures=procedures,
            constraints=constraints,
        )
        return filtered_packet.to_context_string(max_chars=max_chars)

    async def _process_response(
        self,
        tenant_id: str,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str],
        turn_id: Optional[str],
        retrieved_memories: list,
    ) -> dict:
        """Store assistant response and run reconsolidation."""

        # Store assistant response
        write_result = await self.orchestrator.write(
            tenant_id=tenant_id,
            content=assistant_response,
            session_id=session_id,
            context_tags=["conversation", "assistant_response"],
        )

        scope_id = session_id or tenant_id
        tid = turn_id or "turn"
        reconsolidation_applied = False

        if (
            retrieved_memories
            and hasattr(self.orchestrator, "reconsolidation")
            and self.orchestrator.reconsolidation
        ):
            records = [
                m if isinstance(m, MemoryRecord) else getattr(m, "record", m)
                for m in retrieved_memories
            ]
            rec_result = await self.orchestrator.reconsolidation.process_turn(
                tenant_id=tenant_id,
                scope_id=scope_id,
                turn_id=tid,
                user_message=user_message,
                assistant_response=assistant_response,
                retrieved_memories=records,
            )
            reconsolidation_applied = (
                rec_result.memories_processed > 0 or len(rec_result.operations_applied) > 0
            )

        return {
            **write_result,
            "reconsolidation_applied": reconsolidation_applied,
        }
