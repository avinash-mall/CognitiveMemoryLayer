"""Main memory retriever facade."""

from datetime import datetime

from ..core.schemas import MemoryPacket
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..utils.llm import LLMClient
from .classifier import QueryClassifier
from .packet_builder import MemoryPacketBuilder
from .planner import RetrievalPlanner
from .reranker import MemoryReranker
from .retriever import HybridRetriever


class MemoryRetriever:
    """Main entry point for memory retrieval. Coordinates classification, planning, retrieval, rerank, and formatting. Holistic: tenant-only."""

    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        llm_client: LLMClient | None = None,
        cache: object | None = None,
    ):
        self.classifier = QueryClassifier(llm_client)
        self.planner = RetrievalPlanner()
        self.retriever = HybridRetriever(hippocampal, neocortical, cache)
        self.reranker = MemoryReranker()
        self.packet_builder = MemoryPacketBuilder()

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        max_results: int = 20,
        context_filter: list[str] | None = None,
        recent_context: str | None = None,
        return_packet: bool = True,
        memory_types: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        user_timezone: str | None = None,
    ) -> MemoryPacket:
        """Retrieve relevant memories for a query. Holistic: tenant-only."""
        analysis = await self.classifier.classify(query, recent_context=recent_context)
        if user_timezone is not None:
            analysis.user_timezone = user_timezone
        plan = self.planner.plan(analysis)

        # Inject API-level filters into every retrieval step
        if memory_types or since or until:
            for step in plan.steps:
                if memory_types and not step.memory_types:
                    step.memory_types = memory_types
                if since or until:
                    time_filter = step.time_filter or {}
                    if since:
                        time_filter["since"] = since
                    if until:
                        time_filter["until"] = until
                    step.time_filter = time_filter

        raw_results = await self.retriever.retrieve(tenant_id, plan, context_filter=context_filter)
        reranked = self.reranker.rerank(raw_results, query, max_results=max_results)
        if return_packet:
            return self.packet_builder.build(reranked, query)
        return MemoryPacket(query=query, recent_episodes=reranked)

    async def retrieve_for_llm(
        self,
        tenant_id: str,
        query: str,
        max_tokens: int = 2000,
        format: str = "markdown",
        context_filter: list[str] | None = None,
        recent_context: str | None = None,
    ) -> str:
        """Retrieve and format memories for LLM context. Holistic: tenant-only."""
        packet = await self.retrieve(
            tenant_id,
            query,
            context_filter=context_filter,
            recent_context=recent_context,
        )
        return self.packet_builder.to_llm_context(packet, max_tokens, format)
