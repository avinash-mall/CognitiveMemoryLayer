"""Main memory retriever facade."""
from typing import Optional

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
    """Main entry point for memory retrieval. Coordinates classification, planning, retrieval, rerank, and formatting."""

    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        llm_client: Optional[LLMClient] = None,
        cache: Optional[object] = None,
    ):
        self.classifier = QueryClassifier(llm_client)
        self.planner = RetrievalPlanner()
        self.retriever = HybridRetriever(hippocampal, neocortical, cache)
        self.reranker = MemoryReranker()
        self.packet_builder = MemoryPacketBuilder()

    async def retrieve(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        max_results: int = 20,
        return_packet: bool = True,
    ) -> MemoryPacket:
        """Retrieve relevant memories for a query."""
        analysis = await self.classifier.classify(query)
        plan = self.planner.plan(analysis)
        raw_results = await self.retriever.retrieve(tenant_id, user_id, plan)
        reranked = self.reranker.rerank(raw_results, query, max_results=max_results)
        if return_packet:
            return self.packet_builder.build(reranked, query)
        return MemoryPacket(query=query, recent_episodes=reranked)

    async def retrieve_for_llm(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        max_tokens: int = 2000,
        format: str = "markdown",
    ) -> str:
        """Retrieve and format memories for LLM context."""
        packet = await self.retrieve(tenant_id, user_id, query)
        return self.packet_builder.to_llm_context(packet, max_tokens, format)
