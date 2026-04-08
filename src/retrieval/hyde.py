"""HyDE (Hypothetical Document Embedding) for cognitive memory queries.

For LoCoMo-Plus queries where the cue and trigger have low semantic overlap,
HyDE generates a hypothetical memory that *would* answer the question, then
searches with that hypothetical document.  This bridges the semantic gap
between the query and the stored memories.
"""

from __future__ import annotations

import structlog

from ..utils.llm import LLMClient

logger = structlog.get_logger(__name__)

_HYDE_PROMPT = """Given this query from a user in a conversation:
"{query}"

Imagine you have perfect memory of all past conversations with this user.
Write what a relevant past memory entry would look like that would help
answer this query. Focus on implicit constraints, goals, preferences,
emotional states, or behavioral patterns the user may have expressed earlier.

Write 1-2 sentences as if you are recalling the memory. Do NOT answer the
question — just describe what the relevant memory would contain.

Hypothetical memory:"""

_MULTI_QUERY_PROMPT = """Generate 3 different reformulations of this query
that capture different aspects of what information might be needed from
past conversations. Each reformulation should approach the question from
a different angle.

Query: "{query}"

Return exactly 3 reformulations, one per line. No numbering or bullets.
Just the reformulated queries, nothing else."""


class HyDEGenerator:
    """Generates hypothetical documents and query expansions for retrieval."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def generate_hypothetical_memory(self, query: str) -> str | None:
        """Generate a hypothetical memory that would answer the query.

        Returns None if generation fails.
        """
        if not query or not query.strip():
            return None

        try:
            response = await self._llm.complete(
                _HYDE_PROMPT.format(query=query.strip()),
                temperature=0.7,
                max_tokens=200,
            )
            result = response.strip()
            if result and len(result) > 10:
                return result
        except Exception as exc:
            logger.warning("hyde_generation_failed", error=str(exc))

        return None

    async def generate_multi_query(self, query: str) -> list[str]:
        """Generate multiple reformulations of the query.

        Returns a list of 1-3 reformulated queries, or empty list on failure.
        """
        if not query or not query.strip():
            return []

        try:
            response = await self._llm.complete(
                _MULTI_QUERY_PROMPT.format(query=query.strip()),
                temperature=0.7,
                max_tokens=300,
            )
            lines = [
                line.strip().lstrip("0123456789.-) ")
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            return lines[:3]
        except Exception as exc:
            logger.warning("multi_query_generation_failed", error=str(exc))

        return []
