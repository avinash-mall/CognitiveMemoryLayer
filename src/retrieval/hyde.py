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


_COGNITIVE_CUE_PROMPT = """A person just made this remark in a conversation:
"{query}"

This remark may be implicitly connected to a past personal experience, habit,
hard-learned lesson, boundary, fear, allergy, goal, or value they mentioned weeks ago
using completely different words.

Write 3 distinct, concrete 1-sentence past statements the speaker might have said
earlier that explain their current remark (focus on causal events, personal rules,
past scares, or boundaries — avoid repeating the exact words of the remark).
Return exactly 3 lines, no numbering or bullets."""


_BRIDGE_QUERY_PROMPT = """Question: "{query}"

Retrieved memories so far:
{snippets}

If these memories already contain the complete, specific answer (including exact entity names, counts, or items), reply ONLY with: NONE
If they mention an unresolved reference (e.g., "my home country", "a friend", "that event", "the painting") OR only partially answer a multi-part/count question ("how many times", "what types"), write ONE short search query (4 to 12 words) to retrieve the missing fact or additional occurrences. No explanation.

Follow-up query or NONE:"""

_COGNITIVE_SELECT_PROMPT = """Current speaker remark:
"{query}"

Candidate past memories (numbered):
{candidates}

Which 1 to 3 numbered memories above have the strongest hidden causal, habit, boundary, fear, allergy, goal, or past-experience connection to the speaker's current remark (even if they use completely different words)?
Reply ONLY with comma-separated numbers in order of best match (e.g. `14, 3`), or `NONE` if none are relevant."""


class HyDEGenerator:
    """Generates hypothetical documents and query expansions for retrieval."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def select_cognitive_cues(self, query: str, candidates: list[str]) -> list[int]:
        """Select top 1-3 0-based candidate indices with hidden cognitive connection to query."""
        import re

        if not query or not candidates:
            return []
        numbered = "\n".join(f"{i + 1}. {c[:180]}" for i, c in enumerate(candidates[:22]))
        try:
            response = await self._llm.complete(
                _COGNITIVE_SELECT_PROMPT.format(query=query.strip(), candidates=numbered),
                temperature=0.0,
                max_tokens=32,
            )
            txt = response.strip()
            if not txt or txt.upper().startswith("NONE"):
                return []
            nums = [int(n) - 1 for n in re.findall(r"\b(\d{1,2})\b", txt)]
            seen: set[int] = set()
            valid: list[int] = []
            for idx in nums:
                if 0 <= idx < len(candidates) and idx not in seen:
                    seen.add(idx)
                    valid.append(idx)
            return valid[:3]
        except Exception as exc:
            logger.warning("cognitive_cue_select_failed", error=str(exc))
            return []

    async def generate_bridge_query(self, query: str, snippets: list[str]) -> str | None:
        """Generate a targeted 2nd-hop bridge search query when 1st-pass snippets leave a gap."""
        if not query or not snippets:
            return None
        joined = "\n".join(f"- {s[:160]}" for s in snippets[:12])
        try:
            response = await self._llm.complete(
                _BRIDGE_QUERY_PROMPT.format(query=query.strip(), snippets=joined),
                temperature=0.0,
                max_tokens=48,
            )
            cleaned = response.strip().splitlines()[0].strip().strip('"').strip("'")
            if not cleaned or cleaned.upper().startswith("NONE") or len(cleaned) < 6:
                return None
            return cleaned
        except Exception as exc:
            logger.warning("bridge_query_generation_failed", error=str(exc))
            return None

    async def generate_cognitive_cues(self, query: str) -> list[str]:
        """Generate 3 hypothetical past cue statements to bridge semantic disconnect."""
        if not query or not query.strip():
            return []
        try:
            response = await self._llm.complete(
                _COGNITIVE_CUE_PROMPT.format(query=query.strip()),
                temperature=0.6,
                max_tokens=240,
            )
            lines = [
                line.strip().lstrip("0123456789.-•* )")
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 12
            ]
            return lines[:3]
        except Exception as exc:
            logger.warning("cognitive_hyde_generation_failed", error=str(exc))
            return []

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
