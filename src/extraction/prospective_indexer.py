"""Prospective indexing: generate forward-looking implications at write time.

This is the single highest-impact technique from the Kumiho system (93.3% on
LoCoMo-Plus vs 26.1% for Gemini-2.5-Pro).  At write time, for each extracted
memory, we generate future-facing implications that describe scenarios where
this memory would be relevant.  These implications are embedded and stored
alongside the original memory so that retrieval can match against them even
when the original cue has low semantic overlap with the future query.

Example:
  Memory: "User cut sugary drinks after cousin's diabetes diagnosis"
  Implications:
    - "User avoids sugary beverages for health reasons"
    - "Dietary recommendations should exclude high-sugar options"
    - "Family health history influences user's dietary choices"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from ..utils.llm import LLMClient

logger = structlog.get_logger(__name__)

_PROSPECTIVE_PROMPT = """Given this memory extracted from a conversation:
"{memory_content}"

Generate {count} future-facing implications. Each should describe a scenario
where this memory would be relevant to a future query or decision.
Focus on behavioral constraints, actionable preferences, and implicit needs —
not just restating the fact.

Rules:
1. Each implication must be a self-contained statement (understandable without the original memory).
2. Cover different angles: behavioral constraints, recommendations, related topics, decision contexts.
3. Use concise, natural language (1-2 sentences each).
4. Do NOT repeat the original memory verbatim.

Return a JSON array of strings, each being one implication.
Return ONLY the JSON array, no other text."""

_BATCH_PROSPECTIVE_PROMPT = """For each numbered memory below, generate {count} future-facing implications.
Each implication should describe a scenario where this memory would be relevant
to a future query or decision. Focus on behavioral constraints, actionable
preferences, and implicit needs.

Memories:
{memories_block}

Return a JSON object mapping each index (as string key "0", "1", ...) to an
array of implication strings.
Return ONLY valid JSON, no other text."""


@dataclass
class ProspectiveIndex:
    """A single prospective implication linked to its source memory."""

    implication: str
    source_memory_text: str
    source_memory_id: str | None = None


@dataclass
class ProspectiveIndexResult:
    """Result of prospective indexing for one or more memories."""

    indexes: list[ProspectiveIndex] = field(default_factory=list)


class ProspectiveIndexer:
    """Generates forward-looking implications for memories at write time."""

    def __init__(self, llm_client: LLMClient, max_implications: int = 4) -> None:
        self._llm = llm_client
        self._max_implications = max_implications

    async def generate(
        self,
        memory_text: str,
        memory_id: str | None = None,
        count: int | None = None,
    ) -> list[ProspectiveIndex]:
        """Generate prospective indexes for a single memory."""
        if not memory_text or not memory_text.strip():
            return []

        n = count or self._max_implications
        prompt = _PROSPECTIVE_PROMPT.format(
            memory_content=memory_text.strip(),
            count=n,
        )

        try:
            response = await self._llm.complete(
                prompt,
                temperature=0.7,
                max_tokens=600,
                system_prompt="You are a JSON generator. Always respond with a valid JSON array only, no markdown.",
            )
            implications = self._parse_implications(response)
            return [
                ProspectiveIndex(
                    implication=imp,
                    source_memory_text=memory_text,
                    source_memory_id=memory_id,
                )
                for imp in implications[:n]
            ]
        except Exception as exc:
            logger.warning(
                "prospective_indexing_failed",
                error=str(exc),
                memory_id=memory_id,
            )
            return []

    async def generate_batch(
        self,
        memories: list[dict[str, Any]],
        count: int | None = None,
    ) -> list[list[ProspectiveIndex]]:
        """Generate prospective indexes for a batch of memories.

        Each entry in `memories` should have keys: "text", and optionally "id".
        Returns a list of lists, one per input memory.
        """
        if not memories:
            return []

        if len(memories) == 1:
            result = await self.generate(
                memories[0].get("text", ""),
                memory_id=memories[0].get("id"),
                count=count,
            )
            return [result]

        n = count or self._max_implications
        block = "\n".join(f'[{i}] "{m.get("text", "").strip()}"' for i, m in enumerate(memories))
        prompt = _BATCH_PROSPECTIVE_PROMPT.format(
            count=n,
            memories_block=block,
        )

        try:
            raw = await self._llm.complete_json(prompt, temperature=0.7)
            if not isinstance(raw, dict):
                raw = json.loads(str(raw)) if isinstance(raw, str) else {}

            results: list[list[ProspectiveIndex]] = []
            for i, mem in enumerate(memories):
                entry = raw.get(str(i))
                if isinstance(entry, list):
                    implications = [s.strip() for s in entry if isinstance(s, str) and s.strip()][
                        :n
                    ]
                else:
                    implications = []
                results.append(
                    [
                        ProspectiveIndex(
                            implication=imp,
                            source_memory_text=mem.get("text", ""),
                            source_memory_id=mem.get("id"),
                        )
                        for imp in implications
                    ]
                )
            return results
        except Exception as exc:
            logger.warning(
                "prospective_indexing_batch_failed",
                error=str(exc),
                batch_size=len(memories),
            )
            return [[] for _ in memories]

    @staticmethod
    def _parse_implications(response: str) -> list[str]:
        """Parse LLM response into a list of implication strings."""
        raw = response.strip()
        # Strip markdown code fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [s.strip() for s in data if isinstance(s, str) and s.strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        return []
