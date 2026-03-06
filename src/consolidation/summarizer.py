"""Gist extraction from episode clusters."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..utils.llm import LLMClient
from .clusterer import EpisodeCluster

GIST_EXTRACTION_PROMPT = """Analyze these related memories and extract the key semantic information.

MEMORIES (from a conversation with a user):
{memories}

COMMON THEMES: {themes}
SOURCE MEMORY TYPES: {source_types}

Extract:
1. The main fact or pattern these memories represent
2. The confidence level (how consistent/certain the info is)
3. The type: one of:
   - "fact" (definite info)
   - "preference" (user likes/dislikes)
   - "pattern" (behavioral tendency)
   - "summary" (general synopsis)
   - "goal" (something the user is working toward or trying to achieve)
   - "value" (something the user considers important or prioritizes)
   - "state" (a current condition or situation the user is in)
   - "causal" (a reason or explanation for user behavior)
   - "policy" (a personal rule the user follows, e.g. "I never...", "I always...")
4. A structured representation if possible (subject, predicate, value)

Return JSON:
{{
  "gist": "User prefers vegetarian food",
  "type": "preference",
  "confidence": 0.9,
  "subject": "user",
  "predicate": "food_preference",
  "value": "vegetarian",
  "key": "user:preference:food"
}}

Rules:
- Combine information across memories to get the core meaning
- Do not include episodic details (times, specific conversations)
- Focus on durable, generalizable information
- Higher confidence if multiple memories support the same conclusion
- Use "goal"/"value"/"state"/"causal"/"policy" types when memories express constraints, commitments, or conditions that should govern future behavior
- IMPORTANT: If source memories include "constraint" type, you MUST classify the gist as "goal", "value", "state", "causal", or "policy" — preserving the constraint-governing nature of the original information. Do NOT downgrade constraints to "fact" or "summary"."""

_BATCH_GIST_CLUSTER_SIZE = 8

BATCH_GIST_EXTRACTION_PROMPT = """Analyze these memory clusters and extract semantic gists for EACH cluster.

CLUSTERS:
{clusters}

For each cluster, return one or more gist objects with:
- gist
- type ("fact","preference","pattern","summary","goal","value","state","causal","policy")
- confidence (0.0-1.0)
- optional: subject, predicate, value, key

Return JSON with this exact shape:
{{
  "clusters": [
    {{"cluster_index": 0, "gists": [{{"gist": "...", "type": "...", "confidence": 0.8}}]}},
    {{"cluster_index": 1, "gists": [{{"gist": "...", "type": "...", "confidence": 0.7}}]}}
  ]
}}
"""


@dataclass
class ExtractedGist:
    """Extracted semantic gist from a cluster."""

    text: str
    gist_type: str
    confidence: float
    supporting_episode_ids: list[str]

    key: str | None = None
    subject: str | None = None
    predicate: str | None = None
    value: Any | None = None
    source_memory_types: list[str] | None = None


class SummarizerBackend(Protocol):
    """Async summarizer backend contract used in non-LLM mode."""

    async def summarize(self, text: str, *, max_chars: int | None = None) -> str: ...


class GistExtractor:
    """Extracts semantic gists from episode clusters.

    Primary path: LLM JSON extraction.
    Fallback path: local summarizer backend (e.g., Hugging Face model).
    """

    def __init__(
        self,
        llm_client: LLMClient | None,
        fallback_summarizer: SummarizerBackend | None = None,
    ):
        self.llm = llm_client
        self.fallback_summarizer = fallback_summarizer

    async def extract_gist(self, cluster: EpisodeCluster) -> list[ExtractedGist]:
        """Extract gists from a single cluster."""
        if not cluster.episodes:
            return []
        if self.llm is None:
            return await self._fallback_extract_gist(cluster)

        memory_texts = []
        source_types = self._cluster_source_types(cluster)
        for i, ep in enumerate(cluster.episodes[:10], 1):
            mem_type = ep.type.value if hasattr(ep.type, "value") else str(ep.type)
            memory_texts.append(f"{i}. [{mem_type}] {ep.text}")

        memories_str = "\n".join(memory_texts)
        themes_str = (
            ", ".join(cluster.common_entities) if cluster.common_entities else "none identified"
        )
        source_types_str = ", ".join(source_types) if source_types else "unknown"

        prompt = GIST_EXTRACTION_PROMPT.format(
            memories=memories_str,
            themes=themes_str,
            source_types=source_types_str,
        )

        try:
            response = await self.llm.complete(prompt, temperature=0.0)
            raw = response.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines)
            data = json.loads(raw)
        except Exception:
            return []

        gists_data = data if isinstance(data, list) else [data]
        gists: list[ExtractedGist] = []
        for gd in gists_data:
            if not isinstance(gd, dict):
                continue
            gist_text = str(gd.get("gist", "")).strip()
            if not gist_text:
                continue
            gist_type = str(gd.get("type", "summary")).strip().lower() or "summary"
            confidence = float(gd.get("confidence", 0.7)) * cluster.avg_confidence
            gists.append(
                ExtractedGist(
                    text=gist_text,
                    gist_type=gist_type,
                    confidence=confidence,
                    supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
                    key=gd.get("key"),
                    subject=gd.get("subject"),
                    predicate=gd.get("predicate"),
                    value=gd.get("value"),
                    source_memory_types=source_types,
                )
            )

        return gists

    async def extract_from_clusters(self, clusters: list[EpisodeCluster]) -> list[ExtractedGist]:
        """Extract gists from all clusters."""
        all_gists: list[ExtractedGist] = []
        if not clusters:
            return all_gists
        if self.llm is None:
            fallback = await asyncio.gather(*[self.extract_gist(c) for c in clusters])
            for gist_list in fallback:
                all_gists.extend(gist_list)
            return all_gists

        if len(clusters) == 1:
            return await self.extract_gist(clusters[0])

        for i in range(0, len(clusters), _BATCH_GIST_CLUSTER_SIZE):
            batch = clusters[i : i + _BATCH_GIST_CLUSTER_SIZE]
            batch_result = await self._extract_gist_batch(batch)
            if batch_result is None:
                fallback = await asyncio.gather(*[self.extract_gist(c) for c in batch])
                for gist_list in fallback:
                    all_gists.extend(gist_list)
            else:
                all_gists.extend(batch_result)
        return all_gists

    async def _extract_gist_batch(
        self, clusters: list[EpisodeCluster]
    ) -> list[ExtractedGist] | None:
        """Batch extract multiple clusters in one LLM call."""
        if self.llm is None or not clusters:
            return None

        lines: list[str] = []
        for idx, cluster in enumerate(clusters):
            source_types = self._cluster_source_types(cluster)
            themes = ", ".join(cluster.common_entities) if cluster.common_entities else "none"
            lines.append(f"Cluster {idx}:")
            lines.append(f"Source types: {', '.join(source_types) if source_types else 'unknown'}")
            lines.append(f"Themes: {themes}")
            for j, ep in enumerate(cluster.episodes[:8], 1):
                mem_type = ep.type.value if hasattr(ep.type, "value") else str(ep.type)
                lines.append(f"{j}. [{mem_type}] {ep.text}")
            lines.append("")

        prompt = BATCH_GIST_EXTRACTION_PROMPT.format(clusters="\n".join(lines))

        try:
            data = await self.llm.complete_json(prompt, temperature=0.0)
        except Exception:
            return None

        clusters_out = data.get("clusters", []) if isinstance(data, dict) else []
        if not isinstance(clusters_out, list):
            return None

        out: list[ExtractedGist] = []
        for cluster_obj in clusters_out:
            if not isinstance(cluster_obj, dict):
                continue
            cluster_idx = cluster_obj.get("cluster_index")
            if not isinstance(cluster_idx, int) or cluster_idx < 0 or cluster_idx >= len(clusters):
                continue
            cluster = clusters[cluster_idx]
            source_types = self._cluster_source_types(cluster)

            gists_data = cluster_obj.get("gists", [])
            if not isinstance(gists_data, list):
                continue

            for gd in gists_data:
                if not isinstance(gd, dict):
                    continue
                gist_text = str(gd.get("gist", "")).strip()
                if not gist_text:
                    continue
                gist_type = str(gd.get("type", "summary")).strip().lower() or "summary"
                confidence = float(gd.get("confidence", 0.7)) * cluster.avg_confidence
                out.append(
                    ExtractedGist(
                        text=gist_text,
                        gist_type=gist_type,
                        confidence=confidence,
                        supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
                        key=gd.get("key"),
                        subject=gd.get("subject"),
                        predicate=gd.get("predicate"),
                        value=gd.get("value"),
                        source_memory_types=source_types,
                    )
                )

        return out or None

    @staticmethod
    def _cluster_source_types(cluster: EpisodeCluster) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    (ep.type.value if hasattr(ep.type, "value") else str(ep.type)).lower()
                    for ep in cluster.episodes
                ]
            )
        )

    async def _fallback_extract_gist(self, cluster: EpisodeCluster) -> list[ExtractedGist]:
        """Fallback gist extraction using local summarizer backend."""
        if self.fallback_summarizer is None:
            return []

        source_types = self._cluster_source_types(cluster)
        lines: list[str] = []
        for i, ep in enumerate(cluster.episodes[:8], 1):
            mem_type = ep.type.value if hasattr(ep.type, "value") else str(ep.type)
            lines.append(f"{i}. [{mem_type}] {ep.text}")

        combined = "\n".join(lines)
        try:
            gist_text = await self.fallback_summarizer.summarize(combined, max_chars=None)
        except Exception:
            gist_text = ""

        if not gist_text:
            gist_text = cluster.episodes[0].text[:320]

        gist_type = self._fallback_gist_type(source_types)
        confidence = max(0.45, min(0.85, cluster.avg_confidence * 0.8))
        return [
            ExtractedGist(
                text=gist_text,
                gist_type=gist_type,
                confidence=confidence,
                supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
                source_memory_types=source_types,
            )
        ]

    @staticmethod
    def _fallback_gist_type(source_types: list[str]) -> str:
        constraint_subtypes = {"goal", "state", "value", "causal", "policy"}
        matched = [t for t in source_types if t in constraint_subtypes]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            # Multiple constraint subtypes in one cluster -- prefer the most
            # specific first occurrence rather than collapsing to "policy".
            return matched[0]
        if "constraint" in source_types:
            # Untyped constraint clusters still need a concrete cognitive
            # subtype so alignment can preserve governing semantics.
            return "policy"
        if "preference" in source_types:
            return "preference"
        if "semantic_fact" in source_types:
            return "fact"
        return "summary"
