"""Gist extraction from episode clusters."""

import json
from dataclasses import dataclass
from typing import Any, List, Optional

from .clusterer import EpisodeCluster
from ..utils.llm import LLMClient

GIST_EXTRACTION_PROMPT = """Analyze these related memories and extract the key semantic information.

MEMORIES (from a conversation with a user):
{memories}

COMMON THEMES: {themes}

Extract:
1. The main fact or pattern these memories represent
2. The confidence level (how consistent/certain the info is)
3. The type: "fact" (definite info), "preference" (user likes/dislikes),
   "pattern" (behavioral tendency), or "summary" (general synopsis)
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
- Don't include episodic details (times, specific conversations)
- Focus on durable, generalizable information
- Higher confidence if multiple memories support the same conclusion"""


@dataclass
class ExtractedGist:
    """Extracted semantic gist from a cluster."""

    text: str
    gist_type: str  # "fact", "preference", "pattern", "summary"
    confidence: float
    supporting_episode_ids: List[str]

    key: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    value: Optional[Any] = None


class GistExtractor:
    """Extracts semantic gist from episode clusters."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def extract_gist(self, cluster: EpisodeCluster) -> List[ExtractedGist]:
        """Extract gists from a single cluster."""
        if not cluster.episodes:
            return []

        memory_texts = []
        for i, ep in enumerate(cluster.episodes[:10], 1):
            mem_type = ep.type.value if hasattr(ep.type, "value") else str(ep.type)
            memory_texts.append(f"{i}. [{mem_type}] {ep.text}")

        memories_str = "\n".join(memory_texts)
        themes_str = (
            ", ".join(cluster.common_entities) if cluster.common_entities else "none identified"
        )

        prompt = GIST_EXTRACTION_PROMPT.format(
            memories=memories_str,
            themes=themes_str,
        )

        try:
            response = await self.llm.complete(prompt, temperature=0.0)
            # Strip markdown code block if present
            raw = response.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines)
            data = json.loads(raw)

            if isinstance(data, list):
                gists_data = data
            else:
                gists_data = [data]

            gists = []
            for gd in gists_data:
                conf = float(gd.get("confidence", 0.7)) * cluster.avg_confidence
                gists.append(
                    ExtractedGist(
                        text=gd.get("gist", ""),
                        gist_type=gd.get("type", "summary"),
                        confidence=conf,
                        supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
                        key=gd.get("key"),
                        subject=gd.get("subject"),
                        predicate=gd.get("predicate"),
                        value=gd.get("value"),
                    )
                )
            return gists

        except (json.JSONDecodeError, KeyError, TypeError):
            return [
                ExtractedGist(
                    text=self._simple_summary(cluster),
                    gist_type="summary",
                    confidence=cluster.avg_confidence * 0.5,
                    supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
                )
            ]

    async def extract_from_clusters(self, clusters: List[EpisodeCluster]) -> List[ExtractedGist]:
        """Extract gists from all clusters."""
        import asyncio

        all_gists = []
        results = await asyncio.gather(*[self.extract_gist(c) for c in clusters])
        for gist_list in results:
            all_gists.extend(gist_list)
        return all_gists

    def _simple_summary(self, cluster: EpisodeCluster) -> str:
        if cluster.common_entities:
            return f"User discussed: {', '.join(cluster.common_entities[:3])}"
        if cluster.episodes:
            return cluster.episodes[0].text[:100]
        return "Cluster summary"
