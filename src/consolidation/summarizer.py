"""Gist extraction from episode clusters."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

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

DETAIL_RECOVERY_PROMPT = """You already summarised these memories. The summary is below.

SUMMARY ALREADY EXTRACTED:
{gists}

ORIGINAL MEMORIES:
{memories}

Find durable facts stated in the ORIGINAL MEMORIES that the SUMMARY does NOT capture.
A summary generalises, so specifics are routinely lost: named people, places and
organisations, quantities, relationships, commitments, and constraints.

Return JSON:
{{
  "gists": [
    {{"gist": "User's sister Anna lives in Lisbon", "type": "fact", "confidence": 0.9,
      "subject": "Anna", "predicate": "lives_in", "value": "Lisbon"}}
  ]
}}

Rules:
- Return ONLY information the summary omits. If it omits nothing, return {{"gists": []}}.
- Do not restate, rephrase or elaborate the summary.
- Each item must be supported by the original memories — never inferred beyond them.
- Skip episodic scaffolding (who spoke when, conversational filler).
- Use the same type vocabulary: fact, preference, pattern, summary, goal, value, state,
  causal, policy."""

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


class GistExtractor:
    """Extracts semantic gists from episode clusters via LLM JSON extraction."""

    def __init__(self, llm_client: LLMClient | None):
        self.llm = llm_client

    async def extract_gist(self, cluster: EpisodeCluster) -> list[ExtractedGist]:
        """Extract gists from a single cluster."""
        if not cluster.episodes:
            return []
        if self.llm is None:
            return []

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

        return self._gists_from_objects(data, cluster, source_types)

    @staticmethod
    def _gists_from_objects(
        data: Any,
        cluster: EpisodeCluster,
        source_types: list[str],
    ) -> list[ExtractedGist]:
        """Build gists from parsed LLM output. Shared by extraction and detail recovery,
        which return the same object shape."""
        gists: list[ExtractedGist] = []
        for gd in data if isinstance(data, list) else [data]:
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

    async def recover_details(
        self,
        clusters: list[EpisodeCluster],
        gists: list[ExtractedGist],
    ) -> list[ExtractedGist]:
        """Second pass: find durable facts the gist left out, *conditioned on* the gist.

        Gist abstraction and detail recovery are complementary — a summary generalises,
        and generalising is exactly what drops the named entities and quantities a later
        question asks about. Running the two independently does not work; the recovery
        pass has to see the summary to know what is already covered, which is why this
        takes the extracted gists rather than re-reading the cluster alone.

        Only clusters that produced a gist are revisited: with nothing to condition on
        this would be a second, worse gist extraction.

        ponytail: one LLM call per cluster, run concurrently — up to ``max_clusters``
        (20) extra calls per consolidation run, roughly doubling its LLM cost. Batch it
        the way ``_extract_gist_batch`` batches, if consolidation cost ever becomes the
        bottleneck rather than the write path.
        """
        if self.llm is None or not clusters or not gists:
            return []

        by_episode: dict[str, EpisodeCluster] = {
            str(ep.id): cluster for cluster in clusters for ep in cluster.episodes
        }
        grouped: dict[int, tuple[EpisodeCluster, list[ExtractedGist]]] = {}
        for gist in gists:
            for episode_id in gist.supporting_episode_ids:
                cluster = by_episode.get(str(episode_id))
                if cluster is not None:
                    grouped.setdefault(cluster.cluster_id, (cluster, []))[1].append(gist)
                    break

        if not grouped:
            return []

        results = await asyncio.gather(
            *[
                self._recover_one(cluster, cluster_gists)
                for cluster, cluster_gists in grouped.values()
            ],
            return_exceptions=True,
        )
        recovered: list[ExtractedGist] = []
        for result in results:
            if isinstance(result, list):
                recovered.extend(result)
        return recovered

    async def _recover_one(
        self,
        cluster: EpisodeCluster,
        gists: list[ExtractedGist],
    ) -> list[ExtractedGist]:
        if self.llm is None:
            return []
        source_types = self._cluster_source_types(cluster)
        memories = "\n".join(
            f"{i}. [{ep.type.value if hasattr(ep.type, 'value') else ep.type}] {ep.text}"
            for i, ep in enumerate(cluster.episodes[:10], 1)
        )
        prompt = DETAIL_RECOVERY_PROMPT.format(
            gists="\n".join(f"- {g.text}" for g in gists),
            memories=memories,
        )
        try:
            data = await self.llm.complete_json(prompt, temperature=0.0)
        except Exception:
            return []

        items = data.get("gists", []) if isinstance(data, dict) else data
        recovered = self._gists_from_objects(items, cluster, source_types)

        # A model that ignores "return only what the summary omits" restates it instead,
        # and a restatement would be migrated as a second fact competing with the first.
        already = {g.text.strip().lower() for g in gists}
        return [g for g in recovered if g.text.strip().lower() not in already]

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
