"""Unit tests for non-LLM gist extraction fallback summarizer path."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.consolidation.clusterer import EpisodeCluster
from src.consolidation.summarizer import GistExtractor
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance


def _record(text: str, mem_type: MemoryType = MemoryType.EPISODIC_EVENT) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t",
        context_tags=[],
        type=mem_type,
        text=text,
        confidence=0.9,
        importance=0.7,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(UTC),
    )


class _FallbackSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def summarize(self, text: str, *, max_chars: int | None = None) -> str:
        self.calls += 1
        _ = max_chars
        return f"summary::{text[:40]}"


@pytest.mark.asyncio
async def test_extract_gist_uses_fallback_when_llm_none():
    fallback = _FallbackSummarizer()
    extractor = GistExtractor(llm_client=None, fallback_summarizer=fallback)
    cluster = EpisodeCluster(
        cluster_id=0,
        episodes=[_record("User likes hiking and long walks.")],
        avg_confidence=0.8,
    )

    gists = await extractor.extract_gist(cluster)
    assert len(gists) == 1
    assert gists[0].text.startswith("summary::")
    assert gists[0].gist_type == "summary"
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_constraint_cluster_maps_to_policy_in_fallback():
    fallback = _FallbackSummarizer()
    extractor = GistExtractor(llm_client=None, fallback_summarizer=fallback)
    cluster = EpisodeCluster(
        cluster_id=0,
        episodes=[_record("I never eat shellfish.", MemoryType.CONSTRAINT)],
        avg_confidence=0.9,
    )

    gists = await extractor.extract_gist(cluster)
    assert len(gists) == 1
    assert gists[0].gist_type == "policy"


@pytest.mark.asyncio
async def test_extract_from_clusters_fallback_batches_by_cluster():
    fallback = _FallbackSummarizer()
    extractor = GistExtractor(llm_client=None, fallback_summarizer=fallback)
    clusters = [
        EpisodeCluster(cluster_id=0, episodes=[_record("A")], avg_confidence=0.8),
        EpisodeCluster(cluster_id=1, episodes=[_record("B")], avg_confidence=0.8),
    ]

    gists = await extractor.extract_from_clusters(clusters)
    assert len(gists) == 2
    assert fallback.calls == 2
