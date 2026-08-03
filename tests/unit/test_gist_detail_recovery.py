"""Detail recovery must be conditioned on the gist, and must not restate it.

A summary generalises, and generalising is exactly what drops the named entities and
quantities a later question asks about. The reported +5.72 on LoCoMo comes from a second
pass that reads the summary and hunts for what it *omitted* — not from running a second
independent extraction over the same dialogue, which is a strictly worse first pass.

So the two properties worth pinning are: the prompt actually carries the gist, and
output that merely restates the gist is dropped. A restated gist would be migrated as a
second semantic fact competing with the first, and gist-vs-source demotion only knows
how to demote *episodes*, so both would sit in the packet.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.consolidation.clusterer import EpisodeCluster
from src.consolidation.summarizer import ExtractedGist, GistExtractor
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance


def _record(text: str) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t",
        context_tags=[],
        type=MemoryType.EPISODIC_EVENT,
        text=text,
        confidence=0.9,
        importance=0.5,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(UTC),
        entities=[],
    )


def _cluster(*texts: str, cluster_id: int = 0) -> EpisodeCluster:
    episodes = [_record(t) for t in texts]
    return EpisodeCluster(
        cluster_id=cluster_id,
        episodes=episodes,
        avg_confidence=1.0,
    )


def _gist_for(cluster: EpisodeCluster, text: str = "User travels often") -> ExtractedGist:
    return ExtractedGist(
        text=text,
        gist_type="summary",
        confidence=0.8,
        supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
    )


def _extractor(payload):
    llm = AsyncMock()
    llm.complete_json = AsyncMock(return_value=payload)
    return GistExtractor(llm), llm


class TestItIsConditionedOnTheGist:
    @pytest.mark.asyncio
    async def test_the_prompt_carries_both_the_gist_and_the_source_text(self):
        cluster = _cluster("My sister Anna lives in Lisbon", "I fly to Lisbon twice a year")
        extractor, llm = _extractor({"gists": []})

        await extractor.recover_details([cluster], [_gist_for(cluster)])

        prompt = llm.complete_json.await_args.args[0]
        assert "User travels often" in prompt
        assert "Anna lives in Lisbon" in prompt

    @pytest.mark.asyncio
    async def test_recovered_details_are_returned(self):
        cluster = _cluster("My sister Anna lives in Lisbon")
        extractor, _ = _extractor(
            {
                "gists": [
                    {
                        "gist": "User's sister Anna lives in Lisbon",
                        "type": "fact",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        got = await extractor.recover_details([cluster], [_gist_for(cluster)])

        assert [g.text for g in got] == ["User's sister Anna lives in Lisbon"]
        assert got[0].supporting_episode_ids == [str(cluster.episodes[0].id)]

    @pytest.mark.asyncio
    async def test_a_cluster_with_no_gist_is_not_revisited(self):
        """With nothing to condition on this is just a second, worse gist extraction."""
        cluster = _cluster("Something nobody summarised")
        extractor, llm = _extractor({"gists": []})

        got = await extractor.recover_details([cluster], [])

        assert got == []
        llm.complete_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_call_per_cluster_not_per_gist(self):
        """Two gists over one cluster is one cluster's worth of source text to re-read."""
        cluster = _cluster("a", "b")
        gists = [_gist_for(cluster, "g1"), _gist_for(cluster, "g2")]
        extractor, llm = _extractor({"gists": []})

        await extractor.recover_details([cluster], gists)

        assert llm.complete_json.await_count == 1
        prompt = llm.complete_json.await_args.args[0]
        assert "g1" in prompt and "g2" in prompt


class TestRestatementIsDropped:
    @pytest.mark.asyncio
    async def test_an_exact_restatement_of_the_gist_is_dropped(self):
        cluster = _cluster("I travel a lot")
        extractor, _ = _extractor(
            {"gists": [{"gist": "User travels often", "type": "summary", "confidence": 0.8}]}
        )

        got = await extractor.recover_details([cluster], [_gist_for(cluster)])

        assert got == []

    @pytest.mark.asyncio
    async def test_the_comparison_ignores_case_and_padding(self):
        cluster = _cluster("I travel a lot")
        extractor, _ = _extractor(
            {"gists": [{"gist": "  user TRAVELS often ", "type": "summary", "confidence": 0.8}]}
        )

        got = await extractor.recover_details([cluster], [_gist_for(cluster)])

        assert got == []

    @pytest.mark.asyncio
    async def test_genuine_additions_survive_alongside_a_restatement(self):
        cluster = _cluster("My sister Anna lives in Lisbon")
        extractor, _ = _extractor(
            {
                "gists": [
                    {"gist": "User travels often", "type": "summary", "confidence": 0.8},
                    {"gist": "Anna lives in Lisbon", "type": "fact", "confidence": 0.9},
                ]
            }
        )

        got = await extractor.recover_details([cluster], [_gist_for(cluster)])

        assert [g.text for g in got] == ["Anna lives in Lisbon"]


class TestItDegradesQuietly:
    """Consolidation is a background job; a bad second pass must not lose the first."""

    @pytest.mark.asyncio
    async def test_an_llm_failure_yields_no_details_rather_than_raising(self):
        cluster = _cluster("a")
        llm = AsyncMock()
        llm.complete_json = AsyncMock(side_effect=RuntimeError("model down"))

        got = await GistExtractor(llm).recover_details([cluster], [_gist_for(cluster)])

        assert got == []

    @pytest.mark.asyncio
    async def test_one_failing_cluster_does_not_lose_the_others(self):
        good = _cluster("alpha-episode", cluster_id=0)
        bad = _cluster("bravo-episode", cluster_id=1)
        calls = {"n": 0}

        async def flaky(prompt, **kwargs):
            calls["n"] += 1
            if "bravo-episode" in prompt:
                raise RuntimeError("model down")
            return {"gists": [{"gist": "kept", "type": "fact", "confidence": 0.9}]}

        llm = AsyncMock()
        llm.complete_json = flaky

        got = await GistExtractor(llm).recover_details(
            [good, bad], [_gist_for(good), _gist_for(bad)]
        )

        assert calls["n"] == 2
        assert [g.text for g in got] == ["kept"]

    @pytest.mark.asyncio
    async def test_no_llm_configured_is_a_no_op(self):
        cluster = _cluster("a")
        assert await GistExtractor(None).recover_details([cluster], [_gist_for(cluster)]) == []


class TestItIsOffByDefault:
    def test_the_flag_defaults_off(self):
        """Unmeasured, and roughly one extra LLM call per cluster."""
        from src.core.config import FeatureFlags

        assert FeatureFlags().consolidation_detail_recovery_enabled is False
