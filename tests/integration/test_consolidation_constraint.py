"""Integration test: consolidation preserves constraint semantics.

When episodes contain explicit constraints (e.g. "I never eat shellfish"),
the consolidated gist/fact should retain the policy type and core meaning.

Note: Uses lazy imports to avoid circular import (orchestrator -> consolidation.worker).
"""

import json
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecordCreate, Provenance
from src.memory.neocortical.fact_store import SemanticFactStore
from src.memory.neocortical.store import NeocorticalStore
from src.storage.postgres import PostgresMemoryStore


class _MockGraphStore:
    async def merge_edge(self, *args, **kwargs):
        return "mock-edge-id"

    async def get_entity_facts(self, *args, **kwargs):
        return []

    async def personalized_pagerank(self, *args, **kwargs):
        return []


class _MockLLMConstraintPreserving:
    """Returns JSON that preserves policy constraint from shellfish episode."""

    async def complete(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 500, system_prompt=None
    ):
        if "shellfish" in prompt.lower() or "never" in prompt.lower():
            return json.dumps(
                {
                    "gist": "User never eats shellfish because of allergy",
                    "type": "policy",
                    "confidence": 0.9,
                    "subject": "user",
                    "predicate": "diet_restriction",
                    "value": "no shellfish",
                    "key": "user:policy:diet",
                }
            )
        return "not valid json"

    async def complete_json(self, prompt: str, schema=None, temperature: float = 0.0):
        return {}


@pytest.mark.asyncio
async def test_consolidation_preserves_constraint_policy(pg_session_factory):
    """Ingest episode with 'I never eat shellfish'; consolidate; assert
    neocortical fact retains policy type and core constraint meaning."""
    from src.consolidation.worker import ConsolidationWorker

    episodic = PostgresMemoryStore(pg_session_factory)
    fact_store = SemanticFactStore(pg_session_factory)
    neocortical = NeocorticalStore(graph_store=_MockGraphStore(), fact_store=fact_store)
    worker = ConsolidationWorker(
        episodic_store=episodic,
        neocortical_store=neocortical,
        llm_client=_MockLLMConstraintPreserving(),
    )
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    await episodic.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="I never eat shellfish because I'm allergic.",
            confidence=0.9,
            importance=0.8,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )

    report = await worker.consolidate(tenant_id, user_id)

    assert report.episodes_sampled >= 1
    assert report.gists_extracted >= 1
    assert report.migration.gists_processed >= 1

    profile = await neocortical.get_tenant_profile(tenant_id)
    policy_facts = profile.get("policy", {})
    assert len(policy_facts) >= 1, "Should have at least one policy fact"
    values = list(policy_facts.values())
    constraint_found = any(
        "shellfish" in str(v).lower() or "allergy" in str(v).lower() for v in values
    )
    assert constraint_found, (
        f"Policy facts should retain shellfish/allergy constraint. Got: {values}"
    )
