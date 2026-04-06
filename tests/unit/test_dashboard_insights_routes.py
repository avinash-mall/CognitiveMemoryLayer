from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.api.dashboard import insights_routes
from src.core.enums import MemoryType
from src.memory.hippocampal.write_gate import WriteDecision

from .dashboard_support import (
    ADMIN_AUTH,
    NeoResultStub,
    NeoSessionStub,
    RedisStub,
    ResultStub,
    make_db,
)


def _request(*, orchestrator: object | None = None, db: object | None = None) -> SimpleNamespace:
    state = SimpleNamespace()
    if orchestrator is not None:
        state.orchestrator = orchestrator
    if db is not None:
        state.db = db
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _feature_flags(**overrides: object) -> SimpleNamespace:
    defaults = {
        "use_llm_enabled": False,
        "use_llm_memory_type": False,
        "use_llm_constraint_extractor": False,
        "use_llm_write_time_facts": False,
        "use_llm_write_gate_importance": False,
        "use_llm_context_tags": False,
        "use_llm_confidence": False,
        "use_llm_decay_rate": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_dashboard_retrieval_explain_returns_live_debug_payload() -> None:
    memory_id = uuid4()
    explained = {
        "query": "pizza",
        "analysis": {
            "intent": "fact_lookup",
            "confidence": 0.91,
            "entities": ["pizza"],
            "key_phrases": ["favorite food"],
            "time_reference": None,
            "time_start": None,
            "time_end": None,
            "suggested_sources": ["vector"],
            "suggested_top_k": 5,
            "query_domain": "preferences",
            "constraint_dimensions": ["food"],
            "is_decision_query": False,
            "metadata": {"planner": "hybrid"},
        },
        "plan_steps": [
            {
                "source": "vector",
                "priority": 1,
                "key": None,
                "query": "pizza",
                "seeds": [],
                "memory_types": ["semantic_fact"],
                "time_filter": None,
                "min_confidence": 0.0,
                "top_k": 5,
                "timeout_ms": 50,
                "skip_if_found": False,
                "associative_expansion": False,
                "query_domain": "preferences",
                "constraint_categories": ["food"],
            }
        ],
        "parallel_groups": [[0]],
        "execution_steps": [
            {
                "source": "vector",
                "success": True,
                "elapsed_ms": 12.5,
                "result_count": 1,
                "timed_out": False,
                "error": None,
                "filters": {"tenant_id": "tenant-a"},
                "query_preview": "pizza",
                "result_preview": [{"id": str(memory_id)}],
            }
        ],
        "retrieval_meta": {"planner": "hybrid"},
        "packet_warnings": ["conflicting recency cues"],
        "open_questions": ["No explicit date"],
        "llm_context": "1. Memory: pizza",
        "results": [
            {
                "id": str(memory_id),
                "text": "User likes pizza",
                "type": "semantic_fact",
                "confidence": 0.9,
                "relevance_score": 0.88,
                "retrieval_source": "vector",
                "timestamp": datetime.now(UTC),
                "supersedes_id": None,
                "metadata": {"source": "test"},
            }
        ],
        "rerank": [
            {
                "id": str(memory_id),
                "text": "User likes pizza",
                "source_type": "semantic_fact",
                "retrieval_source": "vector",
                "rank": 1,
                "selected": True,
                "final_score": 0.93,
                "breakdown": {"retrieval": 0.7, "recency": 0.23},
                "notes": ["kept after diversity pass"],
            }
        ],
        "total_count": 1,
    }
    orchestrator = SimpleNamespace(
        retriever=SimpleNamespace(explain=AsyncMock(return_value=explained))
    )

    result = await insights_routes.dashboard_retrieval_explain(
        body=insights_routes.DashboardRetrievalRequest(
            tenant_id="tenant-a",
            query="pizza",
            max_results=5,
        ),
        request=_request(orchestrator=orchestrator),
        auth=ADMIN_AUTH,
    )

    assert result.query == "pizza"
    assert result.analysis.intent == "fact_lookup"
    assert result.execution_steps[0].result_count == 1
    assert result.rerank[0].breakdown["retrieval"] == 0.7
    assert result.total_count == 1
    assert result.elapsed_ms >= 0


@pytest.mark.asyncio
async def test_dashboard_write_simulate_returns_chunk_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    chunk = SimpleNamespace(
        id="chunk-1",
        text="Alice likes pizza",
        chunk_type=SimpleNamespace(value="statement"),
        salience=0.74,
        confidence=0.81,
        timestamp=now,
    )

    class _Chunker:
        def chunk(
            self,
            content: str,
            *,
            turn_id: str | None = None,
            role: str | None = None,
            timestamp: datetime | None = None,
        ) -> list[SimpleNamespace]:
            assert content == "Alice likes pizza"
            assert role == "user"
            assert timestamp == now
            return [chunk]

    class _WriteGate:
        def evaluate(
            self,
            input_chunk: object,
            *,
            existing_memories: list[dict[str, str]],
            unified_result: object | None = None,
        ) -> SimpleNamespace:
            assert input_chunk is chunk
            assert existing_memories == [{"text": "Older memory"}]
            assert unified_result is None
            return SimpleNamespace(
                decision=WriteDecision.STORE,
                memory_types=[MemoryType.SEMANTIC_FACT],
                importance=0.72,
                novelty=0.66,
                risk_flags=["novel"],
                redaction_required=False,
                reason="Useful fact",
            )

    class _FactExtractor:
        def extract(self, input_chunk: object) -> list[SimpleNamespace]:
            assert input_chunk is chunk
            return [
                SimpleNamespace(
                    key="pref:food",
                    category="preference",
                    predicate="likes",
                    value="pizza",
                    confidence=0.8,
                )
            ]

    monkeypatch.setattr(
        insights_routes,
        "get_settings",
        lambda: SimpleNamespace(features=_feature_flags()),
    )
    monkeypatch.setattr(insights_routes, "_ner_entities_for_text", lambda text: [{"text": text}])
    monkeypatch.setattr(
        insights_routes,
        "_ner_relations_for_text",
        lambda text: [{"predicate": "likes", "text": text}],
    )
    monkeypatch.setattr(insights_routes, "WriteTimeFactExtractor", lambda: _FactExtractor())

    store = SimpleNamespace(scan=AsyncMock(return_value=[SimpleNamespace(text="Older memory")]))
    orchestrator = SimpleNamespace(
        hippocampal=SimpleNamespace(
            store=store,
            local_extractor=None,
            unified_extractor=None,
            write_gate=_WriteGate(),
            redactor=SimpleNamespace(redact=lambda *args, **kwargs: None),
            entity_extractor=None,
            relation_extractor=None,
            constraint_extractor=SimpleNamespace(extract=lambda _: []),
            _use_unified_write_path=lambda: False,
            _generate_key=lambda *_args: "pref:food",
        ),
        short_term=SimpleNamespace(working=SimpleNamespace(chunker=_Chunker())),
    )

    result = await insights_routes.dashboard_write_simulate(
        body=insights_routes.DashboardWriteSimulationRequest(
            tenant_id="tenant-a",
            content="Alice likes pizza",
            timestamp=now,
        ),
        request=_request(orchestrator=orchestrator),
        auth=ADMIN_AUTH,
    )

    assert result.summary["chunk_count"] == 1
    assert result.summary["would_store_count"] == 1
    assert result.chunks[0].would_store is True
    assert result.chunks[0].chosen_memory_type == "semantic_fact"
    assert result.chunks[0].key == "pref:food"
    assert result.chunks[0].extracted_facts[0]["value"] == "pizza"


@pytest.mark.asyncio
async def test_dashboard_forgetting_preview_surfaces_scores_and_duplicates() -> None:
    keep_id = uuid4()
    duplicate_id = uuid4()
    memories = [
        SimpleNamespace(
            id=keep_id,
            text="Remember the preference",
            type="preference",
            status="active",
            confidence=0.9,
            importance=0.85,
            access_count=4,
        ),
        SimpleNamespace(
            id=duplicate_id,
            text="Duplicate statement",
            type="semantic_fact",
            status="active",
            confidence=0.5,
            importance=0.4,
            access_count=1,
        ),
    ]
    scores = [
        SimpleNamespace(
            memory_id=str(duplicate_id),
            total_score=0.15,
            importance_score=0.2,
            recency_score=0.1,
            frequency_score=0.1,
            confidence_score=0.2,
            type_bonus_score=0.0,
            dependency_score=0.0,
            suggested_action="delete",
        ),
        SimpleNamespace(
            memory_id=str(keep_id),
            total_score=0.82,
            importance_score=0.9,
            recency_score=0.6,
            frequency_score=0.8,
            confidence_score=0.9,
            type_bonus_score=0.4,
            dependency_score=0.3,
            suggested_action="keep",
        ),
    ]
    duplicates = [
        SimpleNamespace(
            memory_id=str(duplicate_id),
            interfering_memory_id=str(keep_id),
            similarity=0.94,
            interference_type="duplicate",
            recommendation="delete_older",
            keep_id=str(keep_id),
        )
    ]
    worker = SimpleNamespace(
        store=SimpleNamespace(scan=AsyncMock(return_value=memories)),
        _get_dependency_counts=AsyncMock(return_value={str(keep_id): 2, str(duplicate_id): 0}),
        scorer=SimpleNamespace(
            score_batch=lambda *_args: scores,
            _PROTECTED_TYPES=frozenset({"preference"}),
        ),
        interference=SimpleNamespace(detect_duplicates=lambda *_args: duplicates),
        policy=SimpleNamespace(plan_operations=lambda scored: [{"memory_id": scored[0].memory_id}]),
    )

    result = await insights_routes.dashboard_forgetting_preview(
        body=insights_routes.DashboardForgettingPreviewRequest(
            tenant_id="tenant-a",
            user_id="user-1",
            max_memories=10,
        ),
        request=_request(orchestrator=SimpleNamespace(forgetting=worker)),
        auth=ADMIN_AUTH,
    )

    items = {str(item.memory_id): item for item in result.items}
    assert result.scanned_count == 2
    assert result.duplicates_found == 1
    assert result.operations_planned == 1
    assert items[str(keep_id)].protected is True
    assert items[str(duplicate_id)].duplicate_matches[0]["memory_id"] == str(keep_id)
    assert result.summary == {"delete": 1, "keep": 1}


@pytest.mark.asyncio
async def test_dashboard_reconsolidation_sessions_reads_active_redis_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    redis = RedisStub(
        scan_results=[(0, ["labile:scope:tenant-a:scope-1"])],
        lrange_values={"labile:scope:tenant-a:scope-1": ["sess-1"]},
        values={"labile:session:sess-1": json.dumps({"session": "payload"})},
    )
    db, _ = make_db(redis=redis)
    monkeypatch.setattr(
        insights_routes,
        "_deserialize_session",
        lambda payload: SimpleNamespace(
            tenant_id="tenant-a",
            scope_id="scope-1",
            turn_id="turn-1",
            created_at=now,
            query="What food does the user like?",
            retrieved_texts=("User likes pizza",),
            memories={
                "mem-1": SimpleNamespace(
                    memory_id=uuid4(),
                    retrieved_at=now,
                    relevance_score=0.88,
                    original_confidence=0.77,
                    context="retrieval",
                    expires_at=now + timedelta(minutes=5),
                )
            },
        ),
    )

    result = await insights_routes.dashboard_reconsolidation_sessions(
        tenant_id="tenant-a",
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.total == 1
    assert result.items[0].session_id == "sess-1"
    assert result.items[0].tenant_id == "tenant-a"
    assert result.items[0].memories[0]["context"] == "retrieval"


@pytest.mark.asyncio
async def test_dashboard_quality_overview_aggregates_issues_and_graph_hygiene() -> None:
    low_conf_memory_id = uuid4()
    low_evidence_fact_id = uuid4()
    db, _ = make_db(
        pg_results=[
            ResultStub(scalar_rows=[SimpleNamespace(id=low_conf_memory_id)]),
            ResultStub(scalar=1),
            ResultStub(scalar_rows=[SimpleNamespace(id=low_evidence_fact_id)]),
            ResultStub(scalar=2),
            ResultStub(scalar=1),
            ResultStub(scalar=1),
            ResultStub(scalar=1),
            ResultStub(scalar=2),
            ResultStub(all_rows=[("user:name", 3)]),
        ],
        neo_session=NeoSessionStub(
            [
                NeoResultStub(single={"duplicate_groups": 2}),
                NeoResultStub(single={"stale_nodes": 4}),
            ]
        ),
    )

    result = await insights_routes.dashboard_quality_overview(
        tenant_id="tenant-a",
        auth=ADMIN_AUTH,
        db=db,
    )

    issues = {issue.label: issue for issue in result.issues}
    assert issues["low_confidence_memories"].sample_ids == [str(low_conf_memory_id)]
    assert issues["low_evidence_facts"].sample_ids == [str(low_evidence_fact_id)]
    assert issues["orphan_facts"].count == 1
    assert result.invalidation_hotspots == [{"key": "user:name", "count": 3}]
    assert result.graph_hygiene == {"duplicate_entity_groups": 2, "stale_nodes": 4}
    assert result.labile_sessions == 0


@pytest.mark.asyncio
async def test_dashboard_ops_metrics_summarizes_registry_and_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        SimpleNamespace(name="cml_retrieval_timeout_total", labels={}, value=2),
        SimpleNamespace(name="cml_retrieval_step_failures_total", labels={}, value=1),
        SimpleNamespace(name="memory_reads_total", labels={}, value=5),
        SimpleNamespace(name="memory_writes_total", labels={}, value=7),
        SimpleNamespace(name="cml_db_pool_checked_out", labels={}, value=3),
        SimpleNamespace(name="cml_retrieval_fact_hit_total", labels={"hit": "yes"}, value=3),
        SimpleNamespace(name="cml_retrieval_fact_hit_total", labels={"hit": "no"}, value=1),
    ]
    family = SimpleNamespace(name="cml_dashboard_metrics", samples=samples)
    monkeypatch.setattr(insights_routes.REGISTRY, "collect", lambda: [family])
    db, _ = make_db(
        redis=RedisStub(info_value={"used_memory": 3 * 1024 * 1024}, db_size=9)
    )

    result = await insights_routes.dashboard_ops_metrics(auth=ADMIN_AUTH, db=db)

    assert result.highlights["retrieval_timeouts_total"] == 2.0
    assert result.highlights["retrieval_step_failures_total"] == 1.0
    assert result.highlights["memory_reads_total"] == 5.0
    assert result.highlights["memory_writes_total"] == 7.0
    assert result.highlights["fact_hit_rate"] == 0.75
    assert result.highlights["redis_keys"] == 9
    assert result.highlights["redis_used_memory_mb"] == 3.0
    assert len(result.metrics) == 7


@pytest.mark.asyncio
async def test_dashboard_evaluation_summary_discovers_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    evaluation_dir = tmp_path / "evaluation"
    outputs_dir = evaluation_dir / "outputs"
    data_dir = evaluation_dir / "locomo_plus" / "data"
    scripts_dir = evaluation_dir / "scripts"
    outputs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)

    summary_path = outputs_dir / "locomo_plus_qa_cml_judge_summary.json"
    summary_path.write_text(json.dumps({"score": 0.91}), encoding="utf-8")
    (outputs_dir / "extra.txt").write_text("artifact", encoding="utf-8")
    (evaluation_dir / "COMPARISON.md").write_text("Latest comparison", encoding="utf-8")
    (data_dir / "locomo10.json").write_text("[]", encoding="utf-8")
    (data_dir / "unified_input_samples_v2.json").write_text("[]", encoding="utf-8")
    (scripts_dir / "run_full_eval.py").write_text("print('ok')", encoding="utf-8")

    monkeypatch.setattr(insights_routes, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        insights_routes,
        "get_settings",
        lambda: SimpleNamespace(llm_eval=SimpleNamespace(provider="mock", model="judge-model")),
    )

    result = await insights_routes.dashboard_evaluation_summary(auth=ADMIN_AUTH)

    assert result.readiness["evaluation_dir_present"] is True
    assert result.readiness["outputs_dir_present"] is True
    assert result.readiness["scripts_present"] is True
    assert result.readiness["locomo_data_present"] is True
    assert result.readiness["unified_samples_present"] is True
    assert result.readiness["llm_eval_provider"] == "mock"
    assert result.latest_summary == {"score": 0.91}
    assert result.latest_report == "Latest comparison"
    assert {artifact.name for artifact in result.artifacts} == {
        "extra.txt",
        "locomo_plus_qa_cml_judge_summary.json",
    }
