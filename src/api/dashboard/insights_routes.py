"""Dashboard explainability, quality, operations, and workbench routes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from prometheus_client import REGISTRY
from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from ...core.config import get_settings
from ...core.enums import MemoryStatus, MemoryType
from ...extraction.constraint_extractor import ConstraintExtractor
from ...extraction.local_unified_extractor import LocalUnifiedWriteExtractor
from ...extraction.write_time_facts import WriteTimeFactExtractor
from ...memory.hippocampal.store import _ner_entities_for_text, _ner_relations_for_text
from ...memory.hippocampal.write_gate import WriteDecision
from ...reconsolidation.labile_tracker import _deserialize_session
from ...storage.connection import DatabaseManager
from ...storage.models import MemoryRecordModel, SemanticFactModel
from ..auth import AuthContext, require_admin_permission
from ._shared import (
    DashboardEvaluationArtifact,
    DashboardEvaluationSummary,
    DashboardForgettingPreviewItem,
    DashboardForgettingPreviewRequest,
    DashboardForgettingPreviewResponse,
    DashboardOpsMetric,
    DashboardOpsResponse,
    DashboardQualityIssue,
    DashboardQualityResponse,
    DashboardReconsolidationSessionItem,
    DashboardReconsolidationSessionsResponse,
    DashboardRetrievalExplainResponse,
    DashboardRetrievalRequest,
    DashboardWriteSimulationChunk,
    DashboardWriteSimulationRequest,
    DashboardWriteSimulationResponse,
    _get_db,
    logger,
)

router = APIRouter()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _safe_model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return cast("dict[str, Any]", asdict(cast("Any", value)))
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


def _serialize_fact(value: Any) -> dict[str, Any]:
    category = getattr(value, "category", None)
    return {
        "key": getattr(value, "key", ""),
        "category": _enum_value(category),
        "predicate": getattr(value, "predicate", ""),
        "value": getattr(value, "value", ""),
        "confidence": getattr(value, "confidence", 0.0),
    }


def _serialize_redaction(redaction: tuple[str, str, int, int]) -> dict[str, Any]:
    pii_type, original, start, end = redaction
    return {
        "pii_type": pii_type,
        "original": original,
        "start": start,
        "end": end,
    }


def _serialize_memory_summary(record: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(record, "id", "")),
        "text": getattr(record, "text", ""),
        "type": _enum_value(getattr(record, "type", "")),
        "status": _enum_value(getattr(record, "status", "")),
        "confidence": getattr(record, "confidence", 0.0),
        "importance": getattr(record, "importance", 0.0),
        "timestamp": getattr(record, "timestamp", None),
        "written_at": getattr(record, "written_at", None),
        "source_session_id": getattr(record, "source_session_id", None),
        "supersedes_id": getattr(record, "supersedes_id", None),
        "metadata": getattr(record, "metadata", {}) or {},
    }


def _sample_ids(rows: Sequence[Any], *, limit: int = 5) -> list[str]:
    out = []
    for row in rows[:limit]:
        rid = getattr(row, "id", None)
        if rid is not None:
            out.append(str(rid))
    return out


async def _load_labile_sessions(
    db: DatabaseManager,
    tenant_id: str | None = None,
) -> list[DashboardReconsolidationSessionItem]:
    """Read active labile sessions directly from Redis for the workbench."""
    if not db.redis:
        return []

    items: list[DashboardReconsolidationSessionItem] = []
    cursor = 0
    match_pattern = f"labile:scope:{tenant_id}:*" if tenant_id else "labile:scope:*"
    while True:
        cursor, keys = await db.redis.scan(cursor, match=match_pattern, count=200)
        for key in keys:
            scope_key = key.decode() if isinstance(key, bytes) else key
            session_keys = await db.redis.lrange(scope_key, 0, -1)
            for session_key in session_keys:
                session_key_str = (
                    session_key.decode() if isinstance(session_key, bytes) else session_key
                )
                raw = await db.redis.get(f"labile:session:{session_key_str}")
                if not raw:
                    continue
                try:
                    payload = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                    session = _deserialize_session(payload)
                except Exception as exc:
                    logger.warning("dashboard_labile_session_parse_failed", error=str(exc))
                    continue
                expires_at = None
                if session.memories:
                    expires_at = max(memory.expires_at for memory in session.memories.values())
                items.append(
                    DashboardReconsolidationSessionItem(
                        session_id=session_key_str,
                        tenant_id=session.tenant_id,
                        scope_id=session.scope_id,
                        turn_id=session.turn_id,
                        created_at=session.created_at,
                        expires_at=expires_at,
                        query=session.query,
                        retrieved_texts=list(session.retrieved_texts),
                        memories=[
                            {
                                "memory_id": str(memory.memory_id),
                                "retrieved_at": memory.retrieved_at,
                                "relevance_score": memory.relevance_score,
                                "original_confidence": memory.original_confidence,
                                "context": memory.context,
                            }
                            for memory in session.memories.values()
                        ],
                    )
                )
        if cursor == 0:
            break
    items.sort(key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC), reverse=True)
    return items


@router.post("/retrieval/explain", response_model=DashboardRetrievalExplainResponse)
async def dashboard_retrieval_explain(
    body: DashboardRetrievalRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Run retrieval in explain mode using the live classifier, planner, retriever, and reranker."""
    _ = auth
    try:
        orchestrator = request.app.state.orchestrator
        started = datetime.now(UTC)
        explained = await orchestrator.retriever.explain(
            tenant_id=body.tenant_id,
            query=body.query,
            max_results=body.max_results,
            context_filter=body.context_filter,
            memory_types=body.memory_types,
        )
        explained["elapsed_ms"] = round(
            (datetime.now(UTC) - started).total_seconds() * 1000,
            2,
        )
        return DashboardRetrievalExplainResponse(**explained)
    except Exception as exc:
        logger.error("dashboard_retrieval_explain_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/write/simulate", response_model=DashboardWriteSimulationResponse)
async def dashboard_write_simulate(
    body: DashboardWriteSimulationRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Dry-run the write path without mutating long-term memory."""
    _ = auth
    try:
        orchestrator = request.app.state.orchestrator
        hippocampal = orchestrator.hippocampal
        chunker = orchestrator.short_term.working.chunker
        write_fact_extractor = WriteTimeFactExtractor()

        memory_type_override = None
        if body.memory_type:
            try:
                memory_type_override = MemoryType(body.memory_type)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Unknown memory_type: {body.memory_type}") from exc

        chunks = chunker.chunk(
            body.content,
            turn_id=body.metadata.get("turn_id"),
            role=body.metadata.get("source_role", "user"),
            timestamp=body.timestamp,
        )
        existing = await hippocampal.store.scan(
            body.tenant_id,
            filters={"status": MemoryStatus.ACTIVE.value},
            limit=50,
            order_by="-timestamp",
        )
        existing_dicts = [{"text": memory.text} for memory in existing]

        configured_uses_unified = (
            bool(hippocampal._use_unified_write_path()) and hippocampal.unified_extractor is not None
        )
        compare_local = bool(body.compare_extractors)
        local_extractor = hippocampal.local_extractor
        if compare_local and local_extractor is None:
            candidate = LocalUnifiedWriteExtractor()
            local_extractor = candidate if candidate.available else None

        chunk_results: list[DashboardWriteSimulationChunk] = []
        store_count = 0
        skip_count = 0

        for chunk in chunks:
            unified_result = None
            if configured_uses_unified and hippocampal.unified_extractor:
                unified_result = await hippocampal.unified_extractor.extract(chunk)

            local_result = None
            if local_extractor and local_extractor.available:
                local_result = await local_extractor.extract(chunk.text)

            gate_result = hippocampal.write_gate.evaluate(
                chunk,
                existing_memories=existing_dicts,
                unified_result=unified_result if configured_uses_unified else None,
            )

            redaction_result = None
            pii_spans = None
            if unified_result and getattr(unified_result, "pii_spans", None):
                pii_spans = [(span.start, span.end, span.pii_type) for span in unified_result.pii_spans]

            if gate_result.redaction_required or pii_spans:
                redaction_result = hippocampal.redactor.redact(
                    chunk.text,
                    additional_spans=pii_spans,
                )
            redacted_text = redaction_result.redacted_text if redaction_result else chunk.text

            if configured_uses_unified and unified_result is not None:
                entities = unified_result.entities or []
                relations = unified_result.relations or []
            else:
                if hippocampal.entity_extractor:
                    entities = await hippocampal.entity_extractor.extract(redacted_text)
                else:
                    entities = _ner_entities_for_text(redacted_text)
                if hippocampal.relation_extractor:
                    relations = await hippocampal.relation_extractor.extract(
                        redacted_text,
                        entities=[entity.normalized for entity in entities],
                    )
                else:
                    relations = _ner_relations_for_text(redacted_text)

            settings = get_settings().features
            chosen_memory_type = memory_type_override
            if (
                chosen_memory_type is None
                and configured_uses_unified
                and unified_result
                and settings.use_llm_enabled
                and settings.use_llm_memory_type
                and unified_result.memory_type
            ):
                try:
                    chosen_memory_type = MemoryType(unified_result.memory_type)
                except ValueError:
                    chosen_memory_type = None
            if chosen_memory_type is None and local_result and local_result.get("memory_type"):
                try:
                    chosen_memory_type = MemoryType(local_result["memory_type"])
                except ValueError:
                    chosen_memory_type = None
            if chosen_memory_type is None:
                chosen_memory_type = (
                    gate_result.memory_types[0]
                    if gate_result.memory_types
                    else MemoryType.EPISODIC_EVENT
                )

            if configured_uses_unified and unified_result and settings.use_llm_constraint_extractor:
                extracted_constraints = unified_result.constraints
            else:
                extracted_constraints = hippocampal.constraint_extractor.extract(chunk)

            if configured_uses_unified and unified_result and settings.use_llm_write_time_facts:
                extracted_facts = unified_result.facts
            else:
                extracted_facts = write_fact_extractor.extract(chunk)

            if (
                memory_type_override is None
                and chosen_memory_type != MemoryType.PREFERENCE
                and extracted_constraints
                and any(constraint.confidence >= 0.7 for constraint in extracted_constraints)
            ):
                chosen_memory_type = MemoryType.CONSTRAINT

            if chosen_memory_type == MemoryType.CONSTRAINT and extracted_constraints:
                key = ConstraintExtractor.constraint_fact_key(extracted_constraints[0])
            else:
                key = hippocampal._generate_key(chunk, chosen_memory_type) or ""

            importance = gate_result.importance
            if (
                configured_uses_unified
                and unified_result
                and settings.use_llm_enabled
                and settings.use_llm_write_gate_importance
            ):
                importance = unified_result.importance
            elif local_result and "importance" in local_result:
                importance = local_result["importance"]

            effective_context_tags = list(body.context_tags or [])
            if (
                not effective_context_tags
                and configured_uses_unified
                and unified_result
                and settings.use_llm_enabled
                and settings.use_llm_context_tags
                and getattr(unified_result, "context_tags", None)
            ):
                effective_context_tags = list(unified_result.context_tags)
            elif not effective_context_tags and local_result and local_result.get("context_tags"):
                effective_context_tags = list(local_result["context_tags"])

            confidence = chunk.confidence
            if (
                configured_uses_unified
                and unified_result
                and settings.use_llm_enabled
                and settings.use_llm_confidence
            ):
                confidence = unified_result.confidence
            elif local_result and "confidence" in local_result:
                confidence = local_result["confidence"]

            decay_rate = None
            if (
                configured_uses_unified
                and unified_result
                and settings.use_llm_enabled
                and settings.use_llm_decay_rate
                and getattr(unified_result, "decay_rate", None) is not None
            ):
                decay_rate = unified_result.decay_rate
            elif local_result and local_result.get("decay_rate") is not None:
                decay_rate = float(local_result["decay_rate"])

            would_store = gate_result.decision != WriteDecision.SKIP
            if would_store:
                store_count += 1
            else:
                skip_count += 1

            extractor_outputs: dict[str, Any] = {
                "configured_mode": "unified_llm" if configured_uses_unified else "local_modelpack_or_rules",
                "rule_constraints": [constraint.to_dict() for constraint in hippocampal.constraint_extractor.extract(chunk)],
                "rule_facts": [_serialize_fact(fact) for fact in write_fact_extractor.extract(chunk)],
            }
            if configured_uses_unified and unified_result is not None:
                extractor_outputs["unified_llm"] = {
                    "entities": [_safe_model_dump(entity) for entity in unified_result.entities],
                    "relations": [_safe_model_dump(relation) for relation in unified_result.relations],
                    "constraints": [constraint.to_dict() for constraint in unified_result.constraints],
                    "facts": [_serialize_fact(fact) for fact in unified_result.facts],
                    "importance": unified_result.importance,
                    "salience": unified_result.salience,
                    "memory_type": unified_result.memory_type,
                    "confidence": unified_result.confidence,
                    "context_tags": list(unified_result.context_tags),
                    "decay_rate": unified_result.decay_rate,
                    "contains_secrets": unified_result.contains_secrets,
                    "pii_spans": [
                        {"start": span.start, "end": span.end, "pii_type": span.pii_type}
                        for span in unified_result.pii_spans
                    ],
                }
            if local_result is not None:
                extractor_outputs["local_modelpack"] = {
                    "facts": [_serialize_fact(fact) for fact in local_result.get("facts", [])],
                    "importance": local_result.get("importance"),
                    "pii_spans": list(local_result.get("pii_spans", [])),
                    "memory_type": local_result.get("memory_type"),
                    "constraints": list(local_result.get("constraints", [])),
                    "context_tags": list(local_result.get("context_tags", [])),
                    "confidence": local_result.get("confidence"),
                    "decay_rate": local_result.get("decay_rate"),
                }

            chunk_results.append(
                DashboardWriteSimulationChunk(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    chunk_type=chunk.chunk_type.value,
                    salience=chunk.salience,
                    novelty=gate_result.novelty,
                    confidence=confidence,
                    timestamp=chunk.timestamp,
                    write_decision=gate_result.decision.value,
                    would_store=would_store,
                    reason=gate_result.reason,
                    chosen_memory_type=chosen_memory_type.value if chosen_memory_type else None,
                    key=key,
                    context_tags=effective_context_tags,
                    importance=importance,
                    decay_rate=decay_rate,
                    risk_flags=list(gate_result.risk_flags),
                    redaction_required=gate_result.redaction_required,
                    redacted_text=redacted_text,
                    redactions=[
                        _serialize_redaction(redaction)
                        for redaction in (redaction_result.redactions if redaction_result else [])
                    ],
                    entities=[_safe_model_dump(entity) for entity in entities],
                    relations=[_safe_model_dump(relation) for relation in relations],
                    extracted_constraints=[constraint.to_dict() for constraint in extracted_constraints],
                    extracted_facts=[_serialize_fact(fact) for fact in extracted_facts],
                    extractor_outputs=extractor_outputs,
                )
            )

        acceptance_rate = round((store_count / len(chunks)) * 100, 2) if chunks else 0.0
        return DashboardWriteSimulationResponse(
            tenant_id=body.tenant_id,
            chunks=chunk_results,
            summary={
                "chunk_count": len(chunks),
                "would_store_count": store_count,
                "skipped_count": skip_count,
                "write_gate_acceptance_rate": acceptance_rate,
                "configured_extractor": "unified_llm" if configured_uses_unified else "local_modelpack_or_rules",
                "compare_extractors": body.compare_extractors,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("dashboard_write_simulate_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/forgetting/preview", response_model=DashboardForgettingPreviewResponse)
async def dashboard_forgetting_preview(
    body: DashboardForgettingPreviewRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Preview forgetting scores, suggested actions, and duplicate candidates."""
    _ = auth
    try:
        worker = request.app.state.orchestrator.forgetting
        user_id = body.user_id or body.tenant_id
        memories = await worker.store.scan(
            body.tenant_id,
            filters={"status": MemoryStatus.ACTIVE.value},
            limit=body.max_memories,
        )
        dep_counts = await worker._get_dependency_counts(body.tenant_id, user_id, memories)
        scores = worker.scorer.score_batch(memories, dep_counts)
        duplicates = worker.interference.detect_duplicates(memories)
        planned = worker.policy.plan_operations(scores)

        memory_by_id = {str(memory.id): memory for memory in memories}
        duplicates_by_id: dict[str, list[dict[str, Any]]] = {}
        for duplicate in duplicates:
            duplicates_by_id.setdefault(duplicate.memory_id, []).append(
                {
                    "memory_id": duplicate.interfering_memory_id,
                    "similarity": duplicate.similarity,
                    "interference_type": duplicate.interference_type,
                    "recommendation": duplicate.recommendation,
                    "keep_id": duplicate.keep_id,
                }
            )
            duplicates_by_id.setdefault(duplicate.interfering_memory_id, []).append(
                {
                    "memory_id": duplicate.memory_id,
                    "similarity": duplicate.similarity,
                    "interference_type": duplicate.interference_type,
                    "recommendation": duplicate.recommendation,
                    "keep_id": duplicate.keep_id,
                }
            )

        items = []
        action_counts: dict[str, int] = {}
        protected_types: frozenset[str] = cast(
            "frozenset[str]",
            getattr(worker.scorer, "_PROTECTED_TYPES", frozenset()),
        )
        for score in sorted(scores, key=lambda item: item.total_score):
            record = memory_by_id.get(score.memory_id)
            if record is None:
                continue
            action_counts[score.suggested_action] = action_counts.get(score.suggested_action, 0) + 1
            record_type = _enum_value(record.type)
            protected = record_type in protected_types
            keep_reason = None
            if protected:
                keep_reason = "Protected long-lived memory type"
            items.append(
                DashboardForgettingPreviewItem(
                    memory_id=record.id,
                    text=record.text,
                    type=record_type,
                    status=_enum_value(record.status),
                    confidence=record.confidence,
                    importance=record.importance,
                    access_count=record.access_count,
                    dependency_count=dep_counts.get(str(record.id), 0),
                    total_score=score.total_score,
                    importance_score=score.importance_score,
                    recency_score=score.recency_score,
                    frequency_score=score.frequency_score,
                    confidence_score=score.confidence_score,
                    type_bonus_score=score.type_bonus_score,
                    dependency_score=score.dependency_score,
                    suggested_action=score.suggested_action,
                    protected=protected,
                    keep_reason=keep_reason,
                    duplicate_matches=duplicates_by_id.get(str(record.id), []),
                )
            )

        return DashboardForgettingPreviewResponse(
            tenant_id=body.tenant_id,
            user_id=user_id,
            scanned_count=len(memories),
            duplicates_found=len(duplicates),
            operations_planned=len(planned),
            items=items,
            summary=action_counts,
        )
    except Exception as exc:
        logger.error("dashboard_forgetting_preview_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/reconsolidation/sessions", response_model=DashboardReconsolidationSessionsResponse)
async def dashboard_reconsolidation_sessions(
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """List active labile sessions with retrieval context for the reconsolidation workbench."""
    _ = auth
    try:
        items = await _load_labile_sessions(db, tenant_id=tenant_id)
        return DashboardReconsolidationSessionsResponse(items=items, total=len(items))
    except Exception as exc:
        logger.error("dashboard_reconsolidation_sessions_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/quality/overview", response_model=DashboardQualityResponse)
async def dashboard_quality_overview(
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Aggregate data-quality and graph-hygiene issues for operators."""
    _ = auth
    try:
        issues: list[DashboardQualityIssue] = []
        invalidation_hotspots: list[dict[str, Any]] = []
        graph_hygiene: dict[str, Any] = {}

        async with db.pg_session() as session:
            mem_filters = []
            fact_filters = []
            if tenant_id:
                mem_filters.append(MemoryRecordModel.tenant_id == tenant_id)
                fact_filters.append(SemanticFactModel.tenant_id == tenant_id)

            low_conf_memories_q = (
                select(MemoryRecordModel)
                .where(
                    MemoryRecordModel.confidence < 0.4,
                    MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                    *mem_filters,
                )
                .limit(5)
            )
            low_conf_memories = (await session.execute(low_conf_memories_q)).scalars().all()
            low_conf_count_q = select(func.count()).select_from(MemoryRecordModel).where(
                MemoryRecordModel.confidence < 0.4,
                MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                *mem_filters,
            )
            low_conf_count = (await session.execute(low_conf_count_q)).scalar() or 0
            issues.append(
                DashboardQualityIssue(
                    label="low_confidence_memories",
                    count=int(low_conf_count),
                    description="Active memories with confidence below 0.40.",
                    sample_ids=_sample_ids(low_conf_memories),
                )
            )

            low_evidence_q = (
                select(SemanticFactModel)
                .where(
                    SemanticFactModel.is_current.is_(True),
                    SemanticFactModel.evidence_count <= 1,
                    *fact_filters,
                )
                .limit(5)
            )
            low_evidence = (await session.execute(low_evidence_q)).scalars().all()
            low_evidence_count_q = select(func.count()).select_from(SemanticFactModel).where(
                SemanticFactModel.is_current.is_(True),
                SemanticFactModel.evidence_count <= 1,
                *fact_filters,
            )
            low_evidence_count = (await session.execute(low_evidence_count_q)).scalar() or 0
            issues.append(
                DashboardQualityIssue(
                    label="low_evidence_facts",
                    count=int(low_evidence_count),
                    description="Current semantic facts backed by one or fewer evidence memories.",
                    sample_ids=_sample_ids(low_evidence),
                )
            )

            orphan_facts_q = select(func.count()).select_from(SemanticFactModel).where(
                func.cardinality(SemanticFactModel.evidence_ids) == 0,
                *fact_filters,
            )
            orphan_facts = (await session.execute(orphan_facts_q)).scalar() or 0
            issues.append(
                DashboardQualityIssue(
                    label="orphan_facts",
                    count=int(orphan_facts),
                    description="Semantic facts with no evidence ids attached.",
                )
            )

            duplicate_hash_rows = (
                await session.execute(
                    select(func.count())
                    .select_from(
                        select(MemoryRecordModel.content_hash)
                        .where(
                            MemoryRecordModel.content_hash.isnot(None),
                            *mem_filters,
                        )
                        .group_by(MemoryRecordModel.content_hash)
                        .having(func.count() > 1)
                        .subquery()
                    )
                )
            ).scalar() or 0
            issues.append(
                DashboardQualityIssue(
                    label="duplicate_content_hash_groups",
                    count=int(duplicate_hash_rows),
                    description="Distinct content hashes shared by more than one memory.",
                )
            )

            parent_memory = aliased(MemoryRecordModel)
            broken_memory_lineage_q = (
                select(func.count())
                .select_from(MemoryRecordModel)
                .outerjoin(parent_memory, MemoryRecordModel.supersedes_id == parent_memory.id)
                .where(
                    MemoryRecordModel.supersedes_id.isnot(None),
                    parent_memory.id.is_(None),
                    *mem_filters,
                )
            )
            broken_memory_lineage = (await session.execute(broken_memory_lineage_q)).scalar() or 0
            issues.append(
                DashboardQualityIssue(
                    label="broken_memory_lineage",
                    count=int(broken_memory_lineage),
                    description="Memories whose supersedes pointer no longer resolves.",
                )
            )

            parent_fact = aliased(SemanticFactModel)
            broken_fact_lineage_q = (
                select(func.count())
                .select_from(SemanticFactModel)
                .outerjoin(parent_fact, SemanticFactModel.supersedes_id == parent_fact.id)
                .where(
                    SemanticFactModel.supersedes_id.isnot(None),
                    parent_fact.id.is_(None),
                    *fact_filters,
                )
            )
            broken_fact_lineage = (await session.execute(broken_fact_lineage_q)).scalar() or 0
            issues.append(
                DashboardQualityIssue(
                    label="broken_fact_lineage",
                    count=int(broken_fact_lineage),
                    description="Facts whose supersedes pointer no longer resolves.",
                )
            )

            invalidation_rows = (
                await session.execute(
                    select(SemanticFactModel.key, func.count().label("count"))
                    .where(SemanticFactModel.is_current.is_(False), *fact_filters)
                    .group_by(SemanticFactModel.key)
                    .order_by(func.count().desc())
                    .limit(10)
                )
            ).all()
            invalidation_hotspots = [
                {"key": key, "count": count} for key, count in invalidation_rows if key
            ]

        labile_sessions = len(await _load_labile_sessions(db, tenant_id=tenant_id))

        if db.neo4j_driver:
            async with db.neo4j_session() as neo_session:
                params: dict[str, Any] = {}
                tenant_clause = ""
                if tenant_id:
                    params["tenant_id"] = tenant_id
                    tenant_clause = "WHERE n.tenant_id = $tenant_id"

                dup_result = await neo_session.run(
                    f"""
                    MATCH (n:Entity)
                    {tenant_clause}
                    WITH n.tenant_id AS tid, n.scope_id AS sid, n.entity AS entity, count(*) AS cnt
                    WHERE cnt > 1
                    RETURN count(*) AS duplicate_groups
                    """,
                    **params,
                )
                stale_result = await neo_session.run(
                    f"""
                    MATCH (n:Entity)
                    {tenant_clause}
                    WHERE NOT (n)--()
                    RETURN count(n) AS stale_nodes
                    """,
                    **params,
                )
                duplicate_groups = await dup_result.single()
                stale_nodes = await stale_result.single()
                graph_hygiene = {
                    "duplicate_entity_groups": int(
                        (duplicate_groups or {}).get("duplicate_groups", 0)
                    ),
                    "stale_nodes": int((stale_nodes or {}).get("stale_nodes", 0)),
                }

        return DashboardQualityResponse(
            tenant_id=tenant_id,
            generated_at=datetime.now(UTC),
            issues=issues,
            invalidation_hotspots=invalidation_hotspots,
            graph_hygiene=graph_hygiene,
            labile_sessions=labile_sessions,
        )
    except Exception as exc:
        logger.error("dashboard_quality_overview_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ops/metrics", response_model=DashboardOpsResponse)
async def dashboard_ops_metrics(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Expose selected Prometheus/runtime metrics for the dashboard."""
    _ = auth
    try:
        metrics: list[DashboardOpsMetric] = []
        highlights: dict[str, Any] = {
            "retrieval_timeouts_total": 0.0,
            "retrieval_step_failures_total": 0.0,
            "memory_reads_total": 0.0,
            "memory_writes_total": 0.0,
        }

        for family in REGISTRY.collect():
            if not (
                family.name.startswith("cml_")
                or family.name.startswith("memory_")
                or family.name.startswith("retrieval_")
            ):
                continue
            for sample in family.samples:
                if sample.name.endswith("_created"):
                    continue
                labels = {key: str(value) for key, value in sample.labels.items()}
                value = float(sample.value)
                metrics.append(DashboardOpsMetric(name=sample.name, labels=labels, value=value))
                if sample.name == "cml_retrieval_timeout_total":
                    highlights["retrieval_timeouts_total"] += value
                elif sample.name == "cml_retrieval_step_failures_total":
                    highlights["retrieval_step_failures_total"] += value
                elif sample.name == "memory_reads_total":
                    highlights["memory_reads_total"] += value
                elif sample.name == "memory_writes_total":
                    highlights["memory_writes_total"] += value
                elif sample.name == "cml_db_pool_checked_out":
                    highlights["db_pool_checked_out"] = value
                elif sample.name == "cml_retrieval_fact_hit_total":
                    bucket = "fact_hit_yes" if labels.get("hit") in {"1", "true", "yes"} else "fact_hit_no"
                    highlights[bucket] = highlights.get(bucket, 0.0) + value

        hit_yes = float(highlights.get("fact_hit_yes", 0.0))
        hit_no = float(highlights.get("fact_hit_no", 0.0))
        if hit_yes + hit_no > 0:
            highlights["fact_hit_rate"] = round(hit_yes / (hit_yes + hit_no), 4)

        if db.redis:
            try:
                redis_info = await db.redis.info("memory")
                highlights["redis_keys"] = await db.redis.dbsize()
                highlights["redis_used_memory_mb"] = round(
                    redis_info.get("used_memory", 0) / (1024 * 1024),
                    2,
                )
            except Exception as exc:
                logger.warning("dashboard_ops_redis_metrics_failed", error=str(exc))

        metrics.sort(key=lambda item: (item.name, json.dumps(item.labels, sort_keys=True)))
        return DashboardOpsResponse(
            generated_at=datetime.now(UTC),
            highlights=highlights,
            metrics=metrics,
        )
    except Exception as exc:
        logger.error("dashboard_ops_metrics_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/evaluation/summary", response_model=DashboardEvaluationSummary)
async def dashboard_evaluation_summary(
    auth: AuthContext = Depends(require_admin_permission),
):
    """Show evaluation readiness and discovered benchmark artifacts."""
    _ = auth
    try:
        root = _project_root()
        evaluation_dir = root / "evaluation"
        outputs_dir = evaluation_dir / "outputs"
        unified_samples = evaluation_dir / "locomo_plus" / "data" / "unified_input_samples_v2.json"
        summary_path = outputs_dir / "locomo_plus_qa_cml_judge_summary.json"
        comparison_path = evaluation_dir / "COMPARISON.md"

        artifact_paths = sorted(outputs_dir.glob("*")) if outputs_dir.exists() else []
        artifacts = [
            DashboardEvaluationArtifact(
                name=path.name,
                path=str(path),
                kind=path.suffix.lstrip(".") or "file",
                size_bytes=path.stat().st_size if path.exists() else 0,
                updated_at=(
                    datetime.fromtimestamp(path.stat().st_mtime, tz=UTC) if path.exists() else None
                ),
            )
            for path in artifact_paths
            if path.is_file()
        ]

        latest_summary = None
        if summary_path.is_file():
            latest_summary = json.loads(summary_path.read_text(encoding="utf-8"))

        latest_report = None
        if comparison_path.is_file():
            latest_report = comparison_path.read_text(encoding="utf-8")

        readiness = {
            "evaluation_dir_present": evaluation_dir.is_dir(),
            "outputs_dir_present": outputs_dir.is_dir(),
            "scripts_present": (evaluation_dir / "scripts" / "run_full_eval.py").is_file(),
            "locomo_data_present": (evaluation_dir / "locomo_plus" / "data" / "locomo10.json").is_file(),
            "unified_samples_present": unified_samples.is_file(),
            "latest_summary_present": summary_path.is_file(),
            "llm_eval_provider": get_settings().llm_eval.provider,
            "llm_eval_model": get_settings().llm_eval.model,
        }

        return DashboardEvaluationSummary(
            generated_at=datetime.now(UTC),
            readiness=readiness,
            latest_summary=latest_summary,
            latest_report=latest_report,
            artifacts=artifacts,
            comparisons=[],
        )
    except Exception as exc:
        logger.error("dashboard_evaluation_summary_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
