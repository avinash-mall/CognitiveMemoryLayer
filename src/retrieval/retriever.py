"""Hybrid retriever executing plans across memory sources."""

import asyncio
import inspect
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..core.enums import MemorySource, MemoryType
from ..core.schemas import MemoryRecord, Provenance, RetrievedMemory
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..utils.logging_config import get_logger
from .graph_expansion import (
    _CONTIGUITY_RELEVANCE_FACTOR,
    GRAPH_RELEVANCE_CEILING,
    GRAPH_RELEVANCE_FLOOR,
    GraphExpansionMixin,
    _neighbours_of,
)
from .planner import RetrievalPlan, RetrievalSource, RetrievalStep
from .query_types import QueryAnalysis

logger = get_logger(__name__)

__all__ = [
    "GRAPH_RELEVANCE_CEILING",
    "GRAPH_RELEVANCE_FLOOR",
    "_CONTIGUITY_RELEVANCE_FACTOR",
    "HybridRetriever",
    "RetrievalResult",
    "_neighbours_of",
]


@dataclass
class RetrievalResult:
    """Result from a single retrieval step."""

    source: RetrievalSource
    items: list[dict[str, Any]]
    elapsed_ms: float
    success: bool
    error: str | None = None


class HybridRetriever(GraphExpansionMixin):
    """Executes retrieval plans across multiple memory sources.

    Enforces per-step and total retrieval timeouts so that slow sources
    (e.g. Neo4j GDS) don't dominate tail latency.
    """

    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
    ):
        self.hippocampal = hippocampal
        self.neocortical = neocortical

    async def retrieve(
        self,
        tenant_id: str,
        plan: RetrievalPlan,
        context_filter: list[str] | None = None,
        query_embedding: list[float] | None = None,
        debug: bool = False,
    ) -> list[RetrievedMemory]:
        """Execute a retrieval plan with enforced timeouts. Holistic: tenant-only."""
        from ..core.config import get_settings

        settings = get_settings()
        timeouts_enabled = settings.features.retrieval_timeouts_enabled
        cross_skip = settings.features.skip_if_found_cross_group

        all_results: list[dict[str, Any]] = []
        plan_start = time.perf_counter()
        plan_budget = plan.total_timeout_ms / 1000.0
        skip_remaining = False
        completed_sources: set[str] = set()
        timed_out_sources: set[str] = set()
        failed_sources: set[str] = set()
        debug_steps: list[dict[str, Any]] = []

        for group_indices in plan.parallel_steps:
            if skip_remaining:
                break

            # Check remaining plan budget
            if timeouts_enabled:
                elapsed = time.perf_counter() - plan_start
                remaining = plan_budget - elapsed
                if remaining <= 0:
                    for idx in group_indices:
                        if idx < len(plan.steps):
                            timed_out_sources.add(plan.steps[idx].source.value)
                            if debug:
                                debug_steps.append(
                                    self._step_debug(
                                        plan.steps[idx],
                                        success=False,
                                        elapsed_ms=0.0,
                                        result_count=0,
                                        error="Plan budget exceeded before execution",
                                        timed_out=True,
                                        context_filter=context_filter,
                                    )
                                )
                    logger.info(
                        "retrieval_plan_budget_exceeded", extra={"elapsed_ms": elapsed * 1000}
                    )
                    break
            else:
                remaining = None  # No cap

            group_steps = [plan.steps[i] for i in group_indices if i < len(plan.steps)]

            # Execute group in parallel with per-step timeouts
            if timeouts_enabled:
                coros = [
                    self._execute_step_with_timeout(
                        tenant_id, step, context_filter, query_embedding
                    )
                    for step in group_steps
                ]
                try:
                    group_results = await asyncio.wait_for(
                        asyncio.gather(*coros, return_exceptions=True),
                        timeout=remaining,
                    )
                except TimeoutError:
                    for step in group_steps:
                        timed_out_sources.add(step.source.value)
                        if debug:
                            debug_steps.append(
                                self._step_debug(
                                    step,
                                    success=False,
                                    elapsed_ms=float(step.timeout_ms),
                                    result_count=0,
                                    error="Retrieval group timed out",
                                    timed_out=True,
                                    context_filter=context_filter,
                                )
                            )
                    logger.info("retrieval_group_timeout", extra={"group": group_indices})
                    break
            else:
                group_results = await asyncio.gather(
                    *[
                        self._execute_step(tenant_id, step, context_filter, query_embedding)
                        for step in group_steps
                    ],
                    return_exceptions=True,
                )

            for step, result in zip(group_steps, group_results, strict=False):
                if isinstance(result, BaseException):
                    failed_sources.add(step.source.value)
                    continue
                if result.success:
                    completed_sources.add(step.source.value)
                elif result.error and "timeout" in result.error.lower():
                    timed_out_sources.add(step.source.value)
                else:
                    failed_sources.add(step.source.value)
                if debug:
                    debug_steps.append(
                        self._step_debug(
                            step,
                            success=result.success,
                            elapsed_ms=result.elapsed_ms,
                            result_count=len(result.items),
                            error=result.error,
                            timed_out=bool(result.error and "timeout" in result.error.lower()),
                            context_filter=context_filter,
                            result_items=result.items,
                        )
                    )
                if result.success and result.items:
                    all_results.extend(result.items)
                    # Phase 3.2: cross-group skip
                    if step.skip_if_found and result.items and cross_skip:
                        skip_remaining = True

        # AUD-07: Post-retrieval graph expansion from derived seeds and domain anchors
        if not any(s.source == RetrievalSource.GRAPH for s in plan.steps):
            derived_seeds = self._derive_graph_seeds(all_results)
            if derived_seeds:
                graph_step = RetrievalStep(
                    source=RetrievalSource.GRAPH,
                    seeds=derived_seeds,
                    top_k=10,
                    priority=1,
                    timeout_ms=200,
                )
                try:
                    graph_results = await self._retrieve_graph(
                        tenant_id, graph_step, query_embedding
                    )
                    all_results.extend(graph_results)
                    if debug:
                        debug_steps.append(
                            self._step_debug(
                                graph_step,
                                success=True,
                                elapsed_ms=0.0,
                                result_count=len(graph_results),
                                error=None,
                                timed_out=False,
                                context_filter=context_filter,
                                result_items=graph_results,
                            )
                        )
                except Exception as e:
                    if debug:
                        debug_steps.append(
                            self._step_debug(
                                graph_step,
                                success=False,
                                elapsed_ms=0.0,
                                result_count=0,
                                error=str(e),
                                timed_out=False,
                                context_filter=context_filter,
                            )
                        )
                    logger.warning(
                        "derived_graph_expansion_failed",
                        extra={"error": str(e), "seeds": derived_seeds[:5]},
                    )

        # PR 1: Strict Associative Expansion
        if any(getattr(step, "associative_expansion", False) for step in plan.steps):
            session_ids = set()
            for r in all_results:
                rec = r.get("record")
                if (
                    rec
                    and hasattr(rec, "source_session_id")
                    and getattr(rec, "source_session_id", None)
                ):
                    session_ids.add(rec.source_session_id)
            if session_ids:
                try:
                    batch_limit = min(200, max(20, len(session_ids) * 10))
                    try:
                        extra_constraints = await self.hippocampal.store.scan(
                            tenant_id,
                            filters={
                                "type": [MemoryType.CONSTRAINT.value],
                                "source_session_id": list(session_ids),
                                "status": "active",
                            },
                            limit=batch_limit,
                        )
                    except Exception:
                        # Fallback for stores that don't support list-valued source_session_id filters.
                        extra_constraints = []
                        for sid in session_ids:
                            rows = await self.hippocampal.store.scan(
                                tenant_id,
                                filters={
                                    "type": [MemoryType.CONSTRAINT.value],
                                    "source_session_id": sid,
                                    "status": "active",
                                },
                                limit=10,
                            )
                            extra_constraints.extend(rows)

                    existing_ids = {
                        str(getattr(x.get("record"), "id", ""))
                        for x in all_results
                        if x.get("record") is not None
                    }
                    for c in extra_constraints:
                        cid = str(getattr(c, "id", ""))
                        if cid and cid in existing_ids:
                            continue
                        if cid:
                            existing_ids.add(cid)
                        all_results.append(
                            {
                                "type": MemoryType.CONSTRAINT.value,
                                "source": "constraints_associative",
                                "text": getattr(c, "text", ""),
                                "confidence": getattr(c, "confidence", 0.7),
                                "relevance": 0.8,
                                "timestamp": getattr(c, "timestamp", None),
                                "record": c,
                            }
                        )
                except Exception as e:
                    logger.warning(
                        "associative_expansion_failed", extra={"error": str(e)}, exc_info=True
                    )

        await self._expand_temporal_contiguity(tenant_id, all_results)
        await self._resolve_prospective_parents(tenant_id, all_results, query_embedding)
        await self._expand_second_hop_recollection(
            tenant_id, plan, all_results, query_embedding, plan_start, plan_budget, timeouts_enabled
        )

        elapsed_ms = (time.perf_counter() - plan_start) * 1000
        plan.analysis.metadata["retrieval_meta"] = {
            "sources_attempted": sorted({step.source.value for step in plan.steps}),
            "sources_completed": sorted(completed_sources),
            "sources_timed_out": sorted(timed_out_sources),
            "sources_failed": sorted(failed_sources),
            "total_elapsed_ms": round(elapsed_ms, 2),
        }
        if debug:
            plan.analysis.metadata["retrieval_steps"] = debug_steps
        return self._to_retrieved_memories(all_results, plan.analysis)

    def _step_debug(
        self,
        step: RetrievalStep,
        *,
        success: bool,
        elapsed_ms: float,
        result_count: int,
        error: str | None,
        timed_out: bool,
        context_filter: list[str] | None,
        result_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a serializable execution snapshot for explain mode."""
        time_filter = dict(step.time_filter or {})
        filters: dict[str, Any] = {}
        if context_filter:
            filters["context_tags"] = list(context_filter)
        if step.memory_types:
            filters["memory_types"] = list(step.memory_types)
        if time_filter:
            filters["time_filter"] = time_filter
        if step.constraint_categories:
            filters["constraint_categories"] = list(step.constraint_categories)
        if step.seeds:
            filters["seeds"] = list(step.seeds)
        preview = []
        for item in (result_items or [])[:5]:
            rec = item.get("record")
            preview.append(
                {
                    "id": str(getattr(rec, "id", "")) if rec is not None else "",
                    "type": str(getattr(rec, "type", item.get("type", ""))),
                    "text": str(getattr(rec, "text", item.get("text", "")))[:180],
                    "relevance": float(item.get("relevance", 0.0) or 0.0),
                    "confidence": float(
                        getattr(rec, "confidence", item.get("confidence", 0.0)) or 0.0
                    ),
                }
            )
        return {
            "source": step.source.value,
            "success": success,
            "elapsed_ms": round(elapsed_ms, 2),
            "result_count": result_count,
            "timed_out": timed_out,
            "error": error,
            "filters": filters,
            "query_preview": step.query,
            "result_preview": preview,
        }

    async def _execute_step_with_timeout(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
        query_embedding: list[float] | None = None,
    ) -> RetrievalResult:
        """Execute a step wrapped in its own timeout."""
        timeout_s = step.timeout_ms / 1000.0
        try:
            return await asyncio.wait_for(
                self._execute_step(tenant_id, step, context_filter, query_embedding),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.info(
                "retrieval_step_timeout",
                extra={"source": step.source.value, "timeout_ms": step.timeout_ms},
            )
            try:
                from ..utils.metrics import RETRIEVAL_TIMEOUT_COUNT

                RETRIEVAL_TIMEOUT_COUNT.labels(source=step.source.value).inc()
            except Exception:
                pass
            return RetrievalResult(
                source=step.source,
                items=[],
                elapsed_ms=float(step.timeout_ms),
                success=False,
                error=f"Timeout after {step.timeout_ms}ms",
            )

    async def _execute_step(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
        query_embedding: list[float] | None = None,
    ) -> RetrievalResult:
        """Execute a single retrieval step. Holistic: tenant-only."""
        start = time.perf_counter()
        try:
            if step.source == RetrievalSource.FACTS:
                items = await self._retrieve_facts(tenant_id, step)
            elif step.source == RetrievalSource.VECTOR:
                items = await self._retrieve_vector(
                    tenant_id, step, context_filter, query_embedding
                )
            elif step.source == RetrievalSource.GRAPH:
                items = await self._retrieve_graph(tenant_id, step, query_embedding)
            elif step.source == RetrievalSource.CONSTRAINTS:
                items = await self._retrieve_constraints(
                    tenant_id, step, context_filter, query_embedding
                )
            else:
                items = []
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Phase 6.2: emit step metrics
            self._record_step_metrics(step.source, elapsed_ms, len(items))

            return RetrievalResult(
                source=step.source,
                items=items,
                elapsed_ms=elapsed_ms,
                success=True,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "retrieval_step_failed",
                extra={"source": step.source.value, "error": str(e)},
                exc_info=True,
            )
            try:
                from ..utils.metrics import RETRIEVAL_STEP_FAILURES

                RETRIEVAL_STEP_FAILURES.labels(source=step.source.value).inc()
            except Exception:
                pass  # Metrics optional
            return RetrievalResult(
                source=step.source,
                items=[],
                elapsed_ms=elapsed_ms,
                success=False,
                error=str(e),
            )

    @staticmethod
    def _record_step_metrics(source: RetrievalSource, elapsed_ms: float, count: int) -> None:
        """Emit Prometheus metrics for a retrieval step (best-effort)."""
        try:
            from ..utils.metrics import (
                RETRIEVAL_STEP_DURATION,
                RETRIEVAL_STEP_RESULT_COUNT,
            )

            RETRIEVAL_STEP_DURATION.labels(source=source.value).observe(elapsed_ms)
            RETRIEVAL_STEP_RESULT_COUNT.labels(source=source.value).observe(count)
        except Exception as e:
            logger.debug("metrics_emit_failed", extra={"error": str(e)}, exc_info=True)

    async def _retrieve_facts(
        self,
        tenant_id: str,
        step: RetrievalStep,
    ) -> list[dict[str, Any]]:
        """Retrieve from semantic fact store. Holistic: tenant-only."""
        results: list[dict[str, Any]] = []
        if step.key:
            fact = await self.neocortical.get_fact(tenant_id, step.key)
            if fact:
                results.append(
                    {
                        "type": MemoryType.SEMANTIC_FACT.value,
                        "source": "facts",
                        "key": fact.key,
                        "text": f"{fact.predicate}: {fact.value}",
                        "value": fact.value,
                        "confidence": fact.confidence,
                        "relevance": 1.0,
                        "record": fact,
                    }
                )
        if step.query:
            import re as _re

            facts = await self.neocortical.text_search(tenant_id, step.query, limit=step.top_k)
            q_tokens = [
                t
                for t in _re.findall(r"[a-z0-9]{3,}", (step.query or "").lower())
                if t
                not in {
                    "the",
                    "and",
                    "for",
                    "that",
                    "this",
                    "with",
                    "what",
                    "when",
                    "where",
                    "who",
                    "why",
                    "how",
                }
            ]
            for f in facts:
                f_text = f"{f.get('key', '')}: {f.get('value', '')}"
                f_conf = float(f.get("confidence", 0.5) or 0.5)
                if q_tokens:
                    blob = f_text.lower()
                    hits = sum(1 for t in q_tokens if t in blob)
                    cov = hits / len(q_tokens)
                    rel = round(min(0.92, max(0.50, 0.50 + 0.32 * cov + 0.10 * f_conf)), 4)
                else:
                    rel = 0.8
                results.append(
                    {
                        "type": MemoryType.SEMANTIC_FACT.value,
                        "source": "facts",
                        "key": f.get("key"),
                        "text": f_text,
                        "value": f.get("value"),
                        "confidence": f_conf,
                        "relevance": rel,
                        "context_tags": f.get("context_tags", []),
                        "timestamp": f.get("valid_from") or f.get("updated_at"),
                        "record": f,
                    }
                )
        return results[: step.top_k]

    async def _retrieve_vector(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve via vector similarity search. Holistic: tenant-only."""
        filters = None
        if step.time_filter or step.memory_types or step.min_confidence > 0:
            filters = {}
            if step.time_filter:
                filters.update(step.time_filter)
            if step.memory_types:
                filters["type"] = step.memory_types
            if step.min_confidence > 0:
                filters["min_confidence"] = step.min_confidence
        records = await self.hippocampal.search(
            tenant_id,
            query=step.query or "",
            top_k=step.top_k,
            context_filter=context_filter,
            filters=filters,
            query_embedding=query_embedding,
        )
        return [
            {
                "type": "episode",
                "source": "vector",
                "text": r.text,
                "confidence": r.confidence,
                "relevance": r.metadata.get("_similarity", 0.5),
                "timestamp": r.timestamp,
                "record": r,
            }
            for r in records
        ]

    async def _retrieve_constraints(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve active constraints via vector search + semantic fact lookup.

        Two-pronged approach:
        1. Vector search filtered to MemoryType.CONSTRAINT (episodic constraint records).
        2. Semantic fact lookup for cognitive categories (goal, value, state, causal, policy).
        """
        results: list[dict[str, Any]] = []

        # 1. Vector search for episodic constraint records
        constraint_filters: dict[str, Any] = {
            "type": [MemoryType.CONSTRAINT.value],
            "status": "active",
            "exclude_expired": True,
        }
        if step.time_filter:
            for k in ("since", "until", "source_session_id", "time_basis"):
                if k in step.time_filter:
                    constraint_filters[k] = step.time_filter[k]
        try:
            records = await self.hippocampal.search(
                tenant_id,
                query=step.query or "",
                top_k=step.top_k,
                context_filter=context_filter,
                filters=constraint_filters,
                query_embedding=query_embedding,
            )
            for r in records:
                results.append(
                    {
                        "type": MemoryType.CONSTRAINT.value,
                        "source": "constraints",
                        "text": r.text,
                        "confidence": r.confidence,
                        "relevance": r.metadata.get("_similarity", 0.7),
                        "timestamp": r.timestamp,
                        "record": r,
                    }
                )
        except Exception:
            logger.warning("constraint_vector_search_failed", exc_info=True)

        # 2. Semantic fact lookup for cognitive constraint categories
        from ..memory.neocortical.schemas import FactCategory

        cognitive_categories: list[FactCategory] = []
        if step.constraint_categories:
            for cat_str in step.constraint_categories:
                try:
                    cognitive_categories.append(FactCategory(cat_str.lower()))
                except ValueError:
                    pass

        try:
            facts = []
            if not cognitive_categories:
                pass
            else:
                used_batch = False
                batch_fetch = getattr(self.neocortical.facts, "get_facts_by_categories", None)
                if callable(batch_fetch):
                    maybe_coro = batch_fetch(
                        tenant_id,
                        cognitive_categories,
                        current_only=True,
                        limit=max(step.top_k * max(1, len(cognitive_categories)), step.top_k),
                    )
                    if inspect.isawaitable(maybe_coro):
                        facts = await maybe_coro
                        used_batch = True

                if not used_batch:
                    for category in cognitive_categories:
                        facts.extend(
                            await self.neocortical.facts.get_facts_by_category(
                                tenant_id, category, current_only=True, limit=step.top_k
                            )
                        )

            existing_texts = {str(r.get("text", "")) for r in results}
            for fact in facts:
                # Skip garbage facts: value must be a meaningful phrase (>3 words)
                fact_val = str(fact.value).strip() if fact.value else ""
                if len(fact_val.split()) < 4:
                    continue
                fact_category = getattr(fact, "category", None)
                cat_label = (
                    fact_category.value
                    if isinstance(fact_category, FactCategory)
                    else str(fact_category or "fact")
                )
                fact_text = f"[{cat_label.title()}] {fact_val}"
                if fact_text in existing_texts:
                    continue
                existing_texts.add(fact_text)
                results.append(
                    {
                        "type": MemoryType.CONSTRAINT.value,
                        "source": "constraints",
                        "key": fact.key,
                        "text": fact_text,
                        "value": fact.value,
                        "confidence": fact.confidence,
                        "relevance": 0.75,
                        "record": fact,
                    }
                )
        except Exception:
            logger.warning("constraint_fact_search_failed", exc_info=True)

        rescored = self._rescore_constraints(rows=results)
        return rescored[: step.top_k]

    def _rescore_constraints(self, *, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows

        rows.sort(
            key=lambda x: (
                float(x.get("relevance", 0.0)),
                float(x.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        return rows

    def _to_retrieved_memories(
        self,
        items: list[dict[str, Any]],
        analysis: QueryAnalysis,
    ) -> list[RetrievedMemory]:
        """Convert raw items to RetrievedMemory objects with Synaptic Co-Activation tracking."""
        by_text: dict[str, RetrievedMemory] = {}
        sources_by_text: dict[str, set[str]] = {}
        for item in items:
            text = item.get("text", "")
            src = str(item.get("source", "unknown") or "unknown")
            rel = float(item.get("relevance", 0.5) or 0.5)
            if text in by_text:
                existing = by_text[text]
                sources_by_text[text].add(src)
                if rel > existing.relevance_score:
                    existing.relevance_score = rel
                if isinstance(existing.record.metadata, dict):
                    existing.record.metadata["_converging_sources"] = sorted(sources_by_text[text])
                    existing.record.metadata["_coactivation_count"] = len(sources_by_text[text])
                continue

            record = item.get("record")
            if isinstance(record, MemoryRecord):
                mem_record = record
            elif hasattr(record, "key") and hasattr(record, "value"):
                mem_record = self._fact_to_record(record, item)
            else:
                mem_record = self._dict_to_record(item)
            sources_by_text[text] = {src}
            by_text[text] = RetrievedMemory(
                record=mem_record,
                relevance_score=rel,
                retrieval_source=src,
            )
        retrieved = list(by_text.values())
        retrieved.sort(key=lambda x: x.relevance_score, reverse=True)
        return retrieved

    def _fact_to_record(self, fact: Any, item: dict[str, Any]) -> MemoryRecord:
        """Build MemoryRecord from SemanticFact-like object. Holistic: context_tags."""
        from uuid import UUID

        text = (
            item.get("text", "")
            or f"{getattr(fact, 'predicate', '')}: {getattr(fact, 'value', '')}"
        )
        fid = getattr(fact, "id", None)
        context_tags = getattr(fact, "context_tags", None) or []
        mem_type = MemoryType.SEMANTIC_FACT
        if item.get("type") == MemoryType.CONSTRAINT.value or item.get("source") == "constraints":
            mem_type = MemoryType.CONSTRAINT

        category = getattr(fact, "category", None)
        cat_value = (
            category.value
            if (category is not None and hasattr(category, "value"))
            else str(category or "")
        )
        evidence_ids = getattr(fact, "evidence_ids", None) or []
        valid_from = getattr(fact, "valid_from", None)
        valid_to = getattr(fact, "valid_to", None)
        supersedes_id = getattr(fact, "supersedes_id", None)

        metadata: dict[str, Any] = {}
        if mem_type == MemoryType.CONSTRAINT and cat_value:
            metadata["constraints"] = [
                {
                    "constraint_type": cat_value,
                    "evidence_ids": list(evidence_ids),
                }
            ]
        if valid_from:
            metadata["valid_from"] = str(valid_from)
        if valid_to:
            metadata["valid_to"] = str(valid_to)
        if supersedes_id:
            metadata["supersedes_id"] = str(supersedes_id)

        record_timestamp = valid_from or getattr(fact, "updated_at", None) or datetime.now(UTC)

        return MemoryRecord(
            id=UUID(fid) if fid else uuid4(),
            tenant_id=getattr(fact, "tenant_id", ""),
            context_tags=list(context_tags),
            source_session_id=None,
            type=mem_type,
            text=text,
            key=getattr(fact, "key", None),
            confidence=getattr(fact, "confidence", 0.5),
            importance=0.5,
            provenance=Provenance(
                source=MemorySource.AGENT_INFERRED,
                evidence_refs=list(evidence_ids),
            ),
            timestamp=record_timestamp,
            written_at=getattr(fact, "created_at", datetime.now(UTC)),
            metadata=metadata,
        )

    def _dict_to_record(self, item: dict[str, Any]) -> MemoryRecord:
        """Create minimal MemoryRecord from dict. Holistic: context_tags."""
        mem_type = item.get("type", "episodic_event")
        try:
            mtype = MemoryType(mem_type)
        except ValueError:
            mtype = MemoryType.EPISODIC_EVENT
        return MemoryRecord(
            id=uuid4(),
            tenant_id=item.get("tenant_id", ""),
            context_tags=item.get("context_tags", []),
            source_session_id=item.get("source_session_id"),
            type=mtype,
            text=item.get("text", ""),
            key=item.get("key"),
            confidence=item.get("confidence", 0.5),
            importance=0.5,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=item.get("timestamp", datetime.now(UTC)),
            written_at=datetime.now(UTC),
        )
