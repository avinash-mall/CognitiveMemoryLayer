"""Hybrid retriever executing plans across memory sources."""

import asyncio
import inspect
import re
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
from ..utils.modelpack import ModelPackRuntime, get_modelpack_runtime
from .planner import RetrievalPlan, RetrievalSource, RetrievalStep
from .query_types import QueryAnalysis

logger = get_logger(__name__)

_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "food": {"food", "eat", "meal", "diet", "restaurant", "allergy", "nutrition", "cuisine"},
    "travel": {"travel", "trip", "flight", "hotel", "vacation", "destination"},
    "finance": {"finance", "bank", "money", "budget", "loan", "invest", "expense"},
    "health": {"health", "medical", "doctor", "medicine", "sleep", "exercise", "allergy"},
    "work": {"work", "job", "career", "project", "deadline", "meeting", "office"},
    "tech": {"tech", "software", "code", "api", "database", "python", "app"},
    "social": {"friend", "family", "social", "party", "relationship", "people"},
}


@dataclass
class RetrievalResult:
    """Result from a single retrieval step."""

    source: RetrievalSource
    items: list[dict[str, Any]]
    elapsed_ms: float
    success: bool
    error: str | None = None


class HybridRetriever:
    """Executes retrieval plans across multiple memory sources.

    Enforces per-step and total retrieval timeouts so that slow sources
    (e.g. Neo4j GDS) don't dominate tail latency.
    """

    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        cache: Any | None = None,
        modelpack: ModelPackRuntime | None = None,
    ):
        self.hippocampal = hippocampal
        self.neocortical = neocortical
        self.cache = cache
        self.modelpack = modelpack if modelpack is not None else get_modelpack_runtime()

    async def retrieve(
        self,
        tenant_id: str,
        plan: RetrievalPlan,
        context_filter: list[str] | None = None,
        query_embedding: list[float] | None = None,
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
                if result.success and result.items:
                    all_results.extend(result.items)
                    # Phase 3.2: cross-group skip
                    if step.skip_if_found and result.items and cross_skip:
                        skip_remaining = True

        # AUD-07: Post-retrieval graph expansion from derived seeds and domain anchors
        if not any(s.source == RetrievalSource.GRAPH for s in plan.steps):
            derived_seeds = self._derive_graph_seeds(
                all_results, query_domain=plan.analysis.query_domain
            )
            if derived_seeds:
                graph_step = RetrievalStep(
                    source=RetrievalSource.GRAPH,
                    seeds=derived_seeds,
                    top_k=10,
                    priority=1,
                    timeout_ms=200,
                )
                try:
                    graph_results = await self._retrieve_graph(tenant_id, graph_step)
                    all_results.extend(graph_results)
                except Exception as e:
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

        elapsed_ms = (time.perf_counter() - plan_start) * 1000
        plan.analysis.metadata["retrieval_meta"] = {
            "sources_attempted": sorted({step.source.value for step in plan.steps}),
            "sources_completed": sorted(completed_sources),
            "sources_timed_out": sorted(timed_out_sources),
            "sources_failed": sorted(failed_sources),
            "total_elapsed_ms": round(elapsed_ms, 2),
        }
        return self._to_retrieved_memories(all_results, plan.analysis)

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
                items = await self._retrieve_graph(tenant_id, step)
            elif step.source == RetrievalSource.CONSTRAINTS:
                items = await self._retrieve_constraints(
                    tenant_id, step, context_filter, query_embedding
                )
            elif step.source == RetrievalSource.CACHE:
                items = await self._retrieve_cache(tenant_id, step)
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
            facts = await self.neocortical.text_search(tenant_id, step.query, limit=step.top_k)
            for f in facts:
                results.append(
                    {
                        "type": MemoryType.SEMANTIC_FACT.value,
                        "source": "facts",
                        "key": f.get("key"),
                        "text": f"{f.get('key', '')}: {f.get('value', '')}",
                        "value": f.get("value"),
                        "confidence": f.get("confidence", 0.5),
                        "relevance": 0.8,
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

    @staticmethod
    def _derive_graph_seeds(
        all_results: list[dict[str, Any]],
        max_seeds: int = 10,
        query_domain: str | None = None,
    ) -> list[str]:
        """Derive entity seeds from top constraint/fact retrieval results.

        Looks at entity mentions, subjects, and scope values from constraint
        and fact results so graph expansion can fire even when the original
        query had no recognised entities.  Also seeds from ``query_domain``
        when provided (e.g. "food", "finance") so domain-anchored graph
        traversal can bridge semantic disconnect.
        """
        seeds: list[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            normalised = name.strip().lower()
            if normalised and normalised not in seen and len(normalised) > 1:
                seen.add(normalised)
                seeds.append(name.strip())

        if query_domain:
            _add(query_domain)

        for item in all_results:
            if len(seeds) >= max_seeds:
                break
            rec = item.get("record")
            if rec is None:
                continue

            for ent in getattr(rec, "entities", None) or []:
                name = getattr(ent, "normalized", None) or getattr(ent, "text", None) or str(ent)
                _add(name)

            meta = getattr(rec, "metadata", None)
            if isinstance(meta, dict):
                for c in meta.get("constraints", []):
                    if isinstance(c, dict):
                        for field_name in ("subject", "scope"):
                            val = c.get(field_name)
                            if isinstance(val, str):
                                _add(val)
                subj = meta.get("subject")
                if isinstance(subj, str):
                    _add(subj)

        return seeds[:max_seeds]

    async def _retrieve_graph(
        self,
        tenant_id: str,
        step: RetrievalStep,
    ) -> list[dict[str, Any]]:
        """Retrieve via knowledge graph PPR. Holistic: tenant-only."""
        if not step.seeds:
            return []
        results = await self.neocortical.multi_hop_query(
            tenant_id, seed_entities=step.seeds, max_hops=3
        )
        return [
            {
                "type": "graph",
                "source": "graph",
                "entity": r.get("entity"),
                "text": self._format_entity_info(r),
                "relevance": r.get("relevance_score", 0.5),
                "record": r,
            }
            for r in results
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
            for k in ("since", "until", "source_session_id"):
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
                fact_category = getattr(fact, "category", None)
                cat_label = (
                    fact_category.value
                    if isinstance(fact_category, FactCategory)
                    else str(fact_category or "fact")
                )
                fact_text = f"[{cat_label.title()}] {fact.value}"
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

        rescored = self._rescore_constraints(
            query=step.query or "",
            query_domain=step.query_domain,
            rows=results,
        )
        return rescored[: step.top_k]

    def _rescore_constraints(
        self,
        *,
        query: str,
        query_domain: str | None,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not rows:
            return rows

        supports_task = getattr(self.modelpack, "supports_task", None)
        if callable(supports_task):
            can_constraint_rerank = bool(supports_task("constraint_rerank"))
            can_scope_match = bool(supports_task("scope_match"))
            use_relevance_model = bool(supports_task("retrieval_constraint_relevance_pair"))
        else:
            can_constraint_rerank = bool(getattr(self.modelpack, "available", False))
            can_scope_match = bool(getattr(self.modelpack, "available", False))
            use_relevance_model = bool(
                getattr(self.modelpack, "has_task_model", lambda _task: False)(
                    "retrieval_constraint_relevance_pair"
                )
            )

        modelpack_ready = bool(query.strip()) and (can_constraint_rerank or can_scope_match)

        for row in rows:
            base = float(row.get("relevance", 0.7))
            text = str(row.get("text", "") or "")

            relevance_bonus: float | None = None
            if use_relevance_model and text:
                try:
                    pred = self.modelpack.predict_score_pair(
                        "retrieval_constraint_relevance_pair", query, text
                    )
                    if pred is not None:
                        relevance_bonus = pred.score
                except Exception:
                    pass

            if relevance_bonus is not None:
                score = base + relevance_bonus
            else:
                score = base + self._domain_bonus(query_domain=query_domain, row=row)

            if modelpack_ready and text:
                rel_pred = self.modelpack.predict_pair("constraint_rerank", query, text)
                if rel_pred:
                    rel_signal = (
                        rel_pred.confidence
                        if rel_pred.label == "relevant"
                        else (1.0 - rel_pred.confidence)
                    )
                    score += (rel_signal - 0.5) * 0.45

                scope_pred = self.modelpack.predict_pair("scope_match", query, text)
                if scope_pred:
                    scope_signal = (
                        scope_pred.confidence
                        if scope_pred.label == "match"
                        else (1.0 - scope_pred.confidence)
                    )
                    score += (scope_signal - 0.5) * 0.25

            row["relevance"] = max(0.0, min(1.5, score))

        rows.sort(
            key=lambda x: (
                float(x.get("relevance", 0.0)),
                float(x.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        return rows

    def _domain_bonus(self, *, query_domain: str | None, row: dict[str, Any]) -> float:
        domain = (query_domain or "").strip().lower()
        if not domain or domain == "general":
            return 0.0

        bonus = 0.0
        text = str(row.get("text", "") or "").lower()

        # Prefer constraints already tagged with matching scope/context.
        row_tags: set[str] = set()
        record = row.get("record")
        record_tags = getattr(record, "context_tags", None)
        if isinstance(record_tags, list):
            row_tags.update(str(t).strip().lower() for t in record_tags if str(t).strip())

        metadata = getattr(record, "metadata", None) or {}
        constraints_meta = metadata.get("constraints", []) if isinstance(metadata, dict) else []
        if constraints_meta and isinstance(constraints_meta, list):
            scope_vals = constraints_meta[0].get("scope", [])
            if isinstance(scope_vals, list):
                row_tags.update(str(t).strip().lower() for t in scope_vals if str(t).strip())

        if domain in row_tags:
            bonus += 0.25

        keywords = _DOMAIN_KEYWORDS.get(domain, set())
        if keywords and text:
            words = set(re.findall(r"[a-z0-9_]+", text))
            overlap = len(words & keywords)
            if overlap > 0:
                bonus += min(0.18, 0.06 * overlap)

        return bonus

    async def _retrieve_cache(
        self,
        tenant_id: str,
        step: RetrievalStep,
    ) -> list[dict[str, Any]]:
        """Retrieve from hot cache. Holistic: tenant-only."""
        if not self.cache:
            return []
        cache_key = f"hot:{tenant_id}"
        try:
            cached = await self.cache.get(cache_key)
        except Exception as e:
            logger.debug(
                "cache_get_failed", extra={"cache_key": cache_key, "error": str(e)}, exc_info=True
            )
            return []
        if cached:
            import json

            try:
                return json.loads(cached) if isinstance(cached, str) else cached
            except (TypeError, json.JSONDecodeError) as e:
                get_logger(__name__).warning(
                    "retrieval_cache_decode_error",
                    cache_key=cache_key,
                    error=str(e),
                )
        return []

    def _to_retrieved_memories(
        self,
        items: list[dict[str, Any]],
        analysis: QueryAnalysis,
    ) -> list[RetrievedMemory]:
        """Convert raw items to RetrievedMemory objects."""
        seen_texts: set = set()
        retrieved: list[RetrievedMemory] = []
        for item in items:
            text = item.get("text", "")
            if text in seen_texts:
                continue
            seen_texts.add(text)
            record = item.get("record")
            if isinstance(record, MemoryRecord):
                mem_record = record
            elif hasattr(record, "key") and hasattr(record, "value"):
                mem_record = self._fact_to_record(record, item)
            else:
                mem_record = self._dict_to_record(item)
            retrieved.append(
                RetrievedMemory(
                    record=mem_record,
                    relevance_score=item.get("relevance", 0.5),
                    retrieval_source=item.get("source", "unknown"),
                )
            )
        retrieved.sort(key=lambda x: x.relevance_score, reverse=True)
        return retrieved

    def _format_entity_info(self, entity_data: dict[str, Any]) -> str:
        """Format entity data as readable text."""
        lines = [f"Entity: {entity_data.get('entity', 'Unknown')}"]
        for rel in entity_data.get("relations", [])[:5]:
            lines.append(f"  - {rel.get('predicate', '')}: {rel.get('related_entity', '')}")
        return "\n".join(lines)

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
            timestamp=getattr(fact, "updated_at", datetime.now(UTC)),
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
