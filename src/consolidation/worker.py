"""Consolidation worker orchestrating the full flow."""

import asyncio
import contextlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from ..memory.neocortical.store import NeocorticalStore
from ..storage.base import MemoryStoreBase
from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger
from ..utils.modelpack import get_modelpack_runtime
from .clusterer import EpisodeCluster, SemanticClusterer
from .migrator import ConsolidationMigrator, MigrationResult
from .sampler import EpisodeSampler
from .schema_aligner import SchemaAligner
from .summarizer import ExtractedGist, GistExtractor
from .triggers import ConsolidationScheduler, ConsolidationTask

logger = get_logger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GENERIC_GIST_PATTERNS = (
    "user said",
    "mixed topics",
    "general conversation",
    "various topics",
)


@dataclass
class ConsolidationReport:
    """Report from a consolidation run."""

    tenant_id: str
    user_id: str
    started_at: datetime
    completed_at: datetime

    episodes_sampled: int
    clusters_formed: int
    gists_extracted: int
    migration: MigrationResult

    elapsed_seconds: float

    @property
    def success(self) -> bool:
        return len(self.migration.errors) == 0


class ConsolidationWorker:
    """Main consolidation worker that orchestrates the full process."""

    def __init__(
        self,
        episodic_store: MemoryStoreBase,
        neocortical_store: NeocorticalStore,
        llm_client: LLMClient | None,
        summarizer_backend=None,
        scheduler: ConsolidationScheduler | None = None,
    ):
        self.sampler = EpisodeSampler(episodic_store)
        self.clusterer = SemanticClusterer()
        self.extractor = GistExtractor(llm_client, fallback_summarizer=summarizer_backend)
        self.aligner = SchemaAligner(neocortical_store.facts)
        self.migrator = ConsolidationMigrator(neocortical_store, episodic_store)

        self.scheduler = scheduler or ConsolidationScheduler()
        self.modelpack = get_modelpack_runtime()

        self._running = False
        self._worker_task: asyncio.Task | None = None

    async def consolidate(
        self,
        tenant_id: str,
        user_id: str,
        task: ConsolidationTask | None = None,
    ) -> ConsolidationReport:
        """Run full consolidation for a user."""
        started = datetime.now(UTC)

        episode_limit = task.episode_limit if task else 200
        episodes = await self.sampler.sample(tenant_id, user_id, max_episodes=episode_limit)

        if not episodes:
            return ConsolidationReport(
                tenant_id=tenant_id,
                user_id=user_id,
                started_at=started,
                completed_at=datetime.now(UTC),
                episodes_sampled=0,
                clusters_formed=0,
                gists_extracted=0,
                migration=MigrationResult(0, 0, 0, 0, []),
                elapsed_seconds=0.0,
            )

        # BUG-06: Preserve constraints from episodic memories before gist extraction
        from ..core.enums import MemoryType
        from ..extraction.constraint_extractor import ConstraintExtractor, ConstraintObject
        from ..memory.neocortical.schemas import FactCategory

        cat_map = {
            "goal": FactCategory.GOAL,
            "value": FactCategory.VALUE,
            "state": FactCategory.STATE,
            "causal": FactCategory.CAUSAL,
            "policy": FactCategory.POLICY,
        }
        category_cache: dict[FactCategory, list] = {}
        for ep in episodes:
            if ep.type != MemoryType.CONSTRAINT:
                continue
            meta = ep.metadata or {}
            constraints_meta = meta.get("constraints", [])
            if not isinstance(constraints_meta, list):
                continue
            for cdict in constraints_meta:
                if not isinstance(cdict, dict):
                    continue
                try:
                    c = ConstraintObject(
                        constraint_type=cdict.get("constraint_type", "value"),
                        subject=cdict.get("subject", "user"),
                        description=cdict.get("description", ep.text),
                        scope=cdict.get("scope", []),
                        activation=cdict.get("activation", ""),
                        status=cdict.get("status", "active"),
                        confidence=float(cdict.get("confidence", 0.7)),
                        provenance=cdict.get("provenance", []),
                    )
                    fact_key = ConstraintExtractor.constraint_fact_key(c)

                    cat = cat_map.get((c.constraint_type or "").lower())
                    lineage_refs: list[str] = []
                    if cat is not None:
                        if cat not in category_cache:
                            category_cache[
                                cat
                            ] = await self.migrator.semantic.facts.get_facts_by_category(
                                tenant_id, cat, current_only=True, limit=200
                            )
                        for old in list(category_cache[cat]):
                            if old.key == fact_key:
                                continue
                            old_obj = ConstraintObject(
                                constraint_type=cat.value,
                                subject="user",
                                description=str(old.value),
                                scope=getattr(old, "context_tags", None) or [],
                            )
                            if await ConstraintExtractor.detect_supersession(old_obj, c):
                                await self.migrator.semantic.facts.invalidate_fact(
                                    tenant_id, old.key, reason="superseded_consolidation"
                                )
                                if hasattr(self.migrator.episodic, "deactivate_constraints_by_key"):
                                    await self.migrator.episodic.deactivate_constraints_by_key(
                                        tenant_id,
                                        old.key,
                                        superseded_by_key=fact_key,
                                    )
                                lineage_refs.extend(
                                    [
                                        f"semantic_key:{old.key}",
                                        f"episodic_constraint_key:{old.key}",
                                    ]
                                )

                    evidence = [str(ep.id), *lineage_refs]
                    await self.migrator.semantic.store_fact(
                        tenant_id=tenant_id,
                        key=fact_key,
                        value=c.description,
                        confidence=c.confidence,
                        evidence_ids=list(dict.fromkeys(evidence)),
                        context_tags=c.scope,
                    )
                except Exception as e:
                    logger.warning(
                        "consolidation_constraint_from_episode_failed",
                        extra={"episode_id": str(ep.id), "error": str(e)},
                        exc_info=True,
                    )

        clusters = self.clusterer.cluster(episodes)
        gists = await self.extractor.extract_from_clusters(clusters)
        gists = self._apply_gist_guardrails(gists, clusters)

        alignments = await self.aligner.align_batch(tenant_id, user_id, gists)
        migration = await self.migrator.migrate(
            tenant_id,
            user_id,
            alignments,
            mark_episodes_consolidated=True,
            compress_episodes=False,
        )

        completed = datetime.now(UTC)
        return ConsolidationReport(
            tenant_id=tenant_id,
            user_id=user_id,
            started_at=started,
            completed_at=completed,
            episodes_sampled=len(episodes),
            clusters_formed=len(clusters),
            gists_extracted=len(gists),
            migration=migration,
            elapsed_seconds=(completed - started).total_seconds(),
        )

    async def start_background_worker(self):
        """Start background consolidation worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_background_worker(self):
        """Stop background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

    async def _worker_loop(self):
        """Background worker loop."""
        log = get_logger(__name__)
        while self._running:
            task = await self.scheduler.get_next_task()
            if task:
                try:
                    report = await self.consolidate(
                        task.tenant_id,
                        task.user_id,
                        task,
                    )
                    if report.gists_extracted:
                        log.info(
                            "consolidation complete",
                            tenant_id=task.tenant_id,
                            user_id=task.user_id,
                            gists_extracted=report.gists_extracted,
                            clusters_formed=report.clusters_formed,
                        )
                except Exception:
                    log.exception(
                        "consolidation failed",
                        tenant_id=task.tenant_id,
                        user_id=task.user_id,
                    )
            else:
                await asyncio.sleep(1)

    def _apply_gist_guardrails(
        self,
        gists: list[ExtractedGist],
        clusters: list[EpisodeCluster],
    ) -> list[ExtractedGist]:
        if not clusters:
            return gists

        cluster_by_episode: dict[str, EpisodeCluster] = {}
        for episode_cluster in clusters:
            for ep in episode_cluster.episodes:
                cluster_by_episode[str(ep.id)] = episode_cluster

        accepted: list[ExtractedGist] = []
        covered_clusters: set[int] = set()
        rejected_clusters: set[int] = set()

        for gist in gists:
            cluster = self._cluster_for_gist(gist, cluster_by_episode)
            if cluster is None:
                accepted.append(gist)
                continue
            if self._is_valid_gist(gist, cluster):
                accepted.append(gist)
                covered_clusters.add(cluster.cluster_id)
            else:
                rejected_clusters.add(cluster.cluster_id)
                logger.warning(
                    "consolidation_gist_rejected",
                    extra={
                        "cluster_id": cluster.cluster_id,
                        "gist_text": gist.text[:120],
                    },
                )

        for cluster in clusters:
            if cluster.cluster_id in covered_clusters:
                continue
            if cluster.cluster_id in rejected_clusters or self._is_mixed_topic_cluster(cluster):
                fallback = self._fallback_gist(cluster)
                if fallback is not None:
                    accepted.append(fallback)

        return accepted

    @staticmethod
    def _cluster_for_gist(
        gist: ExtractedGist,
        cluster_by_episode: dict[str, EpisodeCluster],
    ) -> EpisodeCluster | None:
        for episode_id in gist.supporting_episode_ids:
            cluster = cluster_by_episode.get(str(episode_id))
            if cluster is not None:
                return cluster
        return None

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(_TOKEN_RE.findall(text.lower()))

    def _is_valid_gist(self, gist: ExtractedGist, cluster: EpisodeCluster) -> bool:
        text = (gist.text or "").strip()
        if not text:
            return False
        if not (0.0 <= gist.confidence <= 1.0):
            return False

        # --- model path: gist quality scoring ---
        try:
            if getattr(self.modelpack, "has_task_model", lambda _: False)("consolidation_gist_quality"):
                score_pred = self.modelpack.predict_score_single(
                    "consolidation_gist_quality", text,
                )
                if score_pred is not None:
                    return score_pred.score >= 0.5
        except Exception:
            pass

        # --- heuristic path: string-blacklist and overlap checks ---
        if len(cluster.episodes) <= 1:
            return True
        lowered = text.lower()
        if any(pattern in lowered for pattern in _GENERIC_GIST_PATTERNS):
            return False

        gist_tokens = self._token_set(text)
        if not gist_tokens:
            return False
        overlap_hits = 0
        for ep in cluster.episodes:
            ep_tokens = self._token_set(ep.text)
            if gist_tokens & ep_tokens:
                overlap_hits += 1
        overlap_ratio = overlap_hits / max(1, len(cluster.episodes))
        min_overlap = 0.5 if self._is_mixed_topic_cluster(cluster) else 0.3
        return overlap_ratio >= min_overlap

    @staticmethod
    def _is_mixed_topic_cluster(cluster: EpisodeCluster) -> bool:
        if len(cluster.episodes) <= 1:
            return False
        types = {
            (ep.type.value if hasattr(ep.type, "value") else str(ep.type)).lower()
            for ep in cluster.episodes
        }
        if len(types) >= 3:
            return True
        return len(types) >= 2 and not cluster.common_entities

    @staticmethod
    def _fallback_gist(cluster: EpisodeCluster) -> ExtractedGist | None:
        if not cluster.episodes:
            return None
        anchor = max(cluster.episodes, key=lambda ep: ep.confidence)
        dominant = (cluster.dominant_type or "").lower()
        gist_type = "policy" if dominant == "constraint" else "summary"
        if dominant == "preference":
            gist_type = "preference"
        return ExtractedGist(
            text=anchor.text[:220],
            gist_type=gist_type,
            confidence=max(0.45, min(0.75, anchor.confidence * 0.8)),
            supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
            source_memory_types=[
                (ep.type.value if hasattr(ep.type, "value") else str(ep.type)).lower()
                for ep in cluster.episodes
            ],
        )
