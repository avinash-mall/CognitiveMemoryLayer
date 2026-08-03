"""Consolidation worker orchestrating the full flow."""

import asyncio
import contextlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from ..core.config import get_settings
from ..memory.neocortical.store import NeocorticalStore
from ..storage.base import MemoryStoreBase
from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger
from .clusterer import EpisodeCluster, SemanticClusterer
from .migrator import ConsolidationMigrator, MigrationResult
from .sampler import EpisodeSampler
from .schema_aligner import SchemaAligner
from .summarizer import ExtractedGist, GistExtractor
from .triggers import ConsolidationScheduler, ConsolidationTask, TriggerType

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

    #: Clusters below the recurrence threshold, so never sent to the LLM.
    clusters_skipped_no_recurrence: int = 0
    #: Gists the detail-recovery pass added on top of the first-pass summaries.
    details_recovered: int = 0

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
        scheduler: ConsolidationScheduler | None = None,
    ):
        self.episodic_store = episodic_store
        self.sampler = EpisodeSampler(episodic_store)
        self.clusterer = SemanticClusterer()
        self.extractor = GistExtractor(llm_client)
        self.aligner = SchemaAligner(neocortical_store.facts)
        self.migrator = ConsolidationMigrator(neocortical_store, episodic_store)
        self.llm = llm_client

        self.scheduler = scheduler or ConsolidationScheduler()

        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._sweep_task: asyncio.Task | None = None

        # Counters, not logs. The plan's instruction for this item is "verify by trigger
        # count, not by reading the diff" — three features have shipped here flagged-on
        # with no caller, and each time the diff looked correct.
        self.sweeps_run = 0
        self.tasks_enqueued = 0
        self.consolidations_run = 0

    @property
    def status(self) -> dict:
        """Whether the scheduler is actually firing. Surfaced by the admin route."""
        return {
            "running": self._running,
            "sweeps_run": self.sweeps_run,
            "tasks_enqueued": self.tasks_enqueued,
            "consolidations_run": self.consolidations_run,
            "pending_tasks": self.scheduler.pending_count(),
        }

    async def consolidate(
        self,
        tenant_id: str,
        user_id: str,
        task: ConsolidationTask | None = None,
    ) -> ConsolidationReport:
        """Run full consolidation for a user."""
        started = datetime.now(UTC)

        episode_limit = task.episode_limit if task else 200
        episodes = await self.sampler.sample(tenant_id, max_episodes=episode_limit)

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
                            if await ConstraintExtractor.detect_supersession(
                                old_obj, c, llm_client=self.llm
                            ):
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

        # Recurrence gate: only clusters showing repetition earn an LLM call. A cluster
        # of one is a single episode with nothing to generalise from, so "extracting a
        # gist" restates it and then demotes the original in favour of the restatement.
        # Skipping it costs nothing retrievable — raw episodes stay in episodic memory,
        # and the strongest ablation in the research pass (81.10 -> 51.88 when the raw
        # layer is removed) says that is the layer to protect.
        #
        # Defaults to 1, which gates nothing: at 2 a tenant whose episodes never cluster
        # stops producing gists entirely, and that trade is unmeasured. The knob is what
        # the fresh-ingest arm moves.
        settings = get_settings()
        recurrence_min = settings.features.consolidation_recurrence_min
        recurrent = [c for c in clusters if len(c.episodes) >= recurrence_min]
        skipped = len(clusters) - len(recurrent)

        gists = await self.extractor.extract_from_clusters(recurrent)
        gists = self._apply_gist_guardrails(gists, recurrent)

        # Second pass, conditioned on the gist: a summary generalises, and generalising
        # drops the named entities and quantities a later question asks about. Its own
        # flag rather than the scheduler's, so the fresh-ingest arm can attribute this
        # separately — bundling two changes into one arm has cost this project three
        # arms of untangling already.
        recovered: list[ExtractedGist] = []
        if settings.features.consolidation_detail_recovery_enabled:
            recovered = await self.extractor.recover_details(recurrent, gists)
            gists = gists + recovered

        alignments = await self.aligner.align_batch(tenant_id, gists)
        migration = await self.migrator.migrate(
            tenant_id,
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
            clusters_skipped_no_recurrence=skipped,
            gists_extracted=len(gists),
            details_recovered=len(recovered),
            migration=migration,
            elapsed_seconds=(completed - started).total_seconds(),
        )

    async def start_background_worker(self):
        """Start the consolidation consumer and the sweep that feeds it.

        Starting only the consumer — which is all this method used to do, and it had no
        caller — produces a loop that polls an empty queue forever, because nothing but
        the two admin HTTP routes ever enqueued anything.
        """
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._sweep_task = asyncio.create_task(self._sweep_loop())

    async def stop_background_worker(self):
        """Stop background worker."""
        self._running = False
        for task in (self._worker_task, self._sweep_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._worker_task = None
        self._sweep_task = None

    async def sweep_once(self) -> int:
        """Enqueue consolidation for every tenant over the episode quota.

        Returns the number of tenants enqueued. Separate from the loop so it can be
        driven directly by a test or an admin route — a background-only entry point is
        how this subsystem went unverified for so long.
        """
        self.sweeps_run += 1
        candidates = await self.episodic_store.unconsolidated_counts_by_tenant(
            min_count=self.scheduler.quota_episodes,
        )
        for tenant_id, count in candidates:
            await self.scheduler.enqueue(
                tenant_id,
                tenant_id,
                TriggerType.QUOTA,
                f"Quota: {count} un-consolidated records",
            )
            self.tasks_enqueued += 1
        if candidates:
            logger.info(
                "consolidation_sweep_enqueued",
                extra={"tenants": len(candidates), "quota": self.scheduler.quota_episodes},
            )
        return len(candidates)

    async def _sweep_loop(self):
        """Periodically look for tenants with enough un-consolidated material."""
        interval = self.scheduler.default_interval.total_seconds()
        while self._running:
            try:
                await self.sweep_once()
            except Exception:
                logger.exception("consolidation_sweep_failed")
            await asyncio.sleep(interval)

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
                    self.consolidations_run += 1
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
                for fallback in self._split_or_fallback_gists(cluster):
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
        # ponytail: deliberately NOT src.utils.similarity.word_set. This tokenises with
        # _TOKEN_RE, so "food." -> {"food"} where word_set gives {"food."} — and it feeds
        # an intersection-nonempty gist check, not a Jaccard ratio. No test pins the
        # gist-overlap threshold, so switching tokenisers would change which gists survive
        # on punctuated text with nothing to catch the regression. Convert it together with
        # a test that fixes the overlap behaviour.
        return set(_TOKEN_RE.findall(text.lower()))

    def _is_valid_gist(self, gist: ExtractedGist, cluster: EpisodeCluster) -> bool:
        text = (gist.text or "").strip()
        if not text:
            return False
        if not (0.0 <= gist.confidence <= 1.0):
            return False

        # Heuristic quality gate: string-blacklist and overlap checks
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
    def _split_or_fallback_gists(cluster: EpisodeCluster) -> list[ExtractedGist]:
        """Split mixed-topic constraint clusters into per-subtype gists.

        When a mixed cluster contains multiple constraint subtypes, each
        subtype gets its own gist to avoid collapsing distinct constraints
        into a single generic "policy" gist.  Non-constraint episodes in the
        same cluster are summarised separately.
        """
        if not cluster.episodes:
            return []

        constraint_subtypes = {"goal", "state", "value", "causal", "policy"}
        by_subtype: dict[str, list] = {}
        non_constraint: list = []

        for ep in cluster.episodes:
            ep_type = (ep.type.value if hasattr(ep.type, "value") else str(ep.type)).lower()
            meta = getattr(ep, "metadata", None) or {}
            constraint_type = None
            for c in meta.get("constraints", []):
                if isinstance(c, dict):
                    constraint_type = (c.get("constraint_type") or "").lower()
                    break
            if ep_type in constraint_subtypes:
                by_subtype.setdefault(ep_type, []).append(ep)
            elif ep_type == "constraint" and constraint_type in constraint_subtypes:
                by_subtype.setdefault(constraint_type, []).append(ep)
            elif ep_type == "constraint":
                by_subtype.setdefault("policy", []).append(ep)
            else:
                non_constraint.append(ep)

        gists: list[ExtractedGist] = []

        for subtype, episodes in by_subtype.items():
            anchor = max(episodes, key=lambda ep: ep.confidence)
            gists.append(
                ExtractedGist(
                    text=anchor.text[:220],
                    gist_type=subtype,
                    confidence=max(0.45, min(0.75, anchor.confidence * 0.8)),
                    supporting_episode_ids=[str(ep.id) for ep in episodes],
                    source_memory_types=[
                        (ep.type.value if hasattr(ep.type, "value") else str(ep.type)).lower()
                        for ep in episodes
                    ],
                )
            )

        if non_constraint:
            anchor = max(non_constraint, key=lambda ep: ep.confidence)
            dominant = (cluster.dominant_type or "").lower()
            if dominant == "preference":
                gist_type = "preference"
            elif dominant == "semantic_fact":
                gist_type = "fact"
            else:
                gist_type = "summary"
            gists.append(
                ExtractedGist(
                    text=anchor.text[:220],
                    gist_type=gist_type,
                    confidence=max(0.45, min(0.75, anchor.confidence * 0.8)),
                    supporting_episode_ids=[str(ep.id) for ep in non_constraint],
                    source_memory_types=[
                        (ep.type.value if hasattr(ep.type, "value") else str(ep.type)).lower()
                        for ep in non_constraint
                    ],
                )
            )

        if not gists:
            anchor = max(cluster.episodes, key=lambda ep: ep.confidence)
            gists.append(
                ExtractedGist(
                    text=anchor.text[:220],
                    gist_type="summary",
                    confidence=max(0.45, min(0.75, anchor.confidence * 0.8)),
                    supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
                    source_memory_types=[
                        (ep.type.value if hasattr(ep.type, "value") else str(ep.type)).lower()
                        for ep in cluster.episodes
                    ],
                )
            )

        return gists

    @staticmethod
    def _fallback_gist(cluster: EpisodeCluster) -> ExtractedGist | None:
        """Return a single fallback gist for callers that need one best-effort gist."""
        gists = ConsolidationWorker._split_or_fallback_gists(cluster)
        if not gists:
            return None

        dominant = (getattr(cluster, "dominant_type", None) or "").lower()
        if dominant:
            for gist in gists:
                if gist.gist_type == dominant:
                    return gist
            if dominant in {"goal", "value", "state", "causal", "policy"}:
                chosen = max(gists, key=lambda gist: gist.confidence)
                return ExtractedGist(
                    text=chosen.text,
                    gist_type=dominant,
                    confidence=chosen.confidence,
                    supporting_episode_ids=list(chosen.supporting_episode_ids),
                    key=chosen.key,
                    subject=chosen.subject,
                    predicate=chosen.predicate,
                    value=chosen.value,
                    source_memory_types=chosen.source_memory_types,
                )

        return max(gists, key=lambda gist: gist.confidence)
