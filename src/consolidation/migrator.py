"""Migration of consolidated gists to semantic store."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import structlog

from ..core.enums import MemoryStatus
from ..memory.neocortical.store import NeocorticalStore
from ..storage.base import MemoryStoreBase
from .schema_aligner import AlignmentResult

_logger = structlog.get_logger(__name__)


def _stable_fact_key(prefix: str, text: str) -> str:
    """Generate a stable, deterministic key for a semantic fact.

    Uses SHA256 (not Python ``hash()``) so the key is identical across
    different Python processes, workers, deployments, and restarts.

    Format: ``{prefix}:{sha256_hex[:16]}``
    Collision probability: ~1 in 2^64 (negligible).
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{h}"


@dataclass
class MigrationResult:
    """Result of migrating consolidated knowledge."""

    gists_processed: int
    facts_created: int
    facts_updated: int
    episodes_marked: int
    errors: list[str]


class ConsolidationMigrator:
    """Migrates consolidated gists to semantic store. Marks source episodes as consolidated."""

    def __init__(
        self,
        neocortical: NeocorticalStore,
        episodic_store: MemoryStoreBase,
    ):
        self.semantic = neocortical
        self.episodic = episodic_store

    async def migrate(
        self,
        tenant_id: str,
        user_id: str,
        alignments: list[AlignmentResult],
        mark_episodes_consolidated: bool = True,
        compress_episodes: bool = False,
    ) -> MigrationResult:
        """Migrate aligned gists to semantic store."""
        result = MigrationResult(
            gists_processed=0,
            facts_created=0,
            facts_updated=0,
            episodes_marked=0,
            errors=[],
        )

        for alignment in alignments:
            try:
                gist = alignment.gist

                if alignment.can_integrate_rapidly and alignment.integration_key:
                    await self._update_existing_fact(tenant_id, alignment)
                    result.facts_updated += 1
                else:
                    await self._create_new_fact(tenant_id, alignment)
                    result.facts_created += 1

                result.gists_processed += 1

                if mark_episodes_consolidated:
                    marked = await self._mark_episodes_consolidated(
                        gist.supporting_episode_ids,
                        compress_episodes,
                    )
                    result.episodes_marked += marked

            except Exception as e:
                gist_preview = "unknown"
                try:
                    if alignment and getattr(alignment, "gist", None):
                        gist_preview = (getattr(alignment.gist, "text", None) or "unknown")[:50]
                except Exception:
                    _logger.debug("gist_preview_extraction_failed")
                result.errors.append(f"Failed to migrate gist '{gist_preview}': {e}")

        return result

    async def _update_existing_fact(
        self,
        tenant_id: str,
        alignment: AlignmentResult,
    ):
        gist = alignment.gist
        await self.semantic.store_fact(
            tenant_id=tenant_id,
            key=alignment.integration_key or gist.key or "user:custom:unknown",
            value=gist.value if gist.value is not None else gist.text,
            confidence=gist.confidence,
            evidence_ids=gist.supporting_episode_ids,
        )

    async def _create_new_fact(
        self,
        tenant_id: str,
        alignment: AlignmentResult,
    ):
        gist = alignment.gist
        schema = alignment.suggested_schema or {}
        key = schema.get("key") or gist.key or _stable_fact_key("user:custom", gist.text)
        await self.semantic.store_fact(
            tenant_id=tenant_id,
            key=key,
            value=gist.value if gist.value is not None else gist.text,
            confidence=gist.confidence,
            evidence_ids=gist.supporting_episode_ids,
        )

    async def _mark_episodes_consolidated(
        self,
        episode_ids: list[str],
        compress: bool = False,
    ) -> int:
        marked = 0
        now_iso = datetime.now(UTC).isoformat()
        for ep_id in episode_ids:
            try:
                ep_uuid = UUID(ep_id)
                episode = await self.episodic.get_by_id(ep_uuid)
                if not episode:
                    continue
                # BUG-03: merge with existing metadata instead of replacing
                merged_metadata = {
                    **(episode.metadata or {}),
                    "consolidated": True,
                    "consolidated_at": now_iso,
                }
                patch: dict = {"metadata": merged_metadata}
                if compress:
                    patch["status"] = MemoryStatus.COMPRESSED.value
                await self.episodic.update(ep_uuid, patch, increment_version=False)
                marked += 1
            except Exception:
                continue
        return marked
