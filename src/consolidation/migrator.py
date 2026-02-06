"""Migration of consolidated gists to semantic store."""
from dataclasses import dataclass
from datetime import datetime
from typing import List
from uuid import UUID

from .schema_aligner import AlignmentResult
from ..core.enums import MemoryStatus
from ..memory.neocortical.store import NeocorticalStore
from ..storage.postgres import PostgresMemoryStore


@dataclass
class MigrationResult:
    """Result of migrating consolidated knowledge."""

    gists_processed: int
    facts_created: int
    facts_updated: int
    episodes_marked: int
    errors: List[str]


class ConsolidationMigrator:
    """Migrates consolidated gists to semantic store. Marks source episodes as consolidated."""

    def __init__(
        self,
        neocortical: NeocorticalStore,
        episodic_store: PostgresMemoryStore,
    ):
        self.semantic = neocortical
        self.episodic = episodic_store

    async def migrate(
        self,
        tenant_id: str,
        user_id: str,
        alignments: List[AlignmentResult],
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
                result.errors.append(
                    f"Failed to migrate gist '{gist.text[:50]}': {e}"
                )

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
        key = (
            schema.get("key")
            or gist.key
            or f"user:custom:{hash(gist.text) % 10000}"
        )
        await self.semantic.store_fact(
            tenant_id=tenant_id,
            key=key,
            value=gist.value if gist.value is not None else gist.text,
            confidence=gist.confidence,
            evidence_ids=gist.supporting_episode_ids,
        )

    async def _mark_episodes_consolidated(
        self,
        episode_ids: List[str],
        compress: bool = False,
    ) -> int:
        marked = 0
        for ep_id in episode_ids:
            try:
                ep_uuid = UUID(ep_id)
                patch: dict = {
                    "metadata": {
                        "consolidated": True,
                        "consolidated_at": datetime.utcnow().isoformat(),
                    }
                }
                if compress:
                    patch["status"] = MemoryStatus.COMPRESSED.value
                await self.episodic.update(ep_uuid, patch, increment_version=False)
                marked += 1
            except (ValueError, Exception):
                continue
        return marked
