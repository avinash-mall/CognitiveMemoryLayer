"""One-time backfill: sync entities and relations from Postgres into Neo4j.

Usage (from repo root, with .env loaded):
    python -m scripts.backfill_neo4j [--tenant TENANT_ID] [--batch-size 500] [--dry-run]

The script iterates over all memory_records in Postgres that contain
non-empty ``entities`` or ``relations`` JSON and MERGEs the corresponding
nodes and edges into Neo4j.  It is safe to run multiple times — Neo4j
MERGE is idempotent.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path so ``src`` package resolves.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sqlalchemy import func, select  # noqa: E402

from src.core.schemas import EntityMention, Relation  # noqa: E402
from src.memory.neocortical.store import NeocorticalStore  # noqa: E402
from src.memory.neocortical.fact_store import SemanticFactStore  # noqa: E402
from src.storage.connection import DatabaseManager  # noqa: E402
from src.storage.models import MemoryRecordModel  # noqa: E402
from src.storage.neo4j import Neo4jGraphStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill(
    tenant_id: str | None = None,
    batch_size: int = 500,
    dry_run: bool = False,
) -> dict:
    """Scan Postgres and push entities/relations into Neo4j."""
    db = DatabaseManager.get_instance()
    graph_store = Neo4jGraphStore(db.neo4j_driver)
    fact_store = SemanticFactStore(db.pg_session)
    neocortical = NeocorticalStore(graph_store, fact_store)

    stats = {
        "records_scanned": 0,
        "entities_synced": 0,
        "relations_synced": 0,
        "errors": 0,
    }

    async with db.pg_session() as session:
        # Count total records to report progress
        count_q = select(func.count()).select_from(MemoryRecordModel)
        if tenant_id:
            count_q = count_q.where(MemoryRecordModel.tenant_id == tenant_id)
        total = (await session.execute(count_q)).scalar() or 0
        logger.info("Total records to scan: %d", total)

        offset = 0
        while True:
            q = (
                select(MemoryRecordModel)
                .order_by(MemoryRecordModel.timestamp)
                .offset(offset)
                .limit(batch_size)
            )
            if tenant_id:
                q = q.where(MemoryRecordModel.tenant_id == tenant_id)

            rows = (await session.execute(q)).scalars().all()
            if not rows:
                break

            for row in rows:
                stats["records_scanned"] += 1
                tid = row.tenant_id

                # --- Entities ---
                raw_entities = row.entities or []
                for ent_data in raw_entities:
                    try:
                        if isinstance(ent_data, dict):
                            entity = EntityMention(**ent_data)
                        else:
                            continue
                        if dry_run:
                            stats["entities_synced"] += 1
                            continue
                        await graph_store.merge_node(
                            tenant_id=tid,
                            scope_id=tid,
                            entity=entity.normalized,
                            entity_type=entity.entity_type,
                        )
                        stats["entities_synced"] += 1
                    except Exception:
                        stats["errors"] += 1
                        logger.warning(
                            "Failed to sync entity for memory %s",
                            row.id,
                            exc_info=True,
                        )

                # --- Relations ---
                raw_relations = row.relations or []
                for rel_data in raw_relations:
                    try:
                        if isinstance(rel_data, dict):
                            relation = Relation(**rel_data)
                        else:
                            continue
                        if dry_run:
                            stats["relations_synced"] += 1
                            continue
                        await neocortical.store_relation(
                            tenant_id=tid,
                            relation=relation,
                            evidence_ids=[str(row.id)],
                        )
                        stats["relations_synced"] += 1
                    except Exception:
                        stats["errors"] += 1
                        logger.warning(
                            "Failed to sync relation for memory %s",
                            row.id,
                            exc_info=True,
                        )

            offset += batch_size
            logger.info(
                "Progress: %d / %d records  (entities=%d, relations=%d, errors=%d)",
                stats["records_scanned"],
                total,
                stats["entities_synced"],
                stats["relations_synced"],
                stats["errors"],
            )

    await db.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill Neo4j graph from Postgres memories")
    parser.add_argument("--tenant", default=None, help="Limit to a specific tenant ID")
    parser.add_argument("--batch-size", type=int, default=500, help="Records per batch")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't write to Neo4j")
    args = parser.parse_args()

    logger.info(
        "Starting Neo4j backfill (tenant=%s, batch_size=%d, dry_run=%s)",
        args.tenant or "ALL",
        args.batch_size,
        args.dry_run,
    )

    stats = asyncio.run(
        backfill(tenant_id=args.tenant, batch_size=args.batch_size, dry_run=args.dry_run)
    )

    logger.info("Backfill complete: %s", stats)
    if stats["errors"]:
        logger.warning("%d errors occurred — check logs above", stats["errors"])


if __name__ == "__main__":
    main()
