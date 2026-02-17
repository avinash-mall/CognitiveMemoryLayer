"""Manual verification: ingest profile with constraints, run decision-style queries, print packet markdown.

Run from repo root: python scripts/verify_constraint_flow.py (or PYTHONPATH=. python scripts/verify_constraint_flow.py).
Requires .env with database and optional LLM/embedding settings.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure project root on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Load .env
try:
    from dotenv import load_dotenv

    load_dotenv(_root / ".env")
except ImportError:
    pass


async def main() -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from src.core.config import ensure_asyncpg_url, get_settings
    from src.memory.hippocampal.store import HippocampalStore
    from src.memory.neocortical.fact_store import SemanticFactStore
    from src.memory.neocortical.store import NeocorticalStore
    from src.memory.working.models import ChunkType, SemanticChunk
    from src.retrieval.memory_retriever import MemoryRetriever
    from src.storage.models import Base
    from src.storage.postgres import PostgresMemoryStore
    from src.utils.embeddings import MockEmbeddingClient

    settings = get_settings()
    postgres_url = ensure_asyncpg_url(settings.database.postgres_url)
    engine = create_async_engine(postgres_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    pg_store = PostgresMemoryStore(session_factory)
    dims = settings.embedding.dimensions
    embeddings = MockEmbeddingClient(dimensions=dims)
    hippocampal = HippocampalStore(
        vector_store=pg_store,
        embedding_client=embeddings,
        entity_extractor=None,
        relation_extractor=None,
        write_gate=None,
        redactor=None,
    )

    fact_store = SemanticFactStore(session_factory)
    mock_graph = MagicMock()
    mock_graph.merge_edge = AsyncMock(return_value="e1")
    mock_graph.personalized_pagerank = AsyncMock(return_value=[])
    mock_graph.get_entity_facts_batch = AsyncMock(return_value=[])
    neocortical = NeocorticalStore(graph_store=mock_graph, fact_store=fact_store)

    tenant_id = os.environ.get("CML_VERIFY_TENANT", "verify-constraint-tenant")

    # Ingest constraints
    chunks = [
        SemanticChunk(
            id="1",
            text="I want to save money and avoid unnecessary spending.",
            chunk_type=ChunkType.CONSTRAINT,
            salience=0.9,
            confidence=0.85,
        ),
        SemanticChunk(
            id="2",
            text="I never eat shellfish; I'm allergic.",
            chunk_type=ChunkType.CONSTRAINT,
            salience=0.95,
            confidence=0.9,
        ),
    ]
    for c in chunks:
        rec, _ = await hippocampal.encode_chunk(tenant_id, c, existing_memories=None)
        if rec:
            print(f"Stored constraint: {rec.text[:60]}... (key={rec.key})")

    # Retrieve with decision-style query
    retriever = MemoryRetriever(hippocampal=hippocampal, neocortical=neocortical, llm_client=None)
    packet = await retriever.retrieve(
        tenant_id,
        "Can I afford dinner at a restaurant?",
        max_results=15,
        return_packet=True,
    )
    markdown = retriever.packet_builder.to_llm_context(packet, max_tokens=2000, format="markdown")

    print("\n--- Packet markdown (constraints should be first and prominent) ---\n")
    print(markdown)
    print("\n--- End ---")

    if "Active Constraints" in markdown and "Must Follow" in markdown:
        print("\n[OK] Active Constraints (Must Follow) section is visible and prominent.")
    else:
        print("\n[WARN] Expected 'Active Constraints (Must Follow)' section in markdown.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
