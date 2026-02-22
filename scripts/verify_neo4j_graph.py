"""Diagnostic script: connect to Neo4j and run graph queries.

Run from repo root: python scripts/verify_neo4j_graph.py
Requires .env with DATABASE__NEO4J_URL, DATABASE__NEO4J_USER, DATABASE__NEO4J_PASSWORD.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from dotenv import load_dotenv

    load_dotenv(_root / ".env")
except ImportError:
    pass


async def main() -> None:
    from neo4j import AsyncGraphDatabase

    from src.core.config import get_settings

    settings = get_settings()
    url = settings.database.neo4j_url or "bolt://localhost:7687"
    user = settings.database.neo4j_user or "neo4j"
    password = settings.database.neo4j_password or ""

    print("=== Neo4j Graph Diagnostics ===\n")
    print(f"URL: {url}")
    print(f"User: {user}\n")

    driver = AsyncGraphDatabase.driver(url, auth=(user, password))

    try:
        async with driver.session() as session:
            # 1. List all relationship types and counts
            print("--- 1. Relationship types and counts ---")
            result = await session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC"
            )
            records = await result.data()
            if records:
                for r in records:
                    print(f"  {r['rel_type']}: {r['cnt']}")
            else:
                print("  (no relationships)")
            print()

            # 2. Find entities with long/suspicious text
            print("--- 2. Entities with long text (>100 chars) ---")
            result = await session.run("""
                MATCH (n:Entity)
                WHERE size(n.entity) > 100
                RETURN n.entity AS entity, n.entity_type AS entity_type, n.tenant_id AS tid
                LIMIT 20
                """)
            records = await result.data()
            if records:
                for r in records:
                    ent = (r.get("entity") or "")[:80]
                    print(f"  {ent}... [{r.get('entity_type', '?')}]")
            else:
                print("  (none)")
            print()

            # 3. Entity type distribution
            print("--- 3. Entity type distribution ---")
            result = await session.run("""
                MATCH (n:Entity)
                RETURN n.entity_type AS entity_type, count(*) AS cnt
                ORDER BY cnt DESC
                """)
            records = await result.data()
            if records:
                for r in records:
                    print(f"  {r.get('entity_type', 'null')}: {r['cnt']}")
            else:
                print("  (no entities)")
            print()

            # 4. Sample nodes and edges
            print("--- 4. Sample triples (subject - predicate -> object) ---")
            result = await session.run("""
                MATCH (s:Entity)-[r]->(o:Entity)
                RETURN s.entity AS subject, type(r) AS predicate, o.entity AS object,
                       s.entity_type AS subj_type, o.entity_type AS obj_type
                LIMIT 30
                """)
            records = await result.data()
            if records:
                for r in records[:15]:
                    print(f"  {r.get('subject')} -[{r.get('predicate')}]-> {r.get('object')}")
                if len(records) > 15:
                    print(f"  ... and {len(records) - 15} more")
            else:
                print("  (no edges)")
            print()

            # 5. Entities that look like system prompt fragments
            print("--- 5. Possible system-prompt-like entities ---")
            result = await session.run("""
                MATCH (n:Entity)
                WHERE toLower(n.entity) CONTAINS 'you are'
                   OR toLower(n.entity) CONTAINS 'assistant'
                   OR toLower(n.entity) CONTAINS 'system'
                   OR toLower(n.entity) CONTAINS 'instruction'
                RETURN n.entity AS entity, n.entity_type AS entity_type
                LIMIT 20
                """)
            records = await result.data()
            if records:
                for r in records:
                    print(f"  {r.get('entity')} [{r.get('entity_type', '?')}]")
            else:
                print("  (none)")
            print()

    finally:
        await driver.close()

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
