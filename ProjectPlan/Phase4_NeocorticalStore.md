# Phase 4: Neocortical Store (Semantic Memory)

## Overview
**Duration**: Week 4-5  
**Goal**: Implement the structured semantic memory store with knowledge graph, schema management, and fact consolidation.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                  From Hippocampal Store                          │
│       (Episodic memories with entities and relations)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Consolidation Bridge                           │
│   - Pattern detection across episodes                            │
│   - Gist extraction                                              │
│   - Schema alignment                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Neocortical Store                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Knowledge Graph (Neo4j)                     │   │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐            │   │
│   │   │  User   │───▶│Location │◀───│  Event  │            │   │
│   │   │  Node   │    │  Node   │    │  Node   │            │   │
│   │   └─────────┘    └─────────┘    └─────────┘            │   │
│   │        │              │              │                  │   │
│   │        ▼              ▼              ▼                  │   │
│   │   [prefers]      [located_in]   [occurred_at]          │   │
│   └─────────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Semantic Fact Store                         │   │
│   │   - Structured key-value facts                           │   │
│   │   - Schema-aligned properties                            │   │
│   │   - Version history                                      │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 4.1: Neo4j Graph Store Implementation

### Description
Implement the knowledge graph backend using Neo4j for entity relationships and graph queries.

### Subtask 4.1.1: Neo4j Connection and Base Operations

```python
# src/storage/neo4j.py
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
from .base import GraphStoreBase
from ..core.config import get_settings

@dataclass
class GraphNode:
    id: str
    entity: str
    entity_type: str
    properties: Dict[str, Any]
    tenant_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

@dataclass
class GraphEdge:
    id: str
    source_id: str
    target_id: str
    predicate: str
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime

class Neo4jGraphStore(GraphStoreBase):
    """
    Neo4j-based knowledge graph for semantic memory.
    Stores entities as nodes and relations as edges.
    """
    
    def __init__(self, driver: Optional[AsyncDriver] = None):
        if driver:
            self.driver = driver
        else:
            settings = get_settings()
            self.driver = AsyncGraphDatabase.driver(
                settings.database.neo4j_url,
                auth=(settings.database.neo4j_user, settings.database.neo4j_password)
            )
    
    async def close(self):
        await self.driver.close()
    
    async def merge_node(
        self,
        tenant_id: str,
        user_id: str,
        entity: str,
        entity_type: str,
        properties: Dict[str, Any] = None
    ) -> str:
        """
        Create or update a node.
        Uses MERGE to avoid duplicates.
        """
        properties = properties or {}
        now = datetime.utcnow().isoformat()
        
        query = """
        MERGE (n:Entity {
            tenant_id: $tenant_id,
            user_id: $user_id,
            entity: $entity,
            entity_type: $entity_type
        })
        ON CREATE SET
            n.created_at = $now,
            n.updated_at = $now,
            n += $properties
        ON MATCH SET
            n.updated_at = $now,
            n += $properties
        RETURN elementId(n) as node_id
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                tenant_id=tenant_id,
                user_id=user_id,
                entity=entity,
                entity_type=entity_type,
                properties=properties,
                now=now
            )
            record = await result.single()
            return record["node_id"]
    
    async def merge_edge(
        self,
        tenant_id: str,
        user_id: str,
        subject: str,
        predicate: str,
        object: str,
        properties: Dict[str, Any] = None
    ) -> str:
        """
        Create or update an edge between two nodes.
        Creates nodes if they don't exist.
        """
        properties = properties or {}
        now = datetime.utcnow().isoformat()
        
        # Normalize predicate for Neo4j relationship type
        rel_type = predicate.upper().replace(" ", "_")
        
        query = f"""
        MERGE (s:Entity {{
            tenant_id: $tenant_id,
            user_id: $user_id,
            entity: $subject
        }})
        ON CREATE SET s.created_at = $now, s.entity_type = 'UNKNOWN'
        
        MERGE (o:Entity {{
            tenant_id: $tenant_id,
            user_id: $user_id,
            entity: $object
        }})
        ON CREATE SET o.created_at = $now, o.entity_type = 'UNKNOWN'
        
        MERGE (s)-[r:{rel_type}]->(o)
        ON CREATE SET
            r.created_at = $now,
            r.updated_at = $now,
            r.confidence = $confidence,
            r += $properties
        ON MATCH SET
            r.updated_at = $now,
            r.access_count = coalesce(r.access_count, 0) + 1,
            r += $properties
        
        RETURN elementId(r) as edge_id
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                tenant_id=tenant_id,
                user_id=user_id,
                subject=subject,
                object=object,
                properties=properties,
                confidence=properties.get("confidence", 0.8),
                now=now
            )
            record = await result.single()
            return record["edge_id"]
    
    async def get_neighbors(
        self,
        tenant_id: str,
        user_id: str,
        entity: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes up to max_depth hops.
        """
        query = """
        MATCH (start:Entity {
            tenant_id: $tenant_id,
            user_id: $user_id,
            entity: $entity
        })
        CALL apoc.path.subgraphNodes(start, {
            maxLevel: $max_depth,
            relationshipFilter: null,
            labelFilter: '+Entity'
        }) YIELD node
        WHERE node.tenant_id = $tenant_id AND node.user_id = $user_id
        RETURN node.entity as entity,
               node.entity_type as entity_type,
               properties(node) as properties
        """
        
        # Fallback query without APOC
        fallback_query = f"""
        MATCH path = (start:Entity {{
            tenant_id: $tenant_id,
            user_id: $user_id,
            entity: $entity
        }})-[*1..{max_depth}]-(neighbor:Entity)
        WHERE neighbor.tenant_id = $tenant_id AND neighbor.user_id = $user_id
        RETURN DISTINCT neighbor.entity as entity,
               neighbor.entity_type as entity_type,
               properties(neighbor) as properties
        LIMIT 100
        """
        
        async with self.driver.session() as session:
            try:
                result = await session.run(
                    query,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    entity=entity,
                    max_depth=max_depth
                )
            except Exception:
                # APOC not installed, use fallback
                result = await session.run(
                    fallback_query,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    entity=entity
                )
            
            records = await result.data()
            return records
    
    async def personalized_pagerank(
        self,
        tenant_id: str,
        user_id: str,
        seed_entities: List[str],
        top_k: int = 20,
        damping: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Run Personalized PageRank from seed entities.
        
        This mimics hippocampal pattern completion:
        given partial cues (seeds), find related information.
        """
        # Query using GDS (Graph Data Science) library
        # Falls back to simple multi-hop if GDS not available
        
        gds_query = """
        MATCH (source:Entity)
        WHERE source.tenant_id = $tenant_id 
          AND source.user_id = $user_id
          AND source.entity IN $seeds
        
        CALL gds.pageRank.stream({
            nodeQuery: 'MATCH (n:Entity) WHERE n.tenant_id = $tenant_id AND n.user_id = $user_id RETURN id(n) AS id',
            relationshipQuery: 'MATCH (n1:Entity)-[r]->(n2:Entity) WHERE n1.tenant_id = $tenant_id AND n1.user_id = $user_id RETURN id(n1) AS source, id(n2) AS target',
            dampingFactor: $damping,
            sourceNodes: collect(source)
        })
        YIELD nodeId, score
        
        MATCH (n:Entity) WHERE id(n) = nodeId
        RETURN n.entity as entity, 
               n.entity_type as entity_type,
               score,
               properties(n) as properties
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        # Simpler fallback: multi-hop traversal with path length weighting
        fallback_query = """
        MATCH (seed:Entity)
        WHERE seed.tenant_id = $tenant_id 
          AND seed.user_id = $user_id
          AND seed.entity IN $seeds
        
        MATCH path = (seed)-[*1..3]-(related:Entity)
        WHERE related.tenant_id = $tenant_id AND related.user_id = $user_id
        
        WITH related, 
             min(length(path)) as min_distance,
             count(path) as path_count
        
        RETURN related.entity as entity,
               related.entity_type as entity_type,
               1.0 / (min_distance + 1) * path_count as score,
               properties(related) as properties
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        async with self.driver.session() as session:
            try:
                result = await session.run(
                    gds_query,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    seeds=seed_entities,
                    damping=damping,
                    top_k=top_k
                )
            except Exception:
                # GDS not available, use fallback
                result = await session.run(
                    fallback_query,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    seeds=seed_entities,
                    top_k=top_k
                )
            
            records = await result.data()
            return records
    
    async def get_entity_facts(
        self,
        tenant_id: str,
        user_id: str,
        entity: str
    ) -> List[Dict[str, Any]]:
        """
        Get all facts (relations) about an entity.
        """
        query = """
        MATCH (e:Entity {
            tenant_id: $tenant_id,
            user_id: $user_id,
            entity: $entity
        })-[r]-(other:Entity)
        RETURN type(r) as predicate,
               CASE 
                   WHEN startNode(r) = e THEN 'outgoing'
                   ELSE 'incoming'
               END as direction,
               other.entity as related_entity,
               other.entity_type as related_type,
               properties(r) as relation_properties
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                tenant_id=tenant_id,
                user_id=user_id,
                entity=entity
            )
            return await result.data()
    
    async def search_by_pattern(
        self,
        tenant_id: str,
        user_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        limit: int = 50
    ) -> List[Tuple[str, str, str, Dict]]:
        """
        Search for triples matching a pattern.
        None values are wildcards.
        """
        conditions = ["s.tenant_id = $tenant_id", "s.user_id = $user_id"]
        params = {"tenant_id": tenant_id, "user_id": user_id, "limit": limit}
        
        if subject:
            conditions.append("s.entity = $subject")
            params["subject"] = subject
        
        if object:
            conditions.append("o.entity = $object")
            params["object"] = object
        
        rel_pattern = "[r]" if not predicate else f"[r:{predicate.upper()}]"
        
        query = f"""
        MATCH (s:Entity)-{rel_pattern}->(o:Entity)
        WHERE {' AND '.join(conditions)}
        RETURN s.entity as subject,
               type(r) as predicate,
               o.entity as object,
               properties(r) as properties
        LIMIT $limit
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [
                (r["subject"], r["predicate"], r["object"], r["properties"])
                for r in records
            ]
    
    async def delete_entity(
        self,
        tenant_id: str,
        user_id: str,
        entity: str,
        cascade: bool = True
    ) -> int:
        """
        Delete an entity node (and optionally its edges).
        """
        if cascade:
            query = """
            MATCH (n:Entity {
                tenant_id: $tenant_id,
                user_id: $user_id,
                entity: $entity
            })
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
        else:
            query = """
            MATCH (n:Entity {
                tenant_id: $tenant_id,
                user_id: $user_id,
                entity: $entity
            })
            DELETE n
            RETURN count(n) as deleted_count
            """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                tenant_id=tenant_id,
                user_id=user_id,
                entity=entity
            )
            record = await result.single()
            return record["deleted_count"]
```

### Subtask 4.1.2: Graph Store Initialization and Schema

```python
# src/storage/neo4j_init.py
from .neo4j import Neo4jGraphStore

async def initialize_graph_schema(store: Neo4jGraphStore):
    """
    Initialize Neo4j constraints and indexes.
    """
    async with store.driver.session() as session:
        # Unique constraint on entity within tenant/user scope
        await session.run("""
            CREATE CONSTRAINT entity_unique IF NOT EXISTS
            FOR (n:Entity)
            REQUIRE (n.tenant_id, n.user_id, n.entity) IS UNIQUE
        """)
        
        # Index for fast lookups
        await session.run("""
            CREATE INDEX entity_type_idx IF NOT EXISTS
            FOR (n:Entity)
            ON (n.tenant_id, n.user_id, n.entity_type)
        """)
        
        # Index for temporal queries
        await session.run("""
            CREATE INDEX entity_time_idx IF NOT EXISTS
            FOR (n:Entity)
            ON (n.tenant_id, n.user_id, n.updated_at)
        """)
        
        # Full-text index for entity search
        await session.run("""
            CREATE FULLTEXT INDEX entity_text_idx IF NOT EXISTS
            FOR (n:Entity)
            ON EACH [n.entity]
        """)
```

---

## Task 4.2: Semantic Fact Management

### Description
Implement structured fact storage with schema alignment and versioning.

### Subtask 4.2.1: Semantic Fact Schema

```python
# src/memory/neocortical/schemas.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class FactCategory(str, Enum):
    IDENTITY = "identity"          # Name, age, etc.
    LOCATION = "location"          # Where user is/was
    PREFERENCE = "preference"      # Likes, dislikes
    RELATIONSHIP = "relationship"  # People user knows
    OCCUPATION = "occupation"      # Work, education
    TEMPORAL = "temporal"          # Schedules, events
    ATTRIBUTE = "attribute"        # General properties
    CUSTOM = "custom"

@dataclass
class FactSchema:
    """
    Schema definition for a type of fact.
    Defines validation rules and display properties.
    """
    category: FactCategory
    key_pattern: str              # e.g., "user:{category}:{property}"
    value_type: str               # "string", "number", "date", "list", "object"
    required: bool = False
    multi_valued: bool = False    # Can have multiple values
    temporal: bool = False        # Changes over time (needs valid_from/to)
    validators: List[str] = field(default_factory=list)
    
    # Display
    display_name: str = ""
    description: str = ""
    examples: List[str] = field(default_factory=list)

@dataclass
class SemanticFact:
    """
    A structured semantic fact in the neocortical store.
    """
    id: str
    tenant_id: str
    user_id: str
    
    # Fact content
    category: FactCategory
    key: str                      # Unique identifier, e.g., "user:location:current_city"
    subject: str                  # Who/what this is about
    predicate: str                # The property/relationship
    value: Any                    # The value (can be any type)
    value_type: str
    
    # Confidence and provenance
    confidence: float = 0.8
    evidence_count: int = 1       # How many episodes support this
    evidence_ids: List[str] = field(default_factory=list)
    
    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    is_current: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    supersedes_id: Optional[str] = None

# Predefined schemas for common facts
DEFAULT_FACT_SCHEMAS = {
    "user:identity:name": FactSchema(
        category=FactCategory.IDENTITY,
        key_pattern="user:identity:name",
        value_type="string",
        display_name="User's Name",
        description="The user's preferred name",
        examples=["John", "Dr. Smith"]
    ),
    "user:location:current_city": FactSchema(
        category=FactCategory.LOCATION,
        key_pattern="user:location:current_city",
        value_type="string",
        temporal=True,
        display_name="Current City",
        description="Where the user currently lives",
        examples=["Paris", "New York"]
    ),
    "user:preference:cuisine": FactSchema(
        category=FactCategory.PREFERENCE,
        key_pattern="user:preference:cuisine",
        value_type="list",
        multi_valued=True,
        display_name="Food Preferences",
        description="Types of cuisine the user likes/dislikes",
        examples=["vegetarian", "Italian", "no seafood"]
    ),
    "user:relationship:*": FactSchema(
        category=FactCategory.RELATIONSHIP,
        key_pattern="user:relationship:{person}",
        value_type="object",
        display_name="Relationship",
        description="User's relationship with someone",
        examples=["spouse: Jane", "colleague: Bob"]
    ),
}
```

### Subtask 4.2.2: Fact Store Implementation

```python
# src/memory/neocortical/fact_store.py
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, update
from sqlalchemy.dialects.postgresql import insert
from .schemas import SemanticFact, FactCategory, FactSchema, DEFAULT_FACT_SCHEMAS

class SemanticFactStore:
    """
    Stores and manages semantic facts.
    Handles versioning, temporal validity, and schema alignment.
    """
    
    def __init__(self, session_factory, schemas: Dict[str, FactSchema] = None):
        self.session_factory = session_factory
        self.schemas = schemas or DEFAULT_FACT_SCHEMAS
    
    async def upsert_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        value: Any,
        confidence: float = 0.8,
        evidence_ids: List[str] = None,
        valid_from: Optional[datetime] = None
    ) -> SemanticFact:
        """
        Insert or update a semantic fact.
        
        If fact exists:
        - If value is same: update confidence and evidence
        - If value differs: handle based on schema (temporal or overwrite)
        """
        async with self.session_factory() as session:
            # Parse key to determine category and predicate
            category, predicate = self._parse_key(key)
            schema = self._get_schema(key)
            
            # Check for existing fact
            existing = await self._get_existing_fact(
                session, tenant_id, user_id, key
            )
            
            if existing:
                return await self._update_fact(
                    session, existing, value, confidence, 
                    evidence_ids, schema, valid_from
                )
            else:
                return await self._create_fact(
                    session, tenant_id, user_id, key, category,
                    predicate, value, confidence, evidence_ids, valid_from
                )
    
    async def get_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        include_historical: bool = False
    ) -> Optional[SemanticFact]:
        """Get a fact by key."""
        async with self.session_factory() as session:
            query = """
                SELECT * FROM semantic_facts
                WHERE tenant_id = :tenant_id
                AND user_id = :user_id
                AND key = :key
            """
            
            if not include_historical:
                query += " AND is_current = true"
            
            query += " ORDER BY version DESC LIMIT 1"
            
            result = await session.execute(
                query, 
                {"tenant_id": tenant_id, "user_id": user_id, "key": key}
            )
            row = result.fetchone()
            return self._row_to_fact(row) if row else None
    
    async def get_facts_by_category(
        self,
        tenant_id: str,
        user_id: str,
        category: FactCategory,
        current_only: bool = True
    ) -> List[SemanticFact]:
        """Get all facts in a category."""
        async with self.session_factory() as session:
            query = """
                SELECT * FROM semantic_facts
                WHERE tenant_id = :tenant_id
                AND user_id = :user_id
                AND category = :category
            """
            
            if current_only:
                query += " AND is_current = true"
            
            result = await session.execute(
                query,
                {
                    "tenant_id": tenant_id, 
                    "user_id": user_id, 
                    "category": category.value
                }
            )
            return [self._row_to_fact(r) for r in result.fetchall()]
    
    async def get_user_profile(
        self,
        tenant_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get complete user profile as structured dict.
        Organized by category.
        """
        profile = {}
        
        for category in FactCategory:
            facts = await self.get_facts_by_category(
                tenant_id, user_id, category
            )
            if facts:
                profile[category.value] = {
                    f.predicate: f.value for f in facts
                }
        
        return profile
    
    async def search_facts(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int = 20
    ) -> List[SemanticFact]:
        """
        Search facts by text query.
        Searches key, subject, and value fields.
        """
        async with self.session_factory() as session:
            sql = """
                SELECT * FROM semantic_facts
                WHERE tenant_id = :tenant_id
                AND user_id = :user_id
                AND is_current = true
                AND (
                    key ILIKE :pattern
                    OR subject ILIKE :pattern
                    OR value::text ILIKE :pattern
                )
                ORDER BY confidence DESC, updated_at DESC
                LIMIT :limit
            """
            
            result = await session.execute(
                sql,
                {
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "pattern": f"%{query}%",
                    "limit": limit
                }
            )
            return [self._row_to_fact(r) for r in result.fetchall()]
    
    async def invalidate_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        reason: str = "superseded"
    ) -> bool:
        """Mark a fact as no longer current."""
        async with self.session_factory() as session:
            result = await session.execute(
                """
                UPDATE semantic_facts
                SET is_current = false, valid_to = :now
                WHERE tenant_id = :tenant_id
                AND user_id = :user_id
                AND key = :key
                AND is_current = true
                """,
                {
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "key": key,
                    "now": datetime.utcnow()
                }
            )
            await session.commit()
            return result.rowcount > 0
    
    async def _get_existing_fact(
        self,
        session: AsyncSession,
        tenant_id: str,
        user_id: str,
        key: str
    ) -> Optional[SemanticFact]:
        """Get existing current fact."""
        result = await session.execute(
            """
            SELECT * FROM semantic_facts
            WHERE tenant_id = :tenant_id
            AND user_id = :user_id
            AND key = :key
            AND is_current = true
            """,
            {"tenant_id": tenant_id, "user_id": user_id, "key": key}
        )
        row = result.fetchone()
        return self._row_to_fact(row) if row else None
    
    async def _update_fact(
        self,
        session: AsyncSession,
        existing: SemanticFact,
        new_value: Any,
        confidence: float,
        evidence_ids: List[str],
        schema: Optional[FactSchema],
        valid_from: Optional[datetime]
    ) -> SemanticFact:
        """Update existing fact."""
        if existing.value == new_value:
            # Same value: just reinforce
            await session.execute(
                """
                UPDATE semantic_facts
                SET confidence = :confidence,
                    evidence_count = evidence_count + 1,
                    evidence_ids = array_cat(evidence_ids, :new_evidence),
                    updated_at = :now
                WHERE id = :id
                """,
                {
                    "id": existing.id,
                    "confidence": min(1.0, existing.confidence + 0.1),
                    "new_evidence": evidence_ids or [],
                    "now": datetime.utcnow()
                }
            )
            await session.commit()
            existing.confidence = min(1.0, existing.confidence + 0.1)
            return existing
        else:
            # Value changed
            if schema and schema.temporal:
                # Temporal fact: create new version, invalidate old
                await session.execute(
                    """
                    UPDATE semantic_facts
                    SET is_current = false, valid_to = :now
                    WHERE id = :id
                    """,
                    {"id": existing.id, "now": valid_from or datetime.utcnow()}
                )
            
            # Create new version
            new_fact = SemanticFact(
                id=str(uuid4()),
                tenant_id=existing.tenant_id,
                user_id=existing.user_id,
                category=existing.category,
                key=existing.key,
                subject=existing.subject,
                predicate=existing.predicate,
                value=new_value,
                value_type=type(new_value).__name__,
                confidence=confidence,
                evidence_count=1,
                evidence_ids=evidence_ids or [],
                valid_from=valid_from or datetime.utcnow(),
                is_current=True,
                version=existing.version + 1,
                supersedes_id=existing.id
            )
            
            await self._insert_fact(session, new_fact)
            return new_fact
    
    async def _create_fact(
        self,
        session: AsyncSession,
        tenant_id: str,
        user_id: str,
        key: str,
        category: FactCategory,
        predicate: str,
        value: Any,
        confidence: float,
        evidence_ids: List[str],
        valid_from: Optional[datetime]
    ) -> SemanticFact:
        """Create new fact."""
        fact = SemanticFact(
            id=str(uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            category=category,
            key=key,
            subject="user",  # Default subject
            predicate=predicate,
            value=value,
            value_type=type(value).__name__,
            confidence=confidence,
            evidence_count=1,
            evidence_ids=evidence_ids or [],
            valid_from=valid_from or datetime.utcnow(),
            is_current=True,
            version=1
        )
        
        await self._insert_fact(session, fact)
        return fact
    
    async def _insert_fact(self, session: AsyncSession, fact: SemanticFact):
        """Insert fact into database."""
        import json
        
        await session.execute(
            """
            INSERT INTO semantic_facts (
                id, tenant_id, user_id, category, key, subject, predicate,
                value, value_type, confidence, evidence_count, evidence_ids,
                valid_from, valid_to, is_current, created_at, updated_at,
                version, supersedes_id
            ) VALUES (
                :id, :tenant_id, :user_id, :category, :key, :subject, :predicate,
                :value, :value_type, :confidence, :evidence_count, :evidence_ids,
                :valid_from, :valid_to, :is_current, :created_at, :updated_at,
                :version, :supersedes_id
            )
            """,
            {
                "id": fact.id,
                "tenant_id": fact.tenant_id,
                "user_id": fact.user_id,
                "category": fact.category.value,
                "key": fact.key,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "value": json.dumps(fact.value),
                "value_type": fact.value_type,
                "confidence": fact.confidence,
                "evidence_count": fact.evidence_count,
                "evidence_ids": fact.evidence_ids,
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "is_current": fact.is_current,
                "created_at": fact.created_at,
                "updated_at": fact.updated_at,
                "version": fact.version,
                "supersedes_id": fact.supersedes_id
            }
        )
        await session.commit()
    
    def _parse_key(self, key: str) -> tuple:
        """Parse key into category and predicate."""
        parts = key.split(":")
        if len(parts) >= 3:
            category = FactCategory(parts[1]) if parts[1] in [c.value for c in FactCategory] else FactCategory.CUSTOM
            predicate = ":".join(parts[2:])
        else:
            category = FactCategory.CUSTOM
            predicate = key
        return category, predicate
    
    def _get_schema(self, key: str) -> Optional[FactSchema]:
        """Get schema for a key."""
        if key in self.schemas:
            return self.schemas[key]
        
        # Check wildcard patterns
        for pattern, schema in self.schemas.items():
            if "*" in pattern:
                prefix = pattern.split("*")[0]
                if key.startswith(prefix):
                    return schema
        
        return None
    
    def _row_to_fact(self, row) -> SemanticFact:
        """Convert database row to SemanticFact."""
        import json
        return SemanticFact(
            id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            category=FactCategory(row.category),
            key=row.key,
            subject=row.subject,
            predicate=row.predicate,
            value=json.loads(row.value) if isinstance(row.value, str) else row.value,
            value_type=row.value_type,
            confidence=row.confidence,
            evidence_count=row.evidence_count,
            evidence_ids=row.evidence_ids or [],
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            is_current=row.is_current,
            created_at=row.created_at,
            updated_at=row.updated_at,
            version=row.version,
            supersedes_id=row.supersedes_id
        )
```

---

## Task 4.3: Neocortical Store Facade

### Description
Create unified interface combining graph store and fact store.

### Subtask 4.3.1: Neocortical Store Implementation

```python
# src/memory/neocortical/store.py
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from ..hippocampal.store import HippocampalStore
from ...storage.neo4j import Neo4jGraphStore
from .fact_store import SemanticFactStore
from .schemas import SemanticFact, FactCategory
from ...core.schemas import MemoryRecord, Relation

class NeocorticalStore:
    """
    The semantic memory store - the "neocortex" of the system.
    
    Combines:
    1. Knowledge graph for entity relationships
    2. Structured fact store for user profile and knowledge
    
    Responsibilities:
    - Store consolidated semantic facts
    - Maintain entity relationship graph
    - Provide fast lookups for known facts
    - Support multi-hop reasoning via graph
    """
    
    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        fact_store: SemanticFactStore
    ):
        self.graph = graph_store
        self.facts = fact_store
    
    async def store_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        value: Any,
        confidence: float = 0.8,
        evidence_ids: List[str] = None
    ) -> SemanticFact:
        """
        Store a semantic fact.
        Also updates knowledge graph if fact contains entity relations.
        """
        # Store in fact store
        fact = await self.facts.upsert_fact(
            tenant_id, user_id, key, value, confidence, evidence_ids
        )
        
        # Update graph if this represents a relation
        if ":" in key and isinstance(value, str):
            await self._sync_fact_to_graph(tenant_id, user_id, fact)
        
        return fact
    
    async def store_relation(
        self,
        tenant_id: str,
        user_id: str,
        relation: Relation,
        evidence_ids: List[str] = None
    ) -> str:
        """
        Store a relation in the knowledge graph.
        """
        edge_id = await self.graph.merge_edge(
            tenant_id, user_id,
            subject=relation.subject,
            predicate=relation.predicate,
            object=relation.object,
            properties={
                "confidence": relation.confidence,
                "evidence_ids": evidence_ids or []
            }
        )
        return edge_id
    
    async def store_relations_batch(
        self,
        tenant_id: str,
        user_id: str,
        relations: List[Relation],
        evidence_ids: List[str] = None
    ) -> List[str]:
        """Store multiple relations."""
        edge_ids = []
        for relation in relations:
            edge_id = await self.store_relation(
                tenant_id, user_id, relation, evidence_ids
            )
            edge_ids.append(edge_id)
        return edge_ids
    
    async def get_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str
    ) -> Optional[SemanticFact]:
        """Get a fact by key."""
        return await self.facts.get_fact(tenant_id, user_id, key)
    
    async def get_user_profile(
        self,
        tenant_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Get structured user profile."""
        return await self.facts.get_user_profile(tenant_id, user_id)
    
    async def query_entity(
        self,
        tenant_id: str,
        user_id: str,
        entity: str
    ) -> Dict[str, Any]:
        """
        Get all information about an entity.
        Combines graph neighbors and related facts.
        """
        # Get from graph
        graph_facts = await self.graph.get_entity_facts(
            tenant_id, user_id, entity
        )
        
        # Search related facts
        fact_results = await self.facts.search_facts(
            tenant_id, user_id, entity, limit=20
        )
        
        return {
            "entity": entity,
            "relations": graph_facts,
            "facts": [
                {"key": f.key, "value": f.value, "confidence": f.confidence}
                for f in fact_results
            ]
        }
    
    async def multi_hop_query(
        self,
        tenant_id: str,
        user_id: str,
        seed_entities: List[str],
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-hop reasoning from seed entities.
        Uses Personalized PageRank for relevance scoring.
        """
        # Get related entities via PPR
        related = await self.graph.personalized_pagerank(
            tenant_id, user_id,
            seed_entities=seed_entities,
            top_k=20
        )
        
        # Enrich with facts for top entities
        results = []
        for item in related[:10]:
            entity_info = await self.query_entity(
                tenant_id, user_id, item["entity"]
            )
            entity_info["relevance_score"] = item.get("score", 0)
            results.append(entity_info)
        
        return results
    
    async def find_schema_match(
        self,
        tenant_id: str,
        user_id: str,
        candidate_fact: str
    ) -> Optional[SemanticFact]:
        """
        Find if a candidate fact matches existing schema/knowledge.
        Used during consolidation.
        """
        # Search for similar facts
        matches = await self.facts.search_facts(
            tenant_id, user_id, candidate_fact, limit=5
        )
        
        if matches:
            # Return best match above confidence threshold
            best = max(matches, key=lambda f: f.confidence)
            if best.confidence > 0.5:
                return best
        
        return None
    
    async def text_search(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search semantic memory by text.
        """
        facts = await self.facts.search_facts(
            tenant_id, user_id, query, limit
        )
        
        return [
            {
                "type": "fact",
                "key": f.key,
                "value": f.value,
                "confidence": f.confidence,
                "category": f.category.value
            }
            for f in facts
        ]
    
    async def _sync_fact_to_graph(
        self,
        tenant_id: str,
        user_id: str,
        fact: SemanticFact
    ):
        """Sync a fact to the knowledge graph as a relation."""
        # Create edge: user -> predicate -> value
        await self.graph.merge_edge(
            tenant_id, user_id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=str(fact.value),
            properties={
                "confidence": fact.confidence,
                "fact_id": fact.id
            }
        )
```

---

## Task 4.4: Database Migration for Semantic Facts

### Subtask 4.4.1: Semantic Facts Table Migration

```python
# migrations/versions/002_semantic_facts.py
"""Add semantic facts table

Revision ID: 002
Create Date: 2026-02-03
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSON

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'semantic_facts',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('user_id', sa.String(100), nullable=False),
        
        sa.Column('category', sa.String(30), nullable=False),
        sa.Column('key', sa.String(200), nullable=False),
        sa.Column('subject', sa.String(200), nullable=False),
        sa.Column('predicate', sa.String(200), nullable=False),
        sa.Column('value', JSON, nullable=False),
        sa.Column('value_type', sa.String(50), nullable=False),
        
        sa.Column('confidence', sa.Float, default=0.8),
        sa.Column('evidence_count', sa.Integer, default=1),
        sa.Column('evidence_ids', ARRAY(sa.String), default=[]),
        
        sa.Column('valid_from', sa.DateTime, nullable=True),
        sa.Column('valid_to', sa.DateTime, nullable=True),
        sa.Column('is_current', sa.Boolean, default=True),
        
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('supersedes_id', UUID(as_uuid=True), nullable=True),
    )
    
    # Indexes
    op.create_index(
        'ix_semantic_facts_tenant_user_key',
        'semantic_facts',
        ['tenant_id', 'user_id', 'key', 'is_current']
    )
    op.create_index(
        'ix_semantic_facts_tenant_user_category',
        'semantic_facts',
        ['tenant_id', 'user_id', 'category', 'is_current']
    )
    op.create_index(
        'ix_semantic_facts_search',
        'semantic_facts',
        ['tenant_id', 'user_id'],
        postgresql_using='gin',
        postgresql_ops={'value': 'jsonb_path_ops'}
    )

def downgrade() -> None:
    op.drop_table('semantic_facts')
```

---

## Deliverables Checklist

- [ ] Neo4jGraphStore with all CRUD operations
- [ ] Personalized PageRank for multi-hop reasoning
- [ ] Graph schema initialization script
- [ ] SemanticFact and FactSchema models
- [ ] SemanticFactStore with versioning
- [ ] Temporal fact handling (valid_from/to)
- [ ] NeocorticalStore facade
- [ ] Graph-fact synchronization
- [ ] User profile generation
- [ ] Multi-hop query support
- [ ] Database migration for semantic_facts table
- [ ] Unit tests for graph operations
- [ ] Unit tests for fact store
- [ ] Integration tests for neocortical store
