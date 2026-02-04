# Phase 3: Hippocampal Store (Episodic Memory)

## Overview
**Duration**: Week 3-4  
**Goal**: Implement the fast-write episodic memory store with vector search, write gate, and entity/relation extraction.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                  From Working Memory                             │
│           (SemanticChunks with salience > threshold)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Write Gate                                  │
│   - Salience check (importance, novelty, stability)             │
│   - Risk assessment (PII, secrets)                               │
│   - Deduplication check                                          │
│   Decision: STORE / SKIP / ASYNC_STORE                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌───────────────────┐  ┌───────────────────┐
        │  Sync Write Path  │  │ Async Write Path  │
        │  (Fast, minimal)  │  │ (Full extraction) │
        └───────────────────┘  └───────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Hippocampal Store                               │
│   ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│   │   Vector Index      │  │   Dynamic KG Edges              │  │
│   │   (pgvector/HNSW)   │  │   (Entity associations)         │  │
│   └─────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 3.1: Write Gate Implementation

### Description
The write gate decides whether and how to store incoming information, preventing memory bloat.

### Subtask 3.1.1: Write Gate Decision Model

```python
# src/memory/hippocampal/write_gate.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Set
import re
from ..working.models import SemanticChunk, ChunkType
from ...core.enums import MemoryType

class WriteDecision(str, Enum):
    STORE_SYNC = "store_sync"      # Store immediately (fast path)
    STORE_ASYNC = "store_async"    # Queue for async processing
    SKIP = "skip"                   # Don't store
    REDACT_AND_STORE = "redact_and_store"  # Store after redaction

@dataclass
class WriteGateResult:
    decision: WriteDecision
    memory_types: List[MemoryType]    # Types to extract
    importance: float                  # Computed importance
    novelty: float                     # Computed novelty  
    risk_flags: List[str]             # Any risk concerns
    redaction_required: bool
    reason: str

@dataclass
class WriteGateConfig:
    # Salience thresholds
    min_importance: float = 0.3
    min_novelty: float = 0.2
    
    # Risk patterns
    pii_patterns: List[str] = None
    secret_patterns: List[str] = None
    
    # Sync vs async threshold
    sync_importance_threshold: float = 0.7
    
    def __post_init__(self):
        if self.pii_patterns is None:
            self.pii_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{16}\b',              # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            ]
        if self.secret_patterns is None:
            self.secret_patterns = [
                r'password\s*[:=]\s*\S+',
                r'api[_-]?key\s*[:=]\s*\S+',
                r'secret\s*[:=]\s*\S+',
                r'token\s*[:=]\s*\S+',
            ]


class WriteGate:
    """
    Decides whether to store information in long-term memory.
    
    Implements the filtering logic to prevent memory saturation
    while ensuring important information is captured.
    """
    
    def __init__(
        self, 
        config: Optional[WriteGateConfig] = None,
        known_facts_cache: Optional[Set[str]] = None
    ):
        self.config = config or WriteGateConfig()
        self._known_facts = known_facts_cache or set()
        
        # Compile regex patterns
        self._pii_patterns = [re.compile(p, re.I) for p in self.config.pii_patterns]
        self._secret_patterns = [re.compile(p, re.I) for p in self.config.secret_patterns]
    
    def evaluate(
        self,
        chunk: SemanticChunk,
        existing_memories: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> WriteGateResult:
        """
        Evaluate whether a chunk should be stored.
        
        Args:
            chunk: The semantic chunk to evaluate
            existing_memories: Recent relevant memories for novelty check
            context: Additional context (e.g., user preferences)
        
        Returns:
            WriteGateResult with decision and metadata
        """
        risk_flags = []
        redaction_required = False
        
        # 1. Check for PII/secrets
        pii_found = self._check_pii(chunk.text)
        secrets_found = self._check_secrets(chunk.text)
        
        if secrets_found:
            risk_flags.append("contains_secrets")
            return WriteGateResult(
                decision=WriteDecision.SKIP,
                memory_types=[],
                importance=chunk.salience,
                novelty=0.0,
                risk_flags=risk_flags,
                redaction_required=False,
                reason="Contains potential secrets - skipping"
            )
        
        if pii_found:
            risk_flags.append("contains_pii")
            redaction_required = True
        
        # 2. Compute importance
        importance = self._compute_importance(chunk, context)
        
        # 3. Compute novelty
        novelty = self._compute_novelty(chunk, existing_memories)
        
        # 4. Determine memory types to extract
        memory_types = self._determine_memory_types(chunk)
        
        # 5. Make decision
        combined_score = (importance * 0.6) + (novelty * 0.4)
        
        if combined_score < self.config.min_importance:
            return WriteGateResult(
                decision=WriteDecision.SKIP,
                memory_types=[],
                importance=importance,
                novelty=novelty,
                risk_flags=risk_flags,
                redaction_required=False,
                reason=f"Below importance threshold: {combined_score:.2f}"
            )
        
        # Sync for high-importance, async for medium
        if importance >= self.config.sync_importance_threshold:
            decision = WriteDecision.REDACT_AND_STORE if redaction_required else WriteDecision.STORE_SYNC
        else:
            decision = WriteDecision.STORE_ASYNC
        
        return WriteGateResult(
            decision=decision,
            memory_types=memory_types,
            importance=importance,
            novelty=novelty,
            risk_flags=risk_flags,
            redaction_required=redaction_required,
            reason=f"Score: {combined_score:.2f}, importance: {importance:.2f}, novelty: {novelty:.2f}"
        )
    
    def _check_pii(self, text: str) -> bool:
        """Check for PII patterns."""
        for pattern in self._pii_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _check_secrets(self, text: str) -> bool:
        """Check for secret patterns."""
        for pattern in self._secret_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _compute_importance(
        self, 
        chunk: SemanticChunk, 
        context: Optional[Dict] = None
    ) -> float:
        """
        Compute importance score based on multiple factors.
        """
        score = chunk.salience  # Start with chunker's salience
        
        # Boost for certain chunk types
        type_boosts = {
            ChunkType.PREFERENCE: 0.3,
            ChunkType.FACT: 0.2,
            ChunkType.INSTRUCTION: 0.1,
            ChunkType.EVENT: 0.1,
        }
        score += type_boosts.get(chunk.chunk_type, 0.0)
        
        # Boost for explicit markers in text
        text_lower = chunk.text.lower()
        if any(m in text_lower for m in ["always", "never", "important", "remember"]):
            score += 0.2
        if any(m in text_lower for m in ["my name", "i am", "i live", "i work"]):
            score += 0.15
        
        # Boost if contains named entities
        if chunk.entities:
            score += 0.1 * min(len(chunk.entities), 3)
        
        return min(score, 1.0)
    
    def _compute_novelty(
        self,
        chunk: SemanticChunk,
        existing_memories: Optional[List[Dict]] = None
    ) -> float:
        """
        Compute how novel this information is.
        """
        if not existing_memories:
            return 1.0  # Everything is novel if no existing memories
        
        # Check for duplicate or near-duplicate content
        chunk_text_lower = chunk.text.lower()
        
        for mem in existing_memories:
            mem_text = mem.get("text", "").lower()
            
            # Exact match
            if chunk_text_lower == mem_text:
                return 0.0
            
            # High overlap (simple word overlap)
            chunk_words = set(chunk_text_lower.split())
            mem_words = set(mem_text.split())
            
            if chunk_words and mem_words:
                overlap = len(chunk_words & mem_words) / len(chunk_words | mem_words)
                if overlap > 0.8:
                    return 0.2  # Low novelty
                if overlap > 0.5:
                    return 0.5  # Medium novelty
        
        return 1.0  # High novelty
    
    def _determine_memory_types(self, chunk: SemanticChunk) -> List[MemoryType]:
        """Map chunk type to memory types to extract."""
        mapping = {
            ChunkType.PREFERENCE: [MemoryType.PREFERENCE],
            ChunkType.FACT: [MemoryType.SEMANTIC_FACT, MemoryType.EPISODIC_EVENT],
            ChunkType.EVENT: [MemoryType.EPISODIC_EVENT],
            ChunkType.STATEMENT: [MemoryType.EPISODIC_EVENT],
            ChunkType.INSTRUCTION: [MemoryType.TASK_STATE],
            ChunkType.QUESTION: [MemoryType.EPISODIC_EVENT],
            ChunkType.OPINION: [MemoryType.HYPOTHESIS],
        }
        return mapping.get(chunk.chunk_type, [MemoryType.EPISODIC_EVENT])
    
    def add_known_fact(self, fact_key: str):
        """Add a fact key to known facts cache."""
        self._known_facts.add(fact_key)
```

### Subtask 3.1.2: PII Redaction Module

```python
# src/memory/hippocampal/redactor.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

@dataclass
class RedactionResult:
    original_text: str
    redacted_text: str
    redactions: List[Tuple[str, str, int, int]]  # (type, original, start, end)
    has_redactions: bool

class PIIRedactor:
    """
    Redacts PII from text before storage.
    """
    
    PATTERNS = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "CREDIT_CARD": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    def __init__(self, additional_patterns: Optional[dict] = None):
        self.patterns = {**self.PATTERNS}
        if additional_patterns:
            self.patterns.update(additional_patterns)
        
        self._compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
    
    def redact(self, text: str) -> RedactionResult:
        """
        Redact PII from text.
        
        Returns:
            RedactionResult with redacted text and metadata
        """
        redacted = text
        redactions = []
        
        # Track offset changes from redactions
        offset = 0
        
        for pii_type, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                original = match.group()
                replacement = f"[{pii_type}_REDACTED]"
                
                # Calculate position in redacted string
                start = match.start() + offset
                end = match.end() + offset
                
                redactions.append((pii_type, original, match.start(), match.end()))
                
                redacted = redacted[:start] + replacement + redacted[end:]
                offset += len(replacement) - len(original)
        
        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            redactions=redactions,
            has_redactions=len(redactions) > 0
        )
```

---

## Task 3.2: Embedding Service

### Description
Generate embeddings for memory content using configurable models.

### Subtask 3.2.1: Embedding Client

```python
# src/utils/embeddings.py
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from dataclasses import dataclass
import asyncio
import hashlib

@dataclass
class EmbeddingResult:
    embedding: List[float]
    model: str
    dimensions: int
    tokens_used: int

class EmbeddingClient(ABC):
    """Abstract embedding client."""
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        pass


class OpenAIEmbeddings(EmbeddingClient):
    """OpenAI embedding client."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536
    ):
        import openai
        from ..core.config import get_settings
        
        settings = get_settings()
        self.client = openai.AsyncOpenAI(
            api_key=api_key or settings.embedding.api_key
        )
        self.model = model
        self._dimensions = dimensions
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    async def embed(self, text: str) -> EmbeddingResult:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self._dimensions
        )
        
        return EmbeddingResult(
            embedding=response.data[0].embedding,
            model=self.model,
            dimensions=self._dimensions,
            tokens_used=response.usage.total_tokens
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self._dimensions
        )
        
        return [
            EmbeddingResult(
                embedding=item.embedding,
                model=self.model,
                dimensions=self._dimensions,
                tokens_used=response.usage.total_tokens // len(texts)
            )
            for item in response.data
        ]


class LocalEmbeddings(EmbeddingClient):
    """Local sentence-transformers embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dimensions = self.model.get_sentence_embedding_dimension()
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    async def embed(self, text: str) -> EmbeddingResult:
        # Run in thread pool since sentence-transformers is sync
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(text).tolist()
        )
        
        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            dimensions=self._dimensions,
            tokens_used=len(text.split())  # Approximate
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts).tolist()
        )
        
        return [
            EmbeddingResult(
                embedding=emb,
                model=self.model_name,
                dimensions=self._dimensions,
                tokens_used=len(text.split())
            )
            for emb, text in zip(embeddings, texts)
        ]


class CachedEmbeddings(EmbeddingClient):
    """Wrapper that caches embeddings in Redis."""
    
    def __init__(self, client: EmbeddingClient, redis_client, ttl_seconds: int = 86400):
        self.client = client
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    @property
    def dimensions(self) -> int:
        return self.client.dimensions
    
    def _cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
        return f"emb:{self.client.model}:{text_hash}"
    
    async def embed(self, text: str) -> EmbeddingResult:
        import json
        
        cache_key = self._cache_key(text)
        
        # Check cache
        cached = await self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return EmbeddingResult(**data)
        
        # Compute embedding
        result = await self.client.embed(text)
        
        # Cache result
        await self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps({
                "embedding": result.embedding,
                "model": result.model,
                "dimensions": result.dimensions,
                "tokens_used": result.tokens_used
            })
        )
        
        return result
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        # For batch, check each individually (could optimize with pipeline)
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            cached = await self.redis.get(cache_key)
            if cached:
                import json
                results.append((i, EmbeddingResult(**json.loads(cached))))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute uncached
        if uncached_texts:
            computed = await self.client.embed_batch(uncached_texts)
            for idx, result in zip(uncached_indices, computed):
                results.append((idx, result))
                # Cache
                cache_key = self._cache_key(texts[idx])
                import json
                await self.redis.setex(
                    cache_key,
                    self.ttl,
                    json.dumps({
                        "embedding": result.embedding,
                        "model": result.model,
                        "dimensions": result.dimensions,
                        "tokens_used": result.tokens_used
                    })
                )
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]
```

---

## Task 3.3: Entity and Relation Extraction

### Description
Extract structured entities and relations from text for knowledge graph building.

### Subtask 3.3.1: Entity Extractor

```python
# src/extraction/entity_extractor.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json
from ..core.schemas import EntityMention
from ..utils.llm import LLMClient

class EntityType(str, Enum):
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    PREFERENCE = "PREFERENCE"
    ATTRIBUTE = "ATTRIBUTE"

ENTITY_EXTRACTION_PROMPT = """Extract named entities from the following text.

Text: {text}

For each entity, provide:
1. The exact text as it appears
2. A normalized/canonical form
3. The entity type (PERSON, LOCATION, ORGANIZATION, DATE, TIME, MONEY, PRODUCT, EVENT, CONCEPT, PREFERENCE, ATTRIBUTE)

Return JSON array:
[
  {{"text": "Paris", "normalized": "Paris, France", "type": "LOCATION"}},
  {{"text": "next Monday", "normalized": "2026-02-09", "type": "DATE"}}
]

Extract ALL meaningful entities. Include:
- Named entities (people, places, organizations)
- Temporal expressions (dates, times)
- Preferences and attributes mentioned
- Key concepts

Return only the JSON array, no other text."""

class EntityExtractor:
    """
    Extracts named entities from text using LLM.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def extract(
        self,
        text: str,
        context: Optional[str] = None
    ) -> List[EntityMention]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract from
            context: Optional context for better extraction
        
        Returns:
            List of EntityMention objects
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        try:
            response = await self.llm.complete(
                prompt,
                temperature=0.0,
                max_tokens=500
            )
            
            entities_data = json.loads(response)
            
            return [
                EntityMention(
                    text=e.get("text", ""),
                    normalized=e.get("normalized", e.get("text", "")),
                    entity_type=e.get("type", "CONCEPT")
                )
                for e in entities_data
                if e.get("text")
            ]
        
        except (json.JSONDecodeError, KeyError):
            # Fallback: return empty list
            return []
    
    async def extract_batch(
        self,
        texts: List[str]
    ) -> List[List[EntityMention]]:
        """Extract entities from multiple texts."""
        import asyncio
        return await asyncio.gather(*[self.extract(t) for t in texts])


class SpacyEntityExtractor:
    """
    Fast entity extraction using spaCy (no LLM calls).
    Use for high-volume, lower-precision needs.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(model)
        
        self._type_mapping = {
            "PERSON": EntityType.PERSON,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "ORG": EntityType.ORGANIZATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
        }
    
    def extract(self, text: str) -> List[EntityMention]:
        """Synchronous extraction with spaCy."""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entity_type = self._type_mapping.get(ent.label_, EntityType.CONCEPT)
            
            entities.append(EntityMention(
                text=ent.text,
                normalized=ent.text,  # spaCy doesn't normalize
                entity_type=entity_type.value,
                start_char=ent.start_char,
                end_char=ent.end_char
            ))
        
        return entities
    
    async def extract_async(self, text: str) -> List[EntityMention]:
        """Async wrapper for sync extraction."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract, text)
```

### Subtask 3.3.2: Relation Extractor (OpenIE-style)

```python
# src/extraction/relation_extractor.py
from typing import List, Optional, Tuple
from dataclasses import dataclass
import json
from ..core.schemas import Relation
from ..utils.llm import LLMClient

RELATION_EXTRACTION_PROMPT = """Extract relationships from the following text using Open Information Extraction.

Text: {text}

For each relationship, identify:
1. Subject (who/what)
2. Predicate (the relationship/action)
3. Object (who/what is affected)

Return JSON array of triples:
[
  {{"subject": "John", "predicate": "lives_in", "object": "Paris", "confidence": 0.9}},
  {{"subject": "user", "predicate": "prefers", "object": "vegetarian food", "confidence": 0.85}}
]

Rules:
- Normalize predicates to snake_case (e.g., "is located in" -> "located_in")
- Include implicit relationships
- Assign confidence based on how explicit the relationship is
- Use "user" as subject for first-person statements

Return only the JSON array."""

class RelationExtractor:
    """
    Extracts relations (triples) from text using LLM.
    Implements Open Information Extraction style extraction.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def extract(
        self,
        text: str,
        entities: Optional[List[str]] = None
    ) -> List[Relation]:
        """
        Extract relations from text.
        
        Args:
            text: Text to extract from
            entities: Optional list of known entities for context
        
        Returns:
            List of Relation objects
        """
        prompt = RELATION_EXTRACTION_PROMPT.format(text=text)
        
        if entities:
            prompt += f"\n\nKnown entities: {', '.join(entities)}"
        
        try:
            response = await self.llm.complete(
                prompt,
                temperature=0.0,
                max_tokens=500
            )
            
            relations_data = json.loads(response)
            
            return [
                Relation(
                    subject=r.get("subject", ""),
                    predicate=self._normalize_predicate(r.get("predicate", "")),
                    object=r.get("object", ""),
                    confidence=float(r.get("confidence", 0.8))
                )
                for r in relations_data
                if r.get("subject") and r.get("predicate") and r.get("object")
            ]
        
        except (json.JSONDecodeError, KeyError):
            return []
    
    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate to snake_case."""
        import re
        # Replace spaces and hyphens with underscores
        normalized = re.sub(r'[\s\-]+', '_', predicate.lower())
        # Remove non-alphanumeric except underscores
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized
```

---

## Task 3.4: Hippocampal Store Implementation

### Description
The main episodic memory store with vector search and fast writes.

### Subtask 3.4.1: PostgreSQL Vector Store

```python
# src/storage/postgres.py
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, update, delete, func
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import Vector
from .models import MemoryRecordModel
from .base import MemoryStoreBase
from ..core.schemas import MemoryRecord, MemoryRecordCreate, Provenance
from ..core.enums import MemoryStatus

class PostgresMemoryStore(MemoryStoreBase):
    """
    PostgreSQL-based memory store with pgvector for embeddings.
    """
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
    
    async def upsert(self, record: MemoryRecordCreate) -> MemoryRecord:
        """Insert or update a memory record."""
        async with self.session_factory() as session:
            # Generate content hash for deduplication
            content_hash = self._hash_content(record.text, record.tenant_id, record.user_id)
            
            # Check for existing record with same hash
            existing = await session.execute(
                select(MemoryRecordModel).where(
                    and_(
                        MemoryRecordModel.content_hash == content_hash,
                        MemoryRecordModel.status == MemoryStatus.ACTIVE.value
                    )
                )
            )
            existing_record = existing.scalar_one_or_none()
            
            if existing_record:
                # Update existing
                existing_record.access_count += 1
                existing_record.last_accessed_at = datetime.utcnow()
                existing_record.confidence = max(existing_record.confidence, record.confidence)
                await session.commit()
                return self._to_schema(existing_record)
            
            # Create new
            model = MemoryRecordModel(
                tenant_id=record.tenant_id,
                user_id=record.user_id,
                agent_id=record.agent_id,
                type=record.type.value,
                text=record.text,
                key=record.key,
                embedding=record.embedding if hasattr(record, 'embedding') else None,
                entities=[e.model_dump() for e in record.entities],
                relations=[r.model_dump() for r in record.relations],
                metadata=record.metadata,
                timestamp=record.timestamp or datetime.utcnow(),
                written_at=datetime.utcnow(),
                confidence=record.confidence,
                importance=record.importance,
                provenance=record.provenance.model_dump(),
                content_hash=content_hash,
            )
            
            session.add(model)
            await session.commit()
            await session.refresh(model)
            
            return self._to_schema(model)
    
    async def get_by_id(self, record_id: UUID) -> Optional[MemoryRecord]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(MemoryRecordModel).where(MemoryRecordModel.id == record_id)
            )
            model = result.scalar_one_or_none()
            return self._to_schema(model) if model else None
    
    async def get_by_key(
        self,
        tenant_id: str,
        user_id: str,
        key: str
    ) -> Optional[MemoryRecord]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(MemoryRecordModel).where(
                    and_(
                        MemoryRecordModel.tenant_id == tenant_id,
                        MemoryRecordModel.user_id == user_id,
                        MemoryRecordModel.key == key,
                        MemoryRecordModel.status == MemoryStatus.ACTIVE.value
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._to_schema(model) if model else None
    
    async def delete(self, record_id: UUID, hard: bool = False) -> bool:
        async with self.session_factory() as session:
            if hard:
                result = await session.execute(
                    delete(MemoryRecordModel).where(MemoryRecordModel.id == record_id)
                )
            else:
                result = await session.execute(
                    update(MemoryRecordModel)
                    .where(MemoryRecordModel.id == record_id)
                    .values(status=MemoryStatus.DELETED.value)
                )
            await session.commit()
            return result.rowcount > 0
    
    async def update(
        self,
        record_id: UUID,
        patch: Dict[str, Any],
        increment_version: bool = True
    ) -> Optional[MemoryRecord]:
        async with self.session_factory() as session:
            # Get current record
            result = await session.execute(
                select(MemoryRecordModel).where(MemoryRecordModel.id == record_id)
            )
            model = result.scalar_one_or_none()
            
            if not model:
                return None
            
            # Apply patch
            for key, value in patch.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            if increment_version:
                model.version += 1
            
            await session.commit()
            await session.refresh(model)
            
            return self._to_schema(model)
    
    async def vector_search(
        self,
        tenant_id: str,
        user_id: str,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[MemoryRecord]:
        """Search by vector similarity using pgvector."""
        async with self.session_factory() as session:
            # Build base query with cosine distance
            query = select(
                MemoryRecordModel,
                (1 - MemoryRecordModel.embedding.cosine_distance(embedding)).label('similarity')
            ).where(
                and_(
                    MemoryRecordModel.tenant_id == tenant_id,
                    MemoryRecordModel.user_id == user_id,
                    MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                    MemoryRecordModel.embedding.isnot(None)
                )
            )
            
            # Apply filters
            if filters:
                if 'type' in filters:
                    query = query.where(MemoryRecordModel.type == filters['type'])
                if 'since' in filters:
                    query = query.where(MemoryRecordModel.timestamp >= filters['since'])
                if 'until' in filters:
                    query = query.where(MemoryRecordModel.timestamp <= filters['until'])
            
            # Order by similarity and limit
            query = query.order_by(
                MemoryRecordModel.embedding.cosine_distance(embedding)
            ).limit(top_k)
            
            result = await session.execute(query)
            
            records = []
            for row in result:
                model, similarity = row
                if similarity >= min_similarity:
                    record = self._to_schema(model)
                    record.metadata['_similarity'] = similarity
                    records.append(record)
            
            return records
    
    async def scan(
        self,
        tenant_id: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryRecord]:
        async with self.session_factory() as session:
            query = select(MemoryRecordModel).where(
                and_(
                    MemoryRecordModel.tenant_id == tenant_id,
                    MemoryRecordModel.user_id == user_id
                )
            )
            
            if filters:
                if 'status' in filters:
                    query = query.where(MemoryRecordModel.status == filters['status'])
                if 'type' in filters:
                    query = query.where(MemoryRecordModel.type == filters['type'])
            
            if order_by:
                col = getattr(MemoryRecordModel, order_by.lstrip('-'), None)
                if col:
                    query = query.order_by(col.desc() if order_by.startswith('-') else col)
            
            query = query.offset(offset).limit(limit)
            
            result = await session.execute(query)
            return [self._to_schema(m) for m in result.scalars().all()]
    
    async def count(
        self,
        tenant_id: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        async with self.session_factory() as session:
            query = select(func.count(MemoryRecordModel.id)).where(
                and_(
                    MemoryRecordModel.tenant_id == tenant_id,
                    MemoryRecordModel.user_id == user_id
                )
            )
            
            if filters and 'status' in filters:
                query = query.where(MemoryRecordModel.status == filters['status'])
            
            result = await session.execute(query)
            return result.scalar()
    
    def _hash_content(self, text: str, tenant_id: str, user_id: str) -> str:
        """Generate content hash for deduplication."""
        content = f"{tenant_id}:{user_id}:{text.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _to_schema(self, model: MemoryRecordModel) -> MemoryRecord:
        """Convert SQLAlchemy model to Pydantic schema."""
        from ..core.schemas import EntityMention, Relation
        
        return MemoryRecord(
            id=model.id,
            tenant_id=model.tenant_id,
            user_id=model.user_id,
            agent_id=model.agent_id,
            type=model.type,
            text=model.text,
            key=model.key,
            embedding=list(model.embedding) if model.embedding else None,
            entities=[EntityMention(**e) for e in (model.entities or [])],
            relations=[Relation(**r) for r in (model.relations or [])],
            metadata=model.metadata or {},
            timestamp=model.timestamp,
            written_at=model.written_at,
            valid_from=model.valid_from,
            valid_to=model.valid_to,
            confidence=model.confidence,
            importance=model.importance,
            access_count=model.access_count,
            last_accessed_at=model.last_accessed_at,
            decay_rate=model.decay_rate,
            status=model.status,
            labile=model.labile,
            provenance=Provenance(**model.provenance),
            version=model.version,
            supersedes_id=model.supersedes_id,
            content_hash=model.content_hash,
        )
```

### Subtask 3.4.2: Hippocampal Store Facade

```python
# src/memory/hippocampal/store.py
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from ..working.models import SemanticChunk
from ...core.schemas import MemoryRecord, MemoryRecordCreate, Provenance, EntityMention, Relation
from ...core.enums import MemoryType, MemoryStatus, MemorySource
from ...storage.postgres import PostgresMemoryStore
from ...utils.embeddings import EmbeddingClient
from ...extraction.entity_extractor import EntityExtractor
from ...extraction.relation_extractor import RelationExtractor
from .write_gate import WriteGate, WriteGateResult, WriteDecision
from .redactor import PIIRedactor

class HippocampalStore:
    """
    Fast episodic memory store - the "hippocampus" of the system.
    
    Responsibilities:
    1. Evaluate chunks through write gate
    2. Extract entities and relations
    3. Generate embeddings
    4. Store in vector database
    """
    
    def __init__(
        self,
        vector_store: PostgresMemoryStore,
        embedding_client: EmbeddingClient,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        write_gate: Optional[WriteGate] = None,
        redactor: Optional[PIIRedactor] = None
    ):
        self.store = vector_store
        self.embeddings = embedding_client
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.write_gate = write_gate or WriteGate()
        self.redactor = redactor or PIIRedactor()
    
    async def encode_chunk(
        self,
        tenant_id: str,
        user_id: str,
        chunk: SemanticChunk,
        agent_id: Optional[str] = None,
        existing_memories: Optional[List[Dict]] = None
    ) -> Optional[MemoryRecord]:
        """
        Encode a semantic chunk into episodic memory.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            chunk: Semantic chunk from working memory
            agent_id: Optional agent identifier
            existing_memories: Recent memories for novelty check
        
        Returns:
            Created MemoryRecord or None if skipped
        """
        # 1. Evaluate through write gate
        gate_result = self.write_gate.evaluate(
            chunk, 
            existing_memories=existing_memories
        )
        
        if gate_result.decision == WriteDecision.SKIP:
            return None
        
        # 2. Redact if needed
        text = chunk.text
        if gate_result.redaction_required:
            redaction_result = self.redactor.redact(text)
            text = redaction_result.redacted_text
        
        # 3. Generate embedding
        embedding_result = await self.embeddings.embed(text)
        
        # 4. Extract entities (if extractor available)
        entities = []
        if self.entity_extractor:
            entities = await self.entity_extractor.extract(text)
        elif chunk.entities:
            entities = [
                EntityMention(text=e, normalized=e, entity_type="CONCEPT")
                for e in chunk.entities
            ]
        
        # 5. Extract relations (if extractor available)
        relations = []
        if self.relation_extractor:
            entity_texts = [e.normalized for e in entities]
            relations = await self.relation_extractor.extract(text, entities=entity_texts)
        
        # 6. Determine memory type
        memory_type = gate_result.memory_types[0] if gate_result.memory_types else MemoryType.EPISODIC_EVENT
        
        # 7. Create memory record
        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            type=memory_type,
            text=text,
            key=self._generate_key(chunk, memory_type),
            entities=entities,
            relations=relations,
            metadata={
                "chunk_type": chunk.chunk_type.value,
                "source_turn_id": chunk.source_turn_id,
                "source_role": chunk.source_role,
            },
            timestamp=chunk.timestamp,
            confidence=chunk.confidence,
            importance=gate_result.importance,
            provenance=Provenance(
                source=MemorySource.AGENT_INFERRED,
                evidence_refs=[chunk.source_turn_id] if chunk.source_turn_id else [],
                model_version=embedding_result.model,
            )
        )
        
        # Add embedding to record
        record_dict = record.model_dump()
        record_dict['embedding'] = embedding_result.embedding
        
        # 8. Store
        stored = await self.store.upsert(MemoryRecordCreate(**record_dict))
        
        return stored
    
    async def encode_batch(
        self,
        tenant_id: str,
        user_id: str,
        chunks: List[SemanticChunk],
        agent_id: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Encode multiple chunks."""
        # Get existing memories for novelty check
        existing = await self.store.scan(
            tenant_id, user_id,
            filters={"status": MemoryStatus.ACTIVE.value},
            limit=50,
            order_by="-timestamp"
        )
        existing_dicts = [{"text": m.text} for m in existing]
        
        results = []
        for chunk in chunks:
            record = await self.encode_chunk(
                tenant_id, user_id, chunk, agent_id,
                existing_memories=existing_dicts
            )
            if record:
                results.append(record)
                existing_dicts.append({"text": record.text})
        
        return results
    
    async def search(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[MemoryRecord]:
        """Search episodic memories by semantic similarity."""
        # Embed query
        query_embedding = await self.embeddings.embed(query)
        
        # Search
        results = await self.store.vector_search(
            tenant_id, user_id,
            embedding=query_embedding.embedding,
            top_k=top_k,
            filters=filters
        )
        
        # Update access counts
        for record in results:
            await self.store.update(
                record.id,
                {
                    "access_count": record.access_count + 1,
                    "last_accessed_at": datetime.utcnow()
                },
                increment_version=False
            )
        
        return results
    
    async def get_recent(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[MemoryRecord]:
        """Get recent episodic memories."""
        filters = {"status": MemoryStatus.ACTIVE.value}
        if memory_types:
            filters["type"] = [t.value for t in memory_types]
        
        return await self.store.scan(
            tenant_id, user_id,
            filters=filters,
            order_by="-timestamp",
            limit=limit
        )
    
    def _generate_key(self, chunk: SemanticChunk, memory_type: MemoryType) -> Optional[str]:
        """Generate a key for keyed lookups (preferences, facts)."""
        if memory_type not in [MemoryType.PREFERENCE, MemoryType.SEMANTIC_FACT]:
            return None
        
        # Extract key from entities or chunk type
        if chunk.entities:
            return f"{memory_type.value}:{chunk.entities[0].lower()}"
        
        return None
```

---

## Deliverables Checklist

- [ ] WriteGate with salience, novelty, and risk evaluation
- [ ] WriteGateResult and WriteDecision models
- [ ] PIIRedactor for sensitive data handling
- [ ] EmbeddingClient abstraction (OpenAI + Local)
- [ ] CachedEmbeddings with Redis caching
- [ ] EntityExtractor (LLM-based and spaCy fallback)
- [ ] RelationExtractor for OpenIE-style triples
- [ ] PostgresMemoryStore with pgvector integration
- [ ] HippocampalStore facade coordinating all components
- [ ] Unit tests for write gate logic
- [ ] Unit tests for extraction
- [ ] Integration tests for full encode flow
