# Phase 2: Sensory Buffer & Working Memory

## Overview
**Duration**: Week 2-3  
**Goal**: Implement short-term memory systems that process raw input into semantically meaningful chunks before long-term encoding.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                      Incoming Turn                               │
│   {role: "user", content: "I just moved to Paris..."}           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Sensory Buffer                               │
│   - High fidelity storage of raw tokens                         │
│   - Decay after ~30 seconds                                      │
│   - Capacity: ~500 tokens                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Working Memory                               │
│   - Semantic chunking (sentences → facts)                        │
│   - Limited capacity (~10 chunks)                                │
│   - Actively manipulated for reasoning                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [To Hippocampal Store]
```

---

## Task 2.1: Sensory Buffer Implementation

### Description
A high-fidelity, short-lived buffer that temporarily holds raw input with timestamps.

### Subtask 2.1.1: Token-Level Buffer with Decay

```python
# src/memory/sensory/buffer.py
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Deque
import time
import asyncio
from threading import Lock

@dataclass
class BufferedToken:
    """A token with its ingestion timestamp."""
    token: str
    timestamp: float  # Unix timestamp for fast comparison
    turn_id: Optional[str] = None
    role: Optional[str] = None  # "user", "assistant", "system"

@dataclass
class SensoryBufferConfig:
    max_tokens: int = 500
    decay_seconds: float = 30.0
    cleanup_interval_seconds: float = 5.0

class SensoryBuffer:
    """
    High-fidelity short-term buffer for raw input tokens.
    
    Mimics sensory memory: stores everything briefly, then decays.
    Uses deque for O(1) append and popleft operations.
    """
    
    def __init__(self, config: Optional[SensoryBufferConfig] = None):
        self.config = config or SensoryBufferConfig()
        self._tokens: Deque[BufferedToken] = deque()
        self._lock = Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def ingest(
        self, 
        text: str, 
        turn_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> int:
        """
        Ingest new text into the buffer.
        
        Args:
            text: Raw text to buffer
            turn_id: Optional identifier for the conversation turn
            role: "user", "assistant", or "system"
        
        Returns:
            Number of tokens ingested
        """
        now = time.time()
        
        # Simple tokenization (could use tiktoken for accuracy)
        tokens = self._tokenize(text)
        
        with self._lock:
            # Add new tokens
            for token in tokens:
                self._tokens.append(BufferedToken(
                    token=token,
                    timestamp=now,
                    turn_id=turn_id,
                    role=role
                ))
            
            # Enforce capacity and decay
            self._cleanup(now)
        
        return len(tokens)
    
    def get_recent(
        self, 
        max_tokens: Optional[int] = None,
        since_seconds: Optional[float] = None,
        role_filter: Optional[str] = None
    ) -> List[BufferedToken]:
        """
        Get recent tokens from buffer.
        
        Args:
            max_tokens: Maximum tokens to return
            since_seconds: Only tokens from last N seconds
            role_filter: Filter by role ("user", "assistant")
        
        Returns:
            List of buffered tokens (oldest first)
        """
        now = time.time()
        cutoff = now - (since_seconds or self.config.decay_seconds)
        
        with self._lock:
            result = []
            for bt in self._tokens:
                if bt.timestamp < cutoff:
                    continue
                if role_filter and bt.role != role_filter:
                    continue
                result.append(bt)
                if max_tokens and len(result) >= max_tokens:
                    break
            
            return result
    
    def get_text(
        self,
        max_tokens: Optional[int] = None,
        role_filter: Optional[str] = None
    ) -> str:
        """Get buffered content as joined text."""
        tokens = self.get_recent(max_tokens=max_tokens, role_filter=role_filter)
        return " ".join(bt.token for bt in tokens)
    
    def clear(self):
        """Clear all buffered tokens."""
        with self._lock:
            self._tokens.clear()
    
    def _cleanup(self, now: float):
        """Remove expired tokens and enforce capacity."""
        cutoff = now - self.config.decay_seconds
        
        # Remove expired tokens (they're at the front since deque is ordered by time)
        while self._tokens and self._tokens[0].timestamp < cutoff:
            self._tokens.popleft()
        
        # Enforce capacity (remove oldest if over capacity)
        while len(self._tokens) > self.config.max_tokens:
            self._tokens.popleft()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenization.
        For production, use tiktoken for accurate token counts.
        """
        return text.split()
    
    @property
    def size(self) -> int:
        """Current number of tokens in buffer."""
        return len(self._tokens)
    
    @property
    def is_empty(self) -> bool:
        return len(self._tokens) == 0
    
    async def start_cleanup_loop(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                with self._lock:
                    self._cleanup(time.time())
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop_cleanup_loop(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
```

### Subtask 2.1.2: Per-User Sensory Buffer Manager

```python
# src/memory/sensory/manager.py
from typing import Dict, Optional
from .buffer import SensoryBuffer, SensoryBufferConfig
import asyncio

class SensoryBufferManager:
    """
    Manages per-user sensory buffers.
    Each user gets their own isolated buffer.
    """
    
    def __init__(self, config: Optional[SensoryBufferConfig] = None):
        self.config = config or SensoryBufferConfig()
        self._buffers: Dict[str, SensoryBuffer] = {}
        self._lock = asyncio.Lock()
    
    def _get_key(self, tenant_id: str, user_id: str) -> str:
        return f"{tenant_id}:{user_id}"
    
    async def get_buffer(self, tenant_id: str, user_id: str) -> SensoryBuffer:
        """Get or create buffer for user."""
        key = self._get_key(tenant_id, user_id)
        
        async with self._lock:
            if key not in self._buffers:
                self._buffers[key] = SensoryBuffer(self.config)
            return self._buffers[key]
    
    async def ingest(
        self,
        tenant_id: str,
        user_id: str,
        text: str,
        turn_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> int:
        """Ingest text into user's buffer."""
        buffer = await self.get_buffer(tenant_id, user_id)
        return buffer.ingest(text, turn_id, role)
    
    async def get_recent_text(
        self,
        tenant_id: str,
        user_id: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """Get recent text from user's buffer."""
        buffer = await self.get_buffer(tenant_id, user_id)
        return buffer.get_text(max_tokens=max_tokens)
    
    async def clear_user(self, tenant_id: str, user_id: str):
        """Clear a specific user's buffer."""
        key = self._get_key(tenant_id, user_id)
        async with self._lock:
            if key in self._buffers:
                self._buffers[key].clear()
    
    async def cleanup_inactive(self, inactive_seconds: float = 300):
        """Remove buffers that have been inactive."""
        import time
        now = time.time()
        
        async with self._lock:
            to_remove = []
            for key, buffer in self._buffers.items():
                if buffer.is_empty:
                    to_remove.append(key)
            
            for key in to_remove:
                del self._buffers[key]
```

---

## Task 2.2: Working Memory with Semantic Chunking

### Description
Process sensory buffer contents into semantically meaningful chunks that can be encoded into long-term memory.

### Subtask 2.2.1: Chunk Data Structures

```python
# src/memory/working/models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class ChunkType(str, Enum):
    STATEMENT = "statement"      # Declarative statement
    PREFERENCE = "preference"    # User preference
    QUESTION = "question"        # Question asked
    INSTRUCTION = "instruction"  # Command/request
    FACT = "fact"               # Factual claim
    EVENT = "event"             # Something that happened
    OPINION = "opinion"         # Subjective view

@dataclass
class SemanticChunk:
    """A semantically coherent unit of information."""
    id: str
    text: str
    chunk_type: ChunkType
    
    # Source tracking
    source_turn_id: Optional[str] = None
    source_role: Optional[str] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    
    # Extracted elements
    entities: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    
    # Scores
    salience: float = 0.5      # How important is this?
    novelty: float = 0.5       # How new is this info?
    confidence: float = 1.0    # How certain are we of extraction?
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkingMemoryState:
    """Current state of working memory for a user."""
    tenant_id: str
    user_id: str
    
    # Active chunks (limited capacity)
    chunks: List[SemanticChunk] = field(default_factory=list)
    max_chunks: int = 10
    
    # Current focus
    current_topic: Optional[str] = None
    current_intent: Optional[str] = None
    
    # Conversation context
    turn_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def add_chunk(self, chunk: SemanticChunk):
        """Add chunk, evicting oldest if at capacity."""
        self.chunks.append(chunk)
        if len(self.chunks) > self.max_chunks:
            # Evict lowest salience chunk
            self.chunks.sort(key=lambda c: c.salience, reverse=True)
            self.chunks = self.chunks[:self.max_chunks]
        self.last_updated = datetime.utcnow()
    
    def get_high_salience_chunks(self, min_salience: float = 0.5) -> List[SemanticChunk]:
        """Get chunks above salience threshold."""
        return [c for c in self.chunks if c.salience >= min_salience]
```

### Subtask 2.2.2: Semantic Chunker (LLM-based)

```python
# src/memory/working/chunker.py
from typing import List, Optional, Dict, Any
import json
import hashlib
from .models import SemanticChunk, ChunkType
from ...utils.llm import LLMClient

CHUNKING_PROMPT = """Analyze the following text and extract semantically meaningful chunks.

For each chunk, identify:
1. The core statement or fact
2. The type (statement, preference, question, instruction, fact, event, opinion)
3. Key entities mentioned
4. Importance score (0.0-1.0) based on:
   - Explicit user preferences (high)
   - Personal information (high)
   - Task-relevant details (medium-high)
   - General conversation (low)

Text to analyze:
{text}

Context (previous chunks):
{context}

Return a JSON array of chunks:
[
  {
    "text": "extracted chunk text",
    "type": "preference|statement|fact|event|question|instruction|opinion",
    "entities": ["entity1", "entity2"],
    "key_phrases": ["phrase1"],
    "salience": 0.8,
    "confidence": 0.9
  }
]

Rules:
- Each chunk should be a single coherent idea
- Preserve the user's wording for preferences and facts
- Don't create chunks for filler/acknowledgments
- Combine related short statements if they form one idea"""

class SemanticChunker:
    """
    Uses an LLM to break text into semantic chunks.
    Implements chunking similar to human working memory processing.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def chunk(
        self,
        text: str,
        context_chunks: Optional[List[SemanticChunk]] = None,
        turn_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> List[SemanticChunk]:
        """
        Break text into semantic chunks.
        
        Args:
            text: Raw text to chunk
            context_chunks: Recent chunks for context
            turn_id: Source turn identifier
            role: "user" or "assistant"
        
        Returns:
            List of semantic chunks
        """
        if not text.strip():
            return []
        
        # Build context string from recent chunks
        context_str = ""
        if context_chunks:
            context_str = "\n".join([
                f"- [{c.chunk_type.value}] {c.text}"
                for c in context_chunks[-5:]  # Last 5 chunks
            ])
        
        prompt = CHUNKING_PROMPT.format(
            text=text,
            context=context_str or "None"
        )
        
        try:
            response = await self.llm.complete(
                prompt,
                temperature=0.0,
                max_tokens=1000
            )
            
            chunks_data = json.loads(response)
            
            chunks = []
            for i, cd in enumerate(chunks_data):
                chunk_id = self._generate_chunk_id(text, i)
                
                chunks.append(SemanticChunk(
                    id=chunk_id,
                    text=cd.get("text", ""),
                    chunk_type=ChunkType(cd.get("type", "statement")),
                    source_turn_id=turn_id,
                    source_role=role,
                    entities=cd.get("entities", []),
                    key_phrases=cd.get("key_phrases", []),
                    salience=float(cd.get("salience", 0.5)),
                    confidence=float(cd.get("confidence", 0.8)),
                ))
            
            return chunks
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: treat entire text as single chunk
            return [SemanticChunk(
                id=self._generate_chunk_id(text, 0),
                text=text,
                chunk_type=ChunkType.STATEMENT,
                source_turn_id=turn_id,
                source_role=role,
                salience=0.5,
                confidence=0.5,
            )]
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate deterministic chunk ID."""
        content = f"{text}:{index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class RuleBasedChunker:
    """
    Fast, rule-based chunker for when LLM is too slow.
    Uses heuristics to split text into chunks.
    """
    
    # Patterns indicating high salience
    PREFERENCE_MARKERS = ["i prefer", "i like", "i love", "i hate", "i don't like", "i want"]
    FACT_MARKERS = ["my name is", "i am", "i live", "i work", "i have"]
    INSTRUCTION_MARKERS = ["please", "can you", "could you", "i need", "help me"]
    
    def chunk(
        self,
        text: str,
        turn_id: Optional[str] = None,
        role: Optional[str] = None
    ) -> List[SemanticChunk]:
        """Rule-based chunking."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i, sentence in enumerate(sentences):
            lower = sentence.lower()
            
            # Determine type and salience
            chunk_type = ChunkType.STATEMENT
            salience = 0.3
            
            if any(marker in lower for marker in self.PREFERENCE_MARKERS):
                chunk_type = ChunkType.PREFERENCE
                salience = 0.8
            elif any(marker in lower for marker in self.FACT_MARKERS):
                chunk_type = ChunkType.FACT
                salience = 0.7
            elif any(marker in lower for marker in self.INSTRUCTION_MARKERS):
                chunk_type = ChunkType.INSTRUCTION
                salience = 0.6
            elif "?" in sentence:
                chunk_type = ChunkType.QUESTION
                salience = 0.4
            
            chunks.append(SemanticChunk(
                id=f"rule_{hash(sentence) % 10000}_{i}",
                text=sentence,
                chunk_type=chunk_type,
                source_turn_id=turn_id,
                source_role=role,
                salience=salience,
                confidence=0.6,  # Lower confidence for rule-based
            ))
        
        return chunks
```

### Subtask 2.2.3: Working Memory Manager

```python
# src/memory/working/manager.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from .models import WorkingMemoryState, SemanticChunk
from .chunker import SemanticChunker, RuleBasedChunker
from ..sensory.buffer import SensoryBuffer
from ...utils.llm import LLMClient

class WorkingMemoryManager:
    """
    Manages working memory states per user.
    
    Responsibilities:
    1. Process sensory buffer into chunks
    2. Maintain limited-capacity working memory
    3. Decide what needs long-term encoding
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_chunks_per_user: int = 10,
        use_fast_chunker: bool = False
    ):
        self._states: Dict[str, WorkingMemoryState] = {}
        self._lock = asyncio.Lock()
        self.max_chunks = max_chunks_per_user
        
        # Use LLM chunker if available, otherwise rule-based
        if llm_client and not use_fast_chunker:
            self.chunker = SemanticChunker(llm_client)
            self._use_llm = True
        else:
            self.chunker = RuleBasedChunker()
            self._use_llm = False
    
    def _get_key(self, tenant_id: str, user_id: str) -> str:
        return f"{tenant_id}:{user_id}"
    
    async def get_state(self, tenant_id: str, user_id: str) -> WorkingMemoryState:
        """Get or create working memory state for user."""
        key = self._get_key(tenant_id, user_id)
        
        async with self._lock:
            if key not in self._states:
                self._states[key] = WorkingMemoryState(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    max_chunks=self.max_chunks
                )
            return self._states[key]
    
    async def process_input(
        self,
        tenant_id: str,
        user_id: str,
        text: str,
        turn_id: Optional[str] = None,
        role: str = "user"
    ) -> List[SemanticChunk]:
        """
        Process new input into working memory.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            text: Input text to process
            turn_id: Conversation turn ID
            role: "user" or "assistant"
        
        Returns:
            New chunks added to working memory
        """
        state = await self.get_state(tenant_id, user_id)
        
        # Get existing chunks for context
        context = state.chunks[-5:] if state.chunks else None
        
        # Chunk the input
        if self._use_llm:
            new_chunks = await self.chunker.chunk(
                text, 
                context_chunks=context,
                turn_id=turn_id,
                role=role
            )
        else:
            new_chunks = self.chunker.chunk(text, turn_id=turn_id, role=role)
        
        # Add to working memory
        for chunk in new_chunks:
            state.add_chunk(chunk)
        
        state.turn_count += 1
        
        return new_chunks
    
    async def get_chunks_for_encoding(
        self,
        tenant_id: str,
        user_id: str,
        min_salience: float = 0.4
    ) -> List[SemanticChunk]:
        """
        Get chunks that should be encoded into long-term memory.
        
        Only returns chunks above salience threshold.
        """
        state = await self.get_state(tenant_id, user_id)
        return [c for c in state.chunks if c.salience >= min_salience]
    
    async def get_current_context(
        self,
        tenant_id: str,
        user_id: str,
        max_chunks: int = 5
    ) -> str:
        """
        Get formatted current context for LLM prompts.
        """
        state = await self.get_state(tenant_id, user_id)
        
        # Sort by recency, take top N
        recent = sorted(
            state.chunks,
            key=lambda c: c.timestamp,
            reverse=True
        )[:max_chunks]
        
        lines = []
        for chunk in reversed(recent):  # Chronological order
            lines.append(f"- [{chunk.chunk_type.value}] {chunk.text}")
        
        return "\n".join(lines)
    
    async def clear_user(self, tenant_id: str, user_id: str):
        """Clear working memory for user."""
        key = self._get_key(tenant_id, user_id)
        async with self._lock:
            if key in self._states:
                del self._states[key]
    
    async def get_stats(self, tenant_id: str, user_id: str) -> Dict:
        """Get working memory statistics."""
        state = await self.get_state(tenant_id, user_id)
        
        return {
            "chunk_count": len(state.chunks),
            "max_chunks": state.max_chunks,
            "turn_count": state.turn_count,
            "current_topic": state.current_topic,
            "last_updated": state.last_updated.isoformat(),
            "avg_salience": sum(c.salience for c in state.chunks) / len(state.chunks) if state.chunks else 0,
        }
```

---

## Task 2.3: Integration with Memory Pipeline

### Description
Connect sensory buffer and working memory to the main memory orchestration flow.

### Subtask 2.3.1: Short-Term Memory Facade

```python
# src/memory/short_term.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from .sensory.buffer import SensoryBuffer, SensoryBufferConfig
from .sensory.manager import SensoryBufferManager
from .working.manager import WorkingMemoryManager
from .working.models import SemanticChunk
from ..utils.llm import LLMClient

@dataclass
class ShortTermMemoryConfig:
    # Sensory buffer settings
    sensory_max_tokens: int = 500
    sensory_decay_seconds: float = 30.0
    
    # Working memory settings
    working_max_chunks: int = 10
    use_fast_chunker: bool = False
    min_salience_for_encoding: float = 0.4

class ShortTermMemory:
    """
    Unified interface for sensory buffer + working memory.
    
    This is the entry point for all new information before
    it gets encoded into long-term memory.
    """
    
    def __init__(
        self,
        config: Optional[ShortTermMemoryConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.config = config or ShortTermMemoryConfig()
        
        # Initialize components
        sensory_config = SensoryBufferConfig(
            max_tokens=self.config.sensory_max_tokens,
            decay_seconds=self.config.sensory_decay_seconds
        )
        
        self.sensory = SensoryBufferManager(sensory_config)
        self.working = WorkingMemoryManager(
            llm_client=llm_client,
            max_chunks_per_user=self.config.working_max_chunks,
            use_fast_chunker=self.config.use_fast_chunker
        )
    
    async def ingest_turn(
        self,
        tenant_id: str,
        user_id: str,
        text: str,
        turn_id: Optional[str] = None,
        role: str = "user"
    ) -> Dict[str, Any]:
        """
        Ingest a new conversation turn.
        
        Flow:
        1. Add to sensory buffer (immediate)
        2. Process into working memory chunks
        3. Return chunks ready for potential encoding
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier  
            text: The turn content
            turn_id: Unique turn identifier
            role: "user" or "assistant"
        
        Returns:
            Dict with processing results
        """
        # 1. Sensory buffer (fast, synchronous)
        tokens_added = await self.sensory.ingest(
            tenant_id, user_id, text, turn_id, role
        )
        
        # 2. Working memory chunking
        new_chunks = await self.working.process_input(
            tenant_id, user_id, text, turn_id, role
        )
        
        # 3. Identify chunks for encoding
        chunks_for_encoding = [
            c for c in new_chunks 
            if c.salience >= self.config.min_salience_for_encoding
        ]
        
        return {
            "tokens_buffered": tokens_added,
            "chunks_created": len(new_chunks),
            "chunks_for_encoding": chunks_for_encoding,
            "all_chunks": new_chunks
        }
    
    async def get_immediate_context(
        self,
        tenant_id: str,
        user_id: str,
        include_sensory: bool = True,
        max_working_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Get immediate context for the current conversation.
        
        Useful for providing context to the LLM without
        hitting long-term memory.
        """
        result = {
            "working_memory": await self.working.get_current_context(
                tenant_id, user_id, max_working_chunks
            )
        }
        
        if include_sensory:
            result["recent_text"] = await self.sensory.get_recent_text(
                tenant_id, user_id, max_tokens=200
            )
        
        return result
    
    async def get_encodable_chunks(
        self,
        tenant_id: str,
        user_id: str
    ) -> List[SemanticChunk]:
        """
        Get all chunks that should be encoded into long-term memory.
        """
        return await self.working.get_chunks_for_encoding(
            tenant_id, user_id,
            min_salience=self.config.min_salience_for_encoding
        )
    
    async def clear(self, tenant_id: str, user_id: str):
        """Clear all short-term memory for user."""
        await self.sensory.clear_user(tenant_id, user_id)
        await self.working.clear_user(tenant_id, user_id)
```

---

## Task 2.4: LLM Utility Module

### Description
Implement the LLM client used by the chunker and other components.

### Subtask 2.4.1: LLM Client Abstraction

```python
# src/utils/llm.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import asyncio
from dataclasses import dataclass
import openai
from ..core.config import get_settings

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    finish_reason: str

class LLMClient(ABC):
    """Abstract LLM client interface."""
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None
    ) -> str:
        pass
    
    @abstractmethod
    async def complete_json(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.client = openai.AsyncOpenAI(
            api_key=api_key or settings.llm.api_key
        )
        self.model = model or settings.llm.model
    
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def complete_json(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        import json
        
        response = await self.complete(
            prompt,
            temperature=temperature,
            system_prompt="You are a JSON generator. Always respond with valid JSON only, no markdown."
        )
        
        # Try to parse JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r'\[.*\]|\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise


def get_llm_client() -> LLMClient:
    """Factory function to get configured LLM client."""
    settings = get_settings()
    
    if settings.llm.provider == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")
```

---

## Deliverables Checklist

- [x] SensoryBuffer class with token-level storage and decay
- [x] SensoryBufferManager for per-user buffer management
- [x] SemanticChunk and WorkingMemoryState data models
- [x] SemanticChunker (LLM-based) for intelligent chunking
- [x] RuleBasedChunker for fast fallback
- [x] WorkingMemoryManager with capacity limits
- [x] ShortTermMemory facade unifying both components
- [x] LLMClient abstraction with OpenAI implementation
- [x] Unit tests for buffer operations
- [x] Unit tests for chunking logic
- [x] Integration test: full ingest flow
