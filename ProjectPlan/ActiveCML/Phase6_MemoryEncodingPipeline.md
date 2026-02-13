# Phase 6: Memory Encoding Pipeline — Hippocampal Encoder Redesign

**Intrinsic Phase I-6** (planned; not yet implemented). See [BaseCMLStatus.md](../BaseCML/BaseCMLStatus.md) for the mapping of I-1..I-10 to core CML phases.

## Overview

Phases 2-4 introduced the injection interfaces (Logit, Activation, KV-Cache), each requiring memories in a specific latent format. Phase 6 builds the **unified encoding pipeline** — the **Hippocampal Encoder** — that transforms raw text memories into all required latent representations in a single pass.

### Core Problem
Currently, each interface derives its own representations independently:
- Logit Interface: extracts entities → maps to tokens
- Activation Interface: runs contrastive forward passes → derives steering vectors  
- Synaptic Interface: runs full forward pass → captures KV pairs

This is redundant and expensive. The Hippocampal Encoder consolidates encoding into a single pipeline that produces **all** required representations from one memory record.

### Biological Analogy
In the human brain, the hippocampus encodes new experiences into **multiple representations simultaneously**:
- Pattern-separated episodic traces (activation space)
- Semantic associations (knowledge graph connections)
- Procedural patterns (motor cortex pathways)

Our Hippocampal Encoder mirrors this: one input, multiple output representations.

### Dependencies
- Phase 1: Model Backend, Model Inspector
- Phase 2: Token-Memory Mapper
- Phase 3: Steering Vector derivation (CDD, Identity V, PCA)
- Phase 4: KV Encoder
- Phase 5: Controller (determines which representations are needed)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Hippocampal Encoder (Unified Pipeline)                   │
│                                                                              │
│  Input: MemoryRecord (text + type + metadata)                                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Stage 1: Text Analysis                               │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  - Entity extraction (NER)                                           │    │
│  │  - Key concept identification                                        │    │
│  │  - Sentiment / valence analysis                                      │    │
│  │  - Memory type classification                                        │    │
│  │  Output: AnalyzedMemory (entities, concepts, valence, type)          │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Stage 2: Embedding Generation                        │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  - Standard embedding (existing embedding client: text-embedding-3)  │    │
│  │  - Contextual embedding (model's own encoder, if local)              │    │
│  │  Output: embedding vector [embed_dim]                                │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Stage 3: Latent Representation Generation            │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │                                                                      │    │
│  │  ┌─────────────┐  ┌────────────────┐  ┌────────────────────────┐   │    │
│  │  │ Token Map    │  │ Steering Vector│  │ KV Pairs               │   │    │
│  │  │ (Logit)     │  │ (Activation)   │  │ (Synaptic)             │   │    │
│  │  │             │  │                │  │                        │   │    │
│  │  │ entities →  │  │ Identity V or  │  │ Forward pass →         │   │    │
│  │  │ token_ids → │  │ CDD or         │  │ capture K,V at         │   │    │
│  │  │ bias values │  │ Projection Head│  │ target layers          │   │    │
│  │  └─────────────┘  └────────────────┘  └────────────────────────┘   │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  Output: EncodedMemory                                                       │
│    - memory_id                                                               │
│    - embedding: [embed_dim]                                                  │
│    - token_biases: {token_id: bias_value}                                   │
│    - steering_vector: [hidden_dim] per target layer                         │
│    - kv_pairs: {layer_idx: (K, V)} per target layer                        │
│    - metadata: analysis results                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Unified Encoded Memory Data Structure

### Description
Define the canonical data structure that holds all representations of a single memory.

### Sub-task 1.1: `EncodedMemory` Data Class

**File:** `src/intrinsic/encoding/hippocampal_encoder.py`

```python
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import time

logger = logging.getLogger(__name__)


@dataclass
class MemoryAnalysis:
    """Results of text analysis stage."""
    entities: List[str]                  # Named entities
    key_concepts: List[str]              # Key concepts/keywords
    valence: float                       # Sentiment (-1 to 1)
    urgency: float                       # How urgently relevant (0 to 1)
    memory_type_suggested: str           # Type suggested by analysis
    complexity: float                    # How complex is this memory (0 to 1)


@dataclass
class EncodedMemory:
    """
    Complete multi-representation encoding of a memory.
    This is the output of the Hippocampal Encoder.
    """
    memory_id: str
    source_text: str
    memory_type: str
    
    # Analysis
    analysis: Optional[MemoryAnalysis] = None
    
    # Standard embedding (for vector search / retrieval)
    embedding: Optional[torch.Tensor] = None       # [embed_dim]
    
    # Logit representation
    token_biases: Optional[Dict[int, float]] = None
    key_token_ids: Optional[List[int]] = None
    
    # Activation representation (per layer)
    steering_vectors: Optional[Dict[int, torch.Tensor]] = None  # {layer_idx: [hidden_dim]}
    
    # KV-Cache representation (per layer)
    kv_pairs: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    num_virtual_tokens: int = 0
    
    # Metadata
    encoded_at: float = field(default_factory=time.time)
    encoding_method: str = "full"        # "full" | "fast" | "logit_only"
    encoding_time_ms: float = 0.0
    
    def has_logit_representation(self) -> bool:
        return self.token_biases is not None and len(self.token_biases) > 0
    
    def has_activation_representation(self) -> bool:
        return self.steering_vectors is not None and len(self.steering_vectors) > 0
    
    def has_kv_representation(self) -> bool:
        return self.kv_pairs is not None and len(self.kv_pairs) > 0
    
    def to_memory_vector(self, target_layers: Optional[List[int]] = None):
        """Convert to MemoryVector for the bus."""
        from ..bus import MemoryVector
        
        # Pick the first available steering vector
        sv = None
        if self.steering_vectors:
            first_layer = list(self.steering_vectors.keys())[0]
            sv = self.steering_vectors[first_layer]
        
        return MemoryVector(
            memory_id=self.memory_id,
            source_text=self.source_text,
            steering_vector=sv,
            target_layers=target_layers or list((self.steering_vectors or {}).keys()),
            logit_bias=self.token_biases,
            key_vectors=None,  # Set separately for KV injection
            value_vectors=None,
        )
```

---

## Task 2: Hippocampal Encoder — Main Pipeline

### Description
The unified encoder that runs all three representation stages.

### Sub-task 2.1: `HippocampalEncoder`

**File:** `src/intrinsic/encoding/hippocampal_encoder.py` (continued)

```python
class HippocampalEncoder:
    """
    Unified memory encoder: text → multi-representation latent vectors.
    
    Encoding modes:
    - "full": All representations (token, steering, KV). Expensive.
    - "fast": Only token mapping + Identity V steering. Cheap.
    - "logit_only": Only token mapping. Cheapest.
    - "adaptive": Choose mode based on memory type and backend capabilities.
    
    The encoder reuses the model's forward pass: when computing KV pairs,
    it also captures hidden states for steering vector derivation,
    and extracts token mappings from the analysis stage.
    """

    def __init__(
        self,
        backend,                   # ModelBackend
        tokenizer,
        llm_client=None,          # For text analysis (existing LLM client)
        embedding_client=None,     # For standard embeddings
        target_layers: Optional[List[int]] = None,
        default_mode: str = "adaptive",
    ):
        self._backend = backend
        self._tokenizer = tokenizer
        self._llm_client = llm_client
        self._embedding_client = embedding_client
        self._default_mode = default_mode
        
        # Import sub-encoders
        from .steering_vectors import IdentityVDerivation, ContrastiveDirectionDiscovery
        from .kv_encoder import KVEncoder
        from ..interfaces.logit import TokenMemoryMapper
        
        self._identity_v = IdentityVDerivation(backend, tokenizer)
        self._cdd = ContrastiveDirectionDiscovery(backend, tokenizer)
        self._kv_encoder = KVEncoder(backend, tokenizer, target_layers=target_layers)
        self._token_mapper = TokenMemoryMapper(tokenizer, llm_client)
        
        # Target layers for steering vectors
        if target_layers is None:
            from ..inspector import ModelInspector
            inspector = ModelInspector(backend)
            self._target_layers = inspector.recommended_injection_layers()
        else:
            self._target_layers = target_layers

    async def encode(
        self,
        memory_id: str,
        memory_text: str,
        memory_type: str = "semantic_fact",
        mode: Optional[str] = None,
        relevance: float = 1.0,
        confidence: float = 1.0,
    ) -> EncodedMemory:
        """
        Encode a memory into all required latent representations.
        
        Returns: EncodedMemory with all available representations.
        """
        start_time = time.time()
        mode = mode or self._resolve_mode(memory_type)
        
        encoded = EncodedMemory(
            memory_id=memory_id,
            source_text=memory_text,
            memory_type=memory_type,
            encoding_method=mode,
        )

        # Stage 1: Text Analysis
        encoded.analysis = await self._analyze_text(memory_text, memory_type)

        # Stage 2: Standard Embedding
        if self._embedding_client:
            try:
                embedding = await self._embedding_client.embed(memory_text)
                encoded.embedding = torch.tensor(embedding)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        # Stage 3: Latent Representations (based on mode)
        if mode in ("full", "fast", "logit_only", "adaptive"):
            # Always generate token mapping (cheap)
            try:
                token_map = await self._token_mapper.map_memory(
                    memory_id=memory_id,
                    memory_text=memory_text,
                    relevance=relevance,
                    confidence=confidence,
                    memory_type=memory_type,
                )
                encoded.token_biases = {
                    tb.token_id: tb.bias_value for tb in token_map.token_biases
                }
                encoded.key_token_ids = [tb.token_id for tb in token_map.token_biases]
            except Exception as e:
                logger.warning(f"Token mapping failed: {e}")

        if mode in ("full", "fast", "adaptive"):
            # Generate steering vectors
            try:
                encoded.steering_vectors = await self._derive_steering_vectors(
                    memory_text, memory_type, mode
                )
            except Exception as e:
                logger.warning(f"Steering vector derivation failed: {e}")

        if mode == "full":
            # Generate KV pairs (most expensive)
            try:
                kv_result = await self._kv_encoder.encode(
                    memory_id=memory_id,
                    memory_text=memory_text,
                    relevance=relevance,
                )
                encoded.kv_pairs = kv_result.kv_pairs
                encoded.num_virtual_tokens = kv_result.num_virtual_tokens
            except Exception as e:
                logger.warning(f"KV encoding failed: {e}")

        encoded.encoding_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Encoded memory {memory_id} ({mode}): "
            f"logit={'Y' if encoded.has_logit_representation() else 'N'}, "
            f"activation={'Y' if encoded.has_activation_representation() else 'N'}, "
            f"kv={'Y' if encoded.has_kv_representation() else 'N'} "
            f"[{encoded.encoding_time_ms:.1f}ms]"
        )

        return encoded

    async def _analyze_text(self, text: str, memory_type: str) -> MemoryAnalysis:
        """Run text analysis to extract entities, concepts, and metadata."""
        entities = []
        concepts = []
        
        if self._llm_client:
            try:
                prompt = (
                    f"Analyze this memory and extract:\n"
                    f"1. Named entities (people, places, things)\n"
                    f"2. Key concepts (3-5 words)\n"
                    f"3. Sentiment (-1 to 1)\n"
                    f"4. Urgency (0 to 1)\n\n"
                    f"Memory: \"{text}\"\n\n"
                    f"Return JSON: {{\"entities\": [], \"concepts\": [], \"sentiment\": 0.0, \"urgency\": 0.0}}"
                )
                result = await self._llm_client.complete_json(prompt)
                entities = result.get("entities", [])
                concepts = result.get("concepts", [])
                valence = result.get("sentiment", 0.0)
                urgency = result.get("urgency", 0.0)
            except Exception:
                pass
        
        if not concepts:
            # Fallback: simple keyword extraction
            words = text.split()
            concepts = [w for w in words if len(w) > 4][:5]
        
        return MemoryAnalysis(
            entities=entities,
            key_concepts=concepts,
            valence=valence if 'valence' in dir() else 0.0,
            urgency=urgency if 'urgency' in dir() else 0.5,
            memory_type_suggested=memory_type,
            complexity=min(1.0, len(text.split()) / 50),
        )

    async def _derive_steering_vectors(
        self, text: str, memory_type: str, mode: str
    ) -> Dict[int, torch.Tensor]:
        """Derive steering vectors for target layers."""
        vectors = {}
        
        if mode == "fast":
            # Fast mode: use Identity V (no forward passes)
            try:
                # Extract first 3 words as concept tokens
                concept_tokens = text.split()[:3]
                for layer_idx in self._target_layers:
                    sv = self._identity_v.derive(concept_tokens, layer_idx)
                    vectors[layer_idx] = sv.vector
            except Exception as e:
                logger.warning(f"Identity V derivation failed: {e}")
        
        elif mode in ("full", "adaptive"):
            # Full mode: use CDD (requires forward passes)
            try:
                # Use the first target layer for CDD (reuse for others)
                primary_layer = self._target_layers[len(self._target_layers) // 2]
                sv = await self._cdd.derive_from_memory(text, memory_type, primary_layer)
                
                # Apply same vector to all target layers (scaled)
                for layer_idx in self._target_layers:
                    vectors[layer_idx] = sv.vector.clone()
            except Exception as e:
                logger.warning(f"CDD derivation failed, falling back to Identity V: {e}")
                # Fallback to Identity V
                return await self._derive_steering_vectors(text, memory_type, "fast")
        
        return vectors

    def _resolve_mode(self, memory_type: str) -> str:
        """Determine encoding mode based on memory type and backend."""
        if self._default_mode != "adaptive":
            return self._default_mode
        
        from ..backends.base import InterfaceCapability
        caps = self._backend.get_capabilities()
        
        # If no activation hooks or KV access, logit_only
        has_deep = bool(caps & (InterfaceCapability.ACTIVATION_HOOKS | InterfaceCapability.KV_CACHE_ACCESS))
        
        if not has_deep:
            return "logit_only"
        
        # High-value memories get full encoding
        FULL_ENCODE_TYPES = {"constraint", "semantic_fact", "knowledge"}
        if memory_type in FULL_ENCODE_TYPES:
            return "full"
        
        # Others get fast encoding
        return "fast"
```

---

## Task 3: Encoding Cache & Store

### Description
Encoded memories are expensive to produce. Store them persistently so re-encoding is unnecessary when the same memory is retrieved again.

### Sub-task 3.1: `EncodedMemoryStore`

```python
import hashlib
import os
import pickle
from pathlib import Path
from typing import Optional


class EncodedMemoryStore:
    """
    Persistent cache for encoded memories.
    
    Key: (memory_id, model_name, encoding_mode)
    Value: EncodedMemory
    
    Storage:
    - L1: In-memory dict (fast, limited by RAM)
    - L2: Disk cache (slower, persists across restarts)
    - L3: PostgreSQL (optional, for shared multi-instance deployments)
    
    Invalidation:
    - Memory text changes → invalidate
    - Model changes → invalidate all
    - Encoding mode changes → invalidate affected
    """

    def __init__(
        self,
        cache_dir: str = ".cache/encoded_memories",
        max_memory_entries: int = 10000,
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, EncodedMemory] = {}
        self._max_entries = max_memory_entries

    def _cache_key(self, memory_id: str, model_name: str, mode: str) -> str:
        raw = f"{memory_id}:{model_name}:{mode}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def get(self, memory_id: str, model_name: str, mode: str) -> Optional[EncodedMemory]:
        """Retrieve encoded memory from cache."""
        key = self._cache_key(memory_id, model_name, mode)
        
        # L1: in-memory
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # L2: disk
        path = self._cache_dir / f"{key}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    encoded = pickle.load(f)
                self._memory_cache[key] = encoded
                return encoded
            except Exception:
                pass
        
        return None

    def put(self, model_name: str, encoded: EncodedMemory) -> None:
        """Store encoded memory in cache."""
        key = self._cache_key(encoded.memory_id, model_name, encoded.encoding_method)
        
        # L1: in-memory
        self._memory_cache[key] = encoded
        
        # Evict if over capacity (LRU approximation: remove oldest)
        if len(self._memory_cache) > self._max_entries:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        # L2: disk (serialize without CUDA tensors)
        encoded_cpu = self._to_cpu(encoded)
        path = self._cache_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(encoded_cpu, f)

    def invalidate(self, memory_id: str) -> None:
        """Invalidate all encodings for a memory."""
        keys_to_remove = [
            k for k in self._memory_cache
            if memory_id in k  # Simplified; real impl uses proper key structure
        ]
        for k in keys_to_remove:
            del self._memory_cache[k]
        
        # Remove from disk
        for path in self._cache_dir.glob(f"*{memory_id[:8]}*.pkl"):
            path.unlink(missing_ok=True)

    def invalidate_all(self) -> None:
        """Clear entire cache."""
        self._memory_cache.clear()
        for f in self._cache_dir.glob("*.pkl"):
            f.unlink()

    @staticmethod
    def _to_cpu(encoded: EncodedMemory) -> EncodedMemory:
        """Move all tensors to CPU for serialization."""
        import copy
        encoded = copy.copy(encoded)
        
        if encoded.embedding is not None:
            encoded.embedding = encoded.embedding.cpu()
        
        if encoded.steering_vectors:
            encoded.steering_vectors = {
                k: v.cpu() for k, v in encoded.steering_vectors.items()
            }
        
        if encoded.kv_pairs:
            encoded.kv_pairs = {
                k: (kk.cpu(), vv.cpu()) for k, (kk, vv) in encoded.kv_pairs.items()
            }
        
        return encoded
```

---

## Task 4: Projection Head (Learned Steering Vector)

### Description
For maximum quality, train a small projection network that maps memory embeddings to optimal steering vectors. This replaces the heuristic methods (CDD, Identity V) with a learned mapping.

### Sub-task 4.1: `ProjectionHead`

```python
import torch.nn as nn


class SteeringProjectionHead(nn.Module):
    """
    Learned projection: embedding → steering vector.
    
    A small MLP that maps a standard text embedding (e.g., 1536-dim from
    text-embedding-3) to a steering vector in the model's hidden space
    (e.g., 4096-dim for Llama).
    
    This is trained on (memory_embedding, effective_steering_vector) pairs
    where effective_steering_vector is derived from CDD on labeled examples.
    
    Architecture:
        embed_dim → 2*hidden_dim → hidden_dim → hidden_dim
    
    Training:
        Loss = cosine_similarity(projected, target_cdd_vector)
             + λ * norm_regularization
    
    This is the "learned encoder" that Phase 6 introduces as an upgrade
    over the heuristic methods. Once trained, it replaces CDD for
    fast, high-quality steering vector generation.
    """

    def __init__(
        self,
        input_dim: int = 1536,     # text-embedding-3 dimension
        hidden_dim: int = 4096,     # model hidden dimension
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._hidden_dim = hidden_dim

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Project embedding to steering vector.
        
        Input: [batch, embed_dim] or [embed_dim]
        Output: [batch, hidden_dim] or [hidden_dim] (unit-normalized)
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        projected = self.net(embedding)
        # Normalize to unit vector
        projected = projected / (torch.norm(projected, dim=-1, keepdim=True) + 1e-8)
        
        return projected.squeeze(0) if projected.shape[0] == 1 else projected


class ProjectionHeadTrainer:
    """
    Train the SteeringProjectionHead on collected (embedding, CDD_vector) pairs.
    
    Training data collection:
    1. During normal operation, CDD computes steering vectors for memories
    2. We pair these with the memory's text embedding
    3. Periodically train/fine-tune the projection head on collected pairs
    
    This creates a flywheel:
    - Initially: CDD for all memories (slow)
    - After N memories: projection head takes over (fast)
    - CDD still runs occasionally for calibration/correction
    """

    def __init__(
        self,
        projection_head: SteeringProjectionHead,
        learning_rate: float = 1e-4,
        min_training_pairs: int = 100,
    ):
        self._head = projection_head
        self._lr = learning_rate
        self._min_pairs = min_training_pairs
        self._training_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._optimizer = torch.optim.AdamW(self._head.parameters(), lr=learning_rate)

    def add_training_pair(
        self, embedding: torch.Tensor, target_vector: torch.Tensor
    ) -> None:
        """Add a (embedding, target_steering_vector) pair."""
        self._training_data.append((
            embedding.detach().cpu(),
            target_vector.detach().cpu(),
        ))

    def should_train(self) -> bool:
        return len(self._training_data) >= self._min_pairs

    def train_epoch(self, batch_size: int = 32) -> float:
        """Run one training epoch. Returns average loss."""
        if len(self._training_data) < batch_size:
            return 0.0
        
        self._head.train()
        total_loss = 0.0
        batches = 0
        
        # Shuffle
        import random
        random.shuffle(self._training_data)
        
        for i in range(0, len(self._training_data), batch_size):
            batch = self._training_data[i:i+batch_size]
            embeddings = torch.stack([b[0] for b in batch])
            targets = torch.stack([b[1] for b in batch])
            
            # Forward
            predicted = self._head(embeddings)
            
            # Cosine similarity loss (maximize similarity)
            cos_sim = nn.functional.cosine_similarity(predicted, targets, dim=-1)
            loss = 1.0 - cos_sim.mean()
            
            # Backward
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        self._head.eval()
        avg_loss = total_loss / max(batches, 1)
        logger.info(f"Projection head training: avg_loss={avg_loss:.4f}, "
                    f"pairs={len(self._training_data)}")
        return avg_loss
```

---

## Acceptance Criteria

1. `HippocampalEncoder.encode()` produces all three representations (logit, activation, KV) in a single call
2. Adaptive mode selects encoding depth based on memory type and backend capabilities
3. `EncodedMemoryStore` caches encodings to disk and memory with proper invalidation
4. `SteeringProjectionHead` maps embeddings to steering vectors with < 100ms latency
5. Projection head training collects pairs and trains when threshold is met
6. `EncodedMemory.to_memory_vector()` correctly converts to bus-compatible format
7. Fallback chain: if CDD fails, fall back to Identity V; if that fails, logit-only
8. Text analysis extracts meaningful entities and concepts
9. Re-encoding is skipped when cache hit (verified by encoding time < 1ms on cache hit)
10. All tensor operations support both CPU and CUDA devices

## Estimated Effort
- **Duration:** 3-4 weeks
- **Complexity:** High (unified pipeline with many integration points)
- **Risk:** Medium (projection head quality depends on training data volume)

## Testing Strategy
1. Unit test full encoding pipeline with small model
2. Verify encoding modes: full, fast, logit_only produce correct outputs
3. Cache hit test: encode → store → retrieve → verify identical
4. Projection head training: verify loss decreases over epochs
5. Adaptive mode test: verify correct mode selection for each memory type
6. Performance benchmark: encoding time for full vs. fast vs. logit_only modes
