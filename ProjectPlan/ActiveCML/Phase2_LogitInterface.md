# Phase 2: Logit Interface — Token-Level Memory Integration

**Intrinsic Phase I-2** (planned; not yet implemented). See [ActiveCML/README.md](README.md) for the mapping of I-1..I-10 to core CML phases.

## Overview

The Logit Interface is the **safest and most universally available** memory injection mechanism. It operates at the very end of the forward pass, modulating the probability distribution over the vocabulary before token sampling. This is the only interface that works with API-only providers (OpenAI, Anthropic) via the `logit_bias` parameter.

This phase implements two sub-interfaces:

1. **Simple Logit Bias** — Directly bias specific token probabilities (e.g., boost "Paris" when the user lives in Paris). Available via API `logit_bias` parameter.
2. **kNN-LM Interpolation** — Interpolate the model's parametric distribution with a non-parametric memory distribution derived from nearest-neighbor lookup. Requires access to the full logit distribution (local models only).

### Why Logit Interface First?
- Universally supported (even API-only backends)
- No risk of destabilizing internal representations
- Provides immediate, measurable improvements over RAG
- Serves as the **fallback** when deeper interfaces are unavailable

### Theoretical Foundation
From the kNN-LM paper (Khandelwal et al., 2019):
```
p(w | context) = λ · p_kNN(w | context) + (1 - λ) · p_LM(w | context)
```
Where:
- `p_LM` = model's parametric distribution (from logits)
- `p_kNN` = non-parametric distribution from memory store
- `λ` = interpolation weight (0 = pure LLM, 1 = pure memory)

### Dependencies
- Phase 1: Model Backend, Hook Manager, Memory Bus
- Existing: Memory Retriever, Embedding Client, Vector Store (pgvector)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Logit Interface                                │
│                                                                      │
│  ┌────────────────────┐    ┌────────────────────────────────────┐   │
│  │  Simple Bias Engine │    │  kNN-LM Interpolator               │   │
│  │  ──────────────────│    │  ──────────────────────────────────│   │
│  │  memory → token map │    │  query embedding → kNN lookup      │   │
│  │  token → bias value │    │  distance → probability            │   │
│  │  apply logit_bias   │    │  λ-interpolation with p_LM         │   │
│  │                     │    │  temperature-scaled softmax        │   │
│  │  [API + Local]      │    │  [Local only - needs raw logits]   │   │
│  └────────────────────┘    └────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Token-Memory Mapper                          │   │
│  │  ──────────────────────────────────────────────────────────  │   │
│  │  semantic_fact → key tokens → token_ids → bias values        │   │
│  │  "User lives in Paris" → ["Paris"] → [3681] → {3681: 2.5}  │   │
│  │  Uses LLM extraction + tokenizer mapping                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Adaptive λ Controller                        │   │
│  │  ──────────────────────────────────────────────────────────  │   │
│  │  Adjusts interpolation weight based on:                      │   │
│  │  - Memory relevance score                                    │   │
│  │  - Memory confidence                                         │   │
│  │  - Query-memory distance                                     │   │
│  │  - Memory freshness (temporal decay)                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Token-Memory Mapper

### Description
Convert retrieved memory records (text) into token-level representations. A semantic fact like "User lives in Paris" must be mapped to specific vocabulary tokens ("Paris", "France", "French") with associated bias weights. This mapper is the bridge between textual memory and token-space influence.

### Sub-task 1.1: `TokenMemoryMapper` — Extract Key Tokens from Memories

**File:** `src/intrinsic/interfaces/logit.py`

```python
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import torch

logger = logging.getLogger(__name__)


@dataclass
class TokenBias:
    """A single token bias entry."""
    token_id: int
    token_text: str
    bias_value: float
    source_memory_id: str
    confidence: float


@dataclass
class MemoryTokenMap:
    """Mapping from a memory record to its token-level representation."""
    memory_id: str
    source_text: str
    key_entities: List[str]              # Extracted entities/keywords
    token_biases: List[TokenBias]        # Final token bias entries
    total_bias_budget: float = 0.0       # Sum of absolute bias values


class TokenMemoryMapper:
    """
    Maps memory records to token-level bias entries.
    
    Pipeline:
    1. Extract key entities/concepts from memory text (LLM or rule-based)
    2. Map entities to vocabulary tokens via tokenizer
    3. Compute bias values based on relevance, confidence, and entity salience
    """

    def __init__(
        self,
        tokenizer,
        llm_client=None,
        max_bias_value: float = 5.0,
        max_tokens_per_memory: int = 10,
    ):
        self._tokenizer = tokenizer
        self._llm_client = llm_client
        self._max_bias = max_bias_value
        self._max_tokens = max_tokens_per_memory
        # Cache entity → token mappings
        self._entity_token_cache: Dict[str, List[int]] = {}

    async def map_memory(
        self,
        memory_id: str,
        memory_text: str,
        relevance: float,
        confidence: float,
        memory_type: str = "semantic_fact",
    ) -> MemoryTokenMap:
        """
        Convert a single memory record to token bias entries.
        
        Steps:
        1. Extract key entities from memory text
        2. Map each entity to token IDs
        3. Compute bias value: bias = base_weight * relevance * confidence
        """
        # Step 1: Extract entities
        entities = await self._extract_entities(memory_text, memory_type)

        # Step 2: Map to tokens
        token_biases = []
        seen_tokens: Set[int] = set()

        for entity in entities[:self._max_tokens]:
            token_ids = self._entity_to_tokens(entity)
            for tid in token_ids:
                if tid in seen_tokens:
                    continue
                seen_tokens.add(tid)

                # Step 3: Compute bias value
                # Base weight depends on memory type
                base_weight = self._base_weight_for_type(memory_type)
                bias_value = base_weight * relevance * confidence
                bias_value = max(-self._max_bias, min(self._max_bias, bias_value))

                token_biases.append(TokenBias(
                    token_id=tid,
                    token_text=self._tokenizer.decode([tid]),
                    bias_value=bias_value,
                    source_memory_id=memory_id,
                    confidence=confidence,
                ))

        return MemoryTokenMap(
            memory_id=memory_id,
            source_text=memory_text,
            key_entities=entities,
            token_biases=token_biases,
            total_bias_budget=sum(abs(tb.bias_value) for tb in token_biases),
        )

    async def _extract_entities(self, text: str, memory_type: str) -> List[str]:
        """
        Extract key entities/concepts from memory text.
        
        For semantic_facts: extract subject, object, key nouns
        For preferences: extract the preference target
        For constraints: extract the constrained entity
        
        Uses LLM for complex extraction, falls back to rule-based.
        """
        if self._llm_client:
            prompt = (
                f"Extract the 3-5 most important named entities, concepts, or keywords "
                f"from this {memory_type}. Return as JSON array of strings.\n\n"
                f"Memory: \"{text}\"\n\n"
                f"Entities:"
            )
            try:
                result = await self._llm_client.complete_json(prompt)
                if isinstance(result, list):
                    return result[:5]
            except Exception:
                pass

        # Rule-based fallback: simple noun extraction via tokenization
        # Split on common delimiters, filter stopwords
        words = text.replace(",", " ").replace(".", " ").split()
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                      "at", "to", "for", "of", "with", "that", "this", "user",
                      "prefers", "likes", "lives", "has", "does"}
        entities = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        return entities[:5]

    def _entity_to_tokens(self, entity: str) -> List[int]:
        """Map an entity string to its vocabulary token IDs."""
        if entity in self._entity_token_cache:
            return self._entity_token_cache[entity]

        # Tokenize the entity
        token_ids = self._tokenizer.encode(entity, add_special_tokens=False)

        # Also check for capitalized/lowercase variants
        variants = [entity, entity.lower(), entity.capitalize(), entity.upper()]
        all_ids = set(token_ids)
        for variant in variants:
            ids = self._tokenizer.encode(variant, add_special_tokens=False)
            # Only include single-token variants (whole-word tokens are more meaningful)
            if len(ids) == 1:
                all_ids.add(ids[0])

        result = list(all_ids)
        self._entity_token_cache[entity] = result
        return result

    def _base_weight_for_type(self, memory_type: str) -> float:
        """Base bias weight by memory type."""
        weights = {
            "semantic_fact": 3.0,      # Strong: established facts
            "preference": 2.5,          # Medium-strong: user preferences
            "constraint": 4.0,          # Very strong: must-follow rules
            "episodic_event": 1.5,      # Moderate: recent events
            "hypothesis": 0.5,          # Weak: unconfirmed
            "procedure": 2.0,           # Medium: how-to knowledge
            "knowledge": 2.5,           # Medium-strong: domain knowledge
        }
        return weights.get(memory_type, 1.5)
```

---

## Task 2: Simple Logit Bias Engine

### Description
The simplest injection mechanism — compute a `logit_bias` dictionary from retrieved memories and pass it to the generation API call. Works with OpenAI API, vLLM, and local models.

### Sub-task 2.1: `LogitBiasEngine`

**File:** `src/intrinsic/interfaces/logit.py` (continued)

```python
class LogitBiasEngine:
    """
    Applies token-level logit biases derived from memory.
    
    This is the SIMPLEST and SAFEST injection mechanism.
    Works with API-only backends via the logit_bias parameter.
    
    Flow:
    1. Receive retrieved memories from the bus
    2. Map memories to token biases via TokenMemoryMapper
    3. Aggregate biases (handle conflicts from multiple memories)
    4. Apply to backend (API logit_bias or local logit modification)
    """

    def __init__(
        self,
        backend,  # ModelBackend
        token_mapper: TokenMemoryMapper,
        max_total_tokens: int = 50,
        conflict_strategy: str = "sum_capped",  # "sum_capped" | "max_wins" | "weighted_avg"
    ):
        self._backend = backend
        self._mapper = token_mapper
        self._max_total_tokens = max_total_tokens
        self._conflict_strategy = conflict_strategy
        self._active_biases: Dict[int, float] = {}

    async def process_memories(self, memory_vectors) -> Dict[int, float]:
        """
        Convert memory vectors to aggregated logit bias dict.
        
        memory_vectors: List[MemoryVector] from the bus
        Returns: {token_id: bias_value} ready for API call
        """
        all_maps: List[MemoryTokenMap] = []

        for mv in memory_vectors:
            token_map = await self._mapper.map_memory(
                memory_id=mv.memory_id,
                memory_text=mv.source_text,
                relevance=mv.relevance_score,
                confidence=mv.injection_strength * mv.decay_factor,
                memory_type="semantic_fact",
            )
            all_maps.append(token_map)

        # Aggregate across all memories
        aggregated = self._aggregate_biases(all_maps)

        # Truncate to budget
        if len(aggregated) > self._max_total_tokens:
            # Keep highest-magnitude biases
            sorted_biases = sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True)
            aggregated = dict(sorted_biases[:self._max_total_tokens])

        self._active_biases = aggregated
        return aggregated

    def _aggregate_biases(self, token_maps: List[MemoryTokenMap]) -> Dict[int, float]:
        """
        Aggregate biases from multiple memories.
        Multiple memories may bias the same token — resolve conflicts.
        """
        merged: Dict[int, List[float]] = {}

        for tmap in token_maps:
            for tb in tmap.token_biases:
                merged.setdefault(tb.token_id, []).append(tb.bias_value)

        result = {}
        for token_id, values in merged.items():
            if self._conflict_strategy == "sum_capped":
                # Sum biases but cap at max_bias
                result[token_id] = max(-5.0, min(5.0, sum(values)))
            elif self._conflict_strategy == "max_wins":
                # Take the largest absolute bias
                result[token_id] = max(values, key=abs)
            elif self._conflict_strategy == "weighted_avg":
                result[token_id] = sum(values) / len(values)

        return result

    async def apply(self, memory_vectors) -> None:
        """
        Full pipeline: process memories → apply bias to backend.
        Called by the IntrinsicMemoryBus handler.
        """
        bias_dict = await self.process_memories(memory_vectors)
        if bias_dict:
            self._backend.apply_logit_bias(bias_dict)
            logger.info(f"Applied logit bias to {len(bias_dict)} tokens")

    def get_active_biases(self) -> Dict[int, float]:
        """Return currently active bias dict (for debugging)."""
        return self._active_biases.copy()

    def clear(self) -> None:
        """Clear all active biases."""
        self._active_biases.clear()
        self._backend.apply_logit_bias({})
```

---

## Task 3: kNN-LM Interpolator

### Description
Implement the full kNN-LM framework that interpolates the model's parametric distribution with a memory-based non-parametric distribution. This provides stronger memory influence than simple logit bias, but requires access to the full logit distribution (local models only).

### Sub-task 3.1: Memory Datastore for kNN Lookup

**File:** `src/intrinsic/interfaces/logit.py` (continued)

```python
import numpy as np
from collections import defaultdict


class MemoryDatastore:
    """
    The kNN-LM datastore: maps context embeddings to next-token targets.
    
    For each memory, we store (key, value) pairs where:
    - key = embedding of the context prefix
    - value = the next token that should follow
    
    This is built from the existing vector store but reorganized for
    token-level prediction.
    """

    def __init__(self, embedding_dim: int, temperature: float = 10.0):
        self._embedding_dim = embedding_dim
        self._keys: List[torch.Tensor] = []     # context embeddings
        self._values: List[int] = []              # next-token IDs
        self._metadata: List[Dict] = []           # source memory info
        self._temperature = temperature           # softmax temperature for distance → probability

    def add_entry(
        self,
        context_embedding: torch.Tensor,
        next_token_id: int,
        memory_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Add a (context → next_token) entry to the datastore."""
        self._keys.append(context_embedding.detach().cpu())
        self._values.append(next_token_id)
        self._metadata.append({"memory_id": memory_id, "confidence": confidence})

    def query(
        self,
        query_embedding: torch.Tensor,
        k: int = 8,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Find k-nearest contexts and return their next-token predictions.
        
        Returns:
            tokens: List[int] — the k nearest next-tokens
            distances: Tensor — L2 distances to each neighbor
        """
        if not self._keys:
            return [], torch.tensor([])

        # Stack all keys into a matrix for batch distance computation
        key_matrix = torch.stack(self._keys)  # [N, embedding_dim]
        query = query_embedding.detach().cpu().unsqueeze(0)  # [1, embedding_dim]

        # L2 distance
        distances = torch.cdist(query, key_matrix).squeeze(0)  # [N]

        # Get top-k nearest
        k = min(k, len(self._keys))
        topk_distances, topk_indices = torch.topk(distances, k, largest=False)

        tokens = [self._values[idx] for idx in topk_indices.tolist()]
        return tokens, topk_distances

    def to_probability_distribution(
        self,
        query_embedding: torch.Tensor,
        vocab_size: int,
        k: int = 8,
    ) -> torch.Tensor:
        """
        Compute the full kNN probability distribution over the vocabulary.
        
        p_kNN(w) = Σ_{(k,v) ∈ N} 1[v=w] · softmax(-d(k, query) / T)
        
        Returns: [vocab_size] probability distribution
        """
        tokens, distances = self.query(query_embedding, k)
        if not tokens:
            return torch.zeros(vocab_size)

        # Convert distances to probabilities via temperature-scaled softmax
        # Lower distance = higher probability
        neg_distances = -distances / self._temperature
        knn_probs_k = torch.softmax(neg_distances, dim=0)  # [k]

        # Scatter into full vocabulary distribution
        vocab_dist = torch.zeros(vocab_size)
        for i, token_id in enumerate(tokens):
            vocab_dist[token_id] += knn_probs_k[i].item()

        return vocab_dist

    @property
    def size(self) -> int:
        return len(self._keys)


class KNNLMInterpolator:
    """
    Interpolates parametric LLM distribution with non-parametric memory distribution.
    
    p(w | context) = λ · p_kNN(w | context) + (1 - λ) · p_LM(w | context)
    
    This is the core kNN-LM algorithm adapted for the CognitiveMemoryLayer.
    
    The interpolation weight λ can be:
    - Fixed (constant λ for all contexts)
    - Adaptive (λ depends on memory relevance and confidence)
    - Learned (small network predicting optimal λ — Phase 6)
    """

    def __init__(
        self,
        datastore: MemoryDatastore,
        base_lambda: float = 0.25,
        adaptive: bool = True,
        k_neighbors: int = 8,
    ):
        self._datastore = datastore
        self._base_lambda = base_lambda
        self._adaptive = adaptive
        self._k = k_neighbors

    def interpolate(
        self,
        lm_logits: torch.Tensor,        # [vocab_size] raw logits from LLM
        context_embedding: torch.Tensor,  # [hidden_dim] current context representation
        relevance_scores: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Interpolate LLM logits with kNN memory distribution.
        
        Returns modified logits (not probabilities — the sampling step applies its own softmax).
        """
        vocab_size = lm_logits.shape[-1]

        # Step 1: Get kNN distribution
        knn_dist = self._datastore.to_probability_distribution(
            context_embedding, vocab_size, self._k
        )

        if knn_dist.sum() < 1e-8:
            # No relevant memories — return original logits
            return lm_logits

        # Step 2: Convert LLM logits to probabilities
        lm_probs = torch.softmax(lm_logits, dim=-1)

        # Step 3: Compute interpolation weight
        lam = self._compute_lambda(context_embedding, relevance_scores)

        # Step 4: Interpolate
        # p(w) = λ · p_kNN(w) + (1 - λ) · p_LM(w)
        interpolated_probs = lam * knn_dist.to(lm_probs.device) + (1 - lam) * lm_probs

        # Step 5: Convert back to logits (log-space for compatibility with sampling)
        interpolated_logits = torch.log(interpolated_probs + 1e-10)

        return interpolated_logits

    def _compute_lambda(
        self,
        context_embedding: torch.Tensor,
        relevance_scores: Optional[List[float]] = None,
    ) -> float:
        """
        Compute the interpolation weight λ.
        
        Adaptive λ considers:
        - Distance to nearest memory (closer = higher λ)
        - Average relevance score of retrieved memories
        - Datastore density in this region of embedding space
        """
        if not self._adaptive or relevance_scores is None:
            return self._base_lambda

        # Average relevance as a proxy for memory confidence
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        # Scale λ by relevance: high relevance → trust memory more
        # Sigmoid mapping: λ = base_λ * sigmoid(k * (avg_relevance - threshold))
        import math
        threshold = 0.5
        k = 5.0
        scale = 1.0 / (1.0 + math.exp(-k * (avg_relevance - threshold)))

        adaptive_lambda = self._base_lambda * scale * 2  # 2x to allow λ up to 2*base when very relevant
        return min(adaptive_lambda, 0.9)  # Cap at 0.9 — never fully override LLM
```

---

## Task 4: Logit Hook for Local Models

### Description
For local models, we need to intercept the logits just before sampling and apply the kNN-LM interpolation. This uses the Hook Manager from Phase 1 to register a pre-sampling hook.

### Sub-task 4.1: `LogitInjectionHook`

**File:** `src/intrinsic/interfaces/logit.py` (continued)

```python
class LogitInjectionHook:
    """
    Registers a hook on the final LM head to modify logits before sampling.
    
    This hook:
    1. Captures the raw logits from the model's lm_head
    2. Applies logit bias from the LogitBiasEngine
    3. Applies kNN-LM interpolation from the KNNLMInterpolator
    4. Returns modified logits for sampling
    """

    def __init__(
        self,
        backend,  # ModelBackend
        bias_engine: Optional[LogitBiasEngine] = None,
        knn_interpolator: Optional[KNNLMInterpolator] = None,
    ):
        self._backend = backend
        self._bias_engine = bias_engine
        self._knn_interpolator = knn_interpolator
        self._hook_handle = None
        self._context_embedding: Optional[torch.Tensor] = None
        self._relevance_scores: Optional[List[float]] = None

    def set_context(
        self,
        context_embedding: torch.Tensor,
        relevance_scores: List[float],
    ) -> None:
        """Set the current context for kNN lookup (called before each generation)."""
        self._context_embedding = context_embedding
        self._relevance_scores = relevance_scores

    def attach(self) -> None:
        """Register the logit modification hook on the model's output layer."""
        model = self._backend._model  # Access underlying model

        # Find the lm_head layer
        lm_head = getattr(model, 'lm_head', None)
        if lm_head is None:
            logger.warning("Cannot find lm_head for logit hook attachment")
            return

        def _logit_hook(module, inputs, output):
            """Modify logits after lm_head computation."""
            logits = output  # [batch, seq_len, vocab_size]

            # Only modify the last token's logits (during generation)
            last_logits = logits[:, -1, :]  # [batch, vocab_size]

            # Apply kNN-LM interpolation
            if self._knn_interpolator and self._context_embedding is not None:
                for b in range(last_logits.shape[0]):
                    last_logits[b] = self._knn_interpolator.interpolate(
                        last_logits[b],
                        self._context_embedding,
                        self._relevance_scores,
                    )

            # Apply simple logit bias
            if self._bias_engine:
                biases = self._bias_engine.get_active_biases()
                for token_id, bias_val in biases.items():
                    if token_id < last_logits.shape[-1]:
                        last_logits[:, token_id] += bias_val

            # Reconstruct full logits tensor
            logits[:, -1, :] = last_logits
            return logits

        self._hook_handle = lm_head.register_forward_hook(_logit_hook)
        logger.info("Logit injection hook attached to lm_head")

    def detach(self) -> None:
        """Remove the logit hook."""
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None
```

---

## Task 5: Memory Bus Integration

### Description
Wire the Logit Interface into the `IntrinsicMemoryBus` so retrieved memories are automatically converted to logit biases.

### Sub-task 5.1: Bus Handler Registration

```python
# Integration code (in the startup/initialization path)

async def setup_logit_interface(bus, backend, tokenizer, llm_client):
    """
    Wire the Logit Interface into the Memory Bus.
    Called during application startup.
    """
    # Create components
    token_mapper = TokenMemoryMapper(
        tokenizer=tokenizer,
        llm_client=llm_client,
        max_bias_value=5.0,
        max_tokens_per_memory=10,
    )
    
    bias_engine = LogitBiasEngine(
        backend=backend,
        token_mapper=token_mapper,
        max_total_tokens=50,
        conflict_strategy="sum_capped",
    )
    
    # Register as bus handler
    bus.register_handler(
        InjectionChannel.LOGIT_BIAS,
        bias_engine.apply,
    )
    
    # If local model: also set up kNN-LM
    if backend.supports(InterfaceCapability.LOGIT_DISTRIBUTION):
        datastore = MemoryDatastore(
            embedding_dim=backend.get_model_spec().hidden_dim,
            temperature=10.0,
        )
        interpolator = KNNLMInterpolator(
            datastore=datastore,
            base_lambda=0.25,
            adaptive=True,
        )
        logit_hook = LogitInjectionHook(
            backend=backend,
            bias_engine=bias_engine,
            knn_interpolator=interpolator,
        )
        logit_hook.attach()
    
    return bias_engine
```

---

## Task 6: API Integration for Logit Bias

### Description
Modify the existing `OpenAICompatibleClient` to accept and forward logit bias from the Logit Interface. This is the critical integration point for API-only backends.

### Sub-task 6.1: Extend `OpenAICompatibleClient`

Modify `src/utils/llm.py` to accept a `logit_bias` parameter:

```python
# In OpenAICompatibleClient.complete() method:

async def complete(
    self,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    system_prompt: Optional[str] = None,
    logit_bias: Optional[Dict[int, float]] = None,  # NEW PARAMETER
) -> str:
    """Return raw text completion with optional logit bias."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    kwargs = {
        "model": self._model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Apply logit bias if provided
    if logit_bias:
        # OpenAI API expects {str(token_id): float}
        kwargs["logit_bias"] = {str(k): v for k, v in logit_bias.items()}
    
    response = await self._client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
```

### Sub-task 6.2: Seamless Provider Integration

Extend `SeamlessMemoryProvider` to use the Logit Interface when available:

```python
# In SeamlessMemoryProvider.process_turn():

async def process_turn(self, tenant_id, user_message, ...):
    """Enhanced process_turn with intrinsic memory support."""
    
    # Existing: retrieve context as text
    memory_context, injected_memories = await self._retrieve_context(tenant_id, user_message)
    
    # NEW: If intrinsic memory is enabled, also prepare logit biases
    logit_bias = None
    if self._intrinsic_bus and self._logit_engine:
        # Convert retrieved memories to memory vectors
        from ..intrinsic.bus import MemoryVector, InjectionChannel
        vectors = [
            MemoryVector(
                memory_id=str(mem.record.id),
                source_text=mem.record.text,
                relevance_score=mem.relevance_score,
                injection_strength=mem.record.confidence,
            )
            for mem in injected_memories
        ]
        
        # Route through bus to logit interface
        self._intrinsic_bus.enqueue(InjectionChannel.LOGIT_BIAS, vectors)
        await self._intrinsic_bus.flush(InjectionChannel.LOGIT_BIAS)
        logit_bias = self._logit_engine.get_active_biases()
    
    return SeamlessTurnResult(
        memory_context=memory_context,
        injected_memories=injected_memories,
        stored_count=stored_count,
        reconsolidation_applied=reconsolidation_applied,
        logit_bias=logit_bias,  # NEW field
    )
```

---

## Acceptance Criteria

1. `TokenMemoryMapper` extracts key entities from memory text and maps them to token IDs
2. `LogitBiasEngine` aggregates biases from multiple memories with conflict resolution
3. `KNNLMInterpolator` correctly interpolates parametric and non-parametric distributions
4. `LogitInjectionHook` attaches to lm_head and modifies logits during generation
5. API-only backend correctly forwards logit_bias to OpenAI API
6. Adaptive λ scales interpolation weight based on relevance scores
7. Safety: bias values are clamped within configured bounds
8. Backward compatible: existing RAG behavior unchanged when intrinsic memory is disabled
9. `SeamlessTurnResult` includes optional logit_bias field
10. Memory Bus handler for `LOGIT_BIAS` channel is registered and functional

## Estimated Effort
- **Duration:** 2-3 weeks
- **Complexity:** Medium
- **Risk:** Low (logit bias is the safest injection mechanism)

## Testing Strategy
1. Unit test `TokenMemoryMapper` with various memory types
2. Unit test `KNNLMInterpolator` with synthetic distributions
3. Integration test with local model (small, e.g., TinyLlama)
4. Integration test with OpenAI API (verify logit_bias parameter forwarding)
5. A/B comparison: RAG-only vs. RAG + logit bias on factual recall benchmark
