# Phase 8: Weight Adaptation Interface — Dynamic LoRA

**Intrinsic Phase I-8** (planned; not yet implemented). See [ActiveCML/README.md](README.md) for the mapping of I-1..I-10 to core CML phases.

## Overview

The Weight Adaptation Interface is the most **advanced** and **deepest** injection mechanism. Instead of modifying activations or caches during inference, it modifies the model's **weights themselves** by loading task-specific Low-Rank Adaptation (LoRA) adapters at runtime. This is the equivalent of **synaptic plasticity** — the model temporarily "learns" from the memory.

### Core Principle: Dynamic LoRA Routing
Pre-train domain-specific LoRA adapters (e.g., "medical knowledge", "user preferences", "code generation") and load them on-the-fly based on the query context. The Controller classifies the query, selects the appropriate adapter(s), and merges them into the base model before generation.

### When to Use Weight Adaptation
- **Specialized domains:** The user's memory includes deep domain knowledge that benefits from fine-tuned behavior
- **Persistent personality/style:** User preferences that should affect all outputs (not just a single token)
- **Skill modules:** Procedural knowledge that requires consistent behavioral changes

### Advanced: Hypernetworks
A small "hypernetwork" that generates LoRA weights on-the-fly based on the query, eliminating the need for pre-trained adapters. This is the most sophisticated form of dynamic adaptation.

### Dependencies
- Phase 1: Model Backend (weight access capability)
- Phase 5: Controller (adapter selection)
- Phase 6: Encoding Pipeline (memory-to-adapter mapping)
- PEFT library (parameter-efficient fine-tuning)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Weight Adaptation Interface                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Adapter Registry                                     │    │
│  │  ──────────────────────────────                                       │    │
│  │  Pre-trained LoRA adapters indexed by domain/task:                    │    │
│  │  - "medical" → adapter_medical.safetensors (r=16, α=32)             │    │
│  │  - "legal" → adapter_legal.safetensors                               │    │
│  │  - "user_style" → adapter_user_123.safetensors                       │    │
│  │  - "coding" → adapter_coding.safetensors                             │    │
│  │  Max loaded simultaneously: 3 (GPU memory constraint)                │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Adapter Router / Classifier                          │    │
│  │  ──────────────────────────────────────                               │    │
│  │  query → embedding → classifier → adapter selection                   │    │
│  │  "What medication for..." → ["medical", "user_preferences"]          │    │
│  │  Supports multi-adapter fusion (weighted merge)                       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Adapter Manager                                      │    │
│  │  ──────────────────────────                                           │    │
│  │  - Load/unload adapters from disk/GPU                                │    │
│  │  - Merge/unmerge with base weights                                   │    │
│  │  - Handle adapter conflicts (mutual exclusion)                       │    │
│  │  - Cache recently used adapters in GPU memory                        │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Hypernetwork (Advanced)                               │    │
│  │  ──────────────────────────────                                       │    │
│  │  Small MLP that generates LoRA Δ weights from query embedding:       │    │
│  │  query_embed → HyperNet → (ΔW_A, ΔW_B) for each target layer       │    │
│  │  Creates "instant experts" without pre-training adapters              │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Adapter Registry & Manager

### Sub-task 1.1: `AdapterRegistry`

**File:** `src/intrinsic/interfaces/weight.py`

```python
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


@dataclass
class AdapterSpec:
    """Specification for a LoRA adapter."""
    name: str
    path: str                           # Path to adapter weights
    domain: str                         # "medical", "legal", "coding", etc.
    rank: int = 16                      # LoRA rank
    alpha: float = 32.0                 # LoRA scaling factor
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ])
    description: str = ""
    size_bytes: int = 0
    loaded: bool = False


class AdapterRegistry:
    """
    Registry of available LoRA adapters.
    
    Manages the catalog of pre-trained adapters, their metadata,
    and which are currently loaded into the model.
    """

    def __init__(self, adapter_dir: str = "./lora_adapters", max_loaded: int = 3):
        self._adapter_dir = Path(adapter_dir)
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        self._max_loaded = max_loaded
        self._catalog: Dict[str, AdapterSpec] = {}
        self._loaded: List[str] = []  # Currently loaded adapter names
        self._scan_adapters()

    def _scan_adapters(self):
        """Scan adapter directory for available adapters."""
        for adapter_path in self._adapter_dir.iterdir():
            if adapter_path.is_dir() and (adapter_path / "adapter_config.json").exists():
                import json
                with open(adapter_path / "adapter_config.json") as f:
                    config = json.load(f)
                
                spec = AdapterSpec(
                    name=adapter_path.name,
                    path=str(adapter_path),
                    domain=config.get("domain", adapter_path.name),
                    rank=config.get("r", 16),
                    alpha=config.get("lora_alpha", 32),
                    target_modules=config.get("target_modules", []),
                    description=config.get("description", ""),
                )
                self._catalog[spec.name] = spec
                logger.info(f"Found adapter: {spec.name} (domain={spec.domain})")

    def get_adapter(self, name: str) -> Optional[AdapterSpec]:
        return self._catalog.get(name)

    def get_adapters_for_domain(self, domain: str) -> List[AdapterSpec]:
        return [a for a in self._catalog.values() if a.domain == domain]

    def list_adapters(self) -> List[AdapterSpec]:
        return list(self._catalog.values())

    def register_adapter(self, spec: AdapterSpec) -> None:
        self._catalog[spec.name] = spec

    @property
    def loaded_adapters(self) -> List[str]:
        return self._loaded.copy()

    @property
    def max_loaded(self) -> int:
        return self._max_loaded


class AdapterManager:
    """
    Manages the lifecycle of LoRA adapters on the model.
    
    Operations:
    - Load: merge adapter weights with base model
    - Unload: revert to base weights
    - Swap: unload current, load new (atomic)
    - Fuse: merge multiple adapters with weights
    """

    def __init__(self, backend, registry: AdapterRegistry):
        self._backend = backend
        self._registry = registry
        self._active_adapters: Dict[str, float] = {}  # name → scaling

    async def load_adapter(self, adapter_name: str, scaling: float = 1.0) -> bool:
        """
        Load a LoRA adapter onto the model.
        
        If max loaded adapters exceeded, unload the least recently used one.
        """
        spec = self._registry.get_adapter(adapter_name)
        if spec is None:
            logger.error(f"Adapter not found: {adapter_name}")
            return False

        # Check capacity
        if len(self._active_adapters) >= self._registry.max_loaded:
            # Unload oldest
            oldest = next(iter(self._active_adapters))
            await self.unload_adapter(oldest)

        try:
            self._backend.load_lora_adapter(
                adapter_path=spec.path,
                adapter_name=adapter_name,
                scaling=scaling,
            )
            self._active_adapters[adapter_name] = scaling
            spec.loaded = True
            logger.info(f"Loaded adapter: {adapter_name} (scaling={scaling})")
            return True
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {e}")
            return False

    async def unload_adapter(self, adapter_name: str) -> bool:
        """Unload a LoRA adapter, reverting to base weights."""
        if adapter_name not in self._active_adapters:
            return False

        try:
            self._backend.unload_lora_adapter(adapter_name)
            del self._active_adapters[adapter_name]
            spec = self._registry.get_adapter(adapter_name)
            if spec:
                spec.loaded = False
            logger.info(f"Unloaded adapter: {adapter_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload adapter {adapter_name}: {e}")
            return False

    async def swap_adapter(self, old_name: str, new_name: str, scaling: float = 1.0) -> bool:
        """Atomically swap one adapter for another."""
        await self.unload_adapter(old_name)
        return await self.load_adapter(new_name, scaling)

    def get_active_adapters(self) -> Dict[str, float]:
        return self._active_adapters.copy()
```

---

## Task 2: Adapter Router / Classifier

### Sub-task 2.1: `AdapterRouter`

```python
class AdapterRouter:
    """
    Routes queries to appropriate LoRA adapters based on content classification.
    
    Classification methods:
    1. Embedding similarity: compare query embedding to domain embeddings
    2. Keyword matching: fast, rule-based routing
    3. LLM classification: most accurate, uses the LLM itself
    """

    def __init__(
        self,
        registry: AdapterRegistry,
        embedding_client=None,
        classification_method: str = "embedding",  # "embedding" | "keyword" | "llm"
    ):
        self._registry = registry
        self._embedding_client = embedding_client
        self._method = classification_method
        self._domain_embeddings: Dict[str, torch.Tensor] = {}

    async def initialize_domain_embeddings(self):
        """Pre-compute embeddings for each adapter domain."""
        if not self._embedding_client:
            return
        
        for spec in self._registry.list_adapters():
            # Use domain name + description as the embedding source
            text = f"{spec.domain}: {spec.description}"
            embedding = await self._embedding_client.embed(text)
            self._domain_embeddings[spec.name] = torch.tensor(embedding)

    async def route(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        top_k: int = 2,
    ) -> List[Tuple[str, float]]:
        """
        Select adapters for a query.
        
        Returns: List of (adapter_name, confidence_score) tuples.
        """
        if self._method == "embedding":
            return await self._route_by_embedding(query, top_k)
        elif self._method == "keyword":
            return self._route_by_keyword(query, top_k)
        else:
            return [(self._registry.list_adapters()[0].name, 1.0)] if self._registry.list_adapters() else []

    async def _route_by_embedding(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Route by embedding similarity."""
        if not self._embedding_client or not self._domain_embeddings:
            return []
        
        query_embedding = torch.tensor(await self._embedding_client.embed(query))
        
        scores = {}
        for name, domain_emb in self._domain_embeddings.items():
            sim = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                domain_emb.unsqueeze(0),
            ).item()
            scores[name] = sim
        
        # Sort by similarity and return top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def _route_by_keyword(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Fast keyword-based routing."""
        query_lower = query.lower()
        scores = {}
        
        DOMAIN_KEYWORDS = {
            "medical": ["health", "medication", "symptom", "doctor", "diagnosis", "treatment"],
            "legal": ["law", "legal", "contract", "court", "regulation", "compliance"],
            "coding": ["code", "function", "class", "debug", "error", "implement"],
            "creative": ["story", "write", "poem", "creative", "narrative", "fiction"],
        }
        
        for spec in self._registry.list_adapters():
            keywords = DOMAIN_KEYWORDS.get(spec.domain, [])
            match_count = sum(1 for kw in keywords if kw in query_lower)
            if match_count > 0:
                scores[spec.name] = match_count / len(keywords)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
```

---

## Task 3: Hypernetwork (Advanced)

### Sub-task 3.1: `HyperNetwork` — Generate LoRA Weights On-the-Fly

```python
import torch.nn as nn


class HyperNetwork(nn.Module):
    """
    Generates LoRA adapter weights on-the-fly from a query embedding.
    
    Instead of loading pre-trained adapters, the hypernetwork produces
    (ΔW_A, ΔW_B) weight matrices for each target layer, creating
    "instant experts" tailored to the specific query context.
    
    Architecture:
        query_embedding [embed_dim]
        → shared encoder (MLP)
        → per-layer heads → (ΔW_A [rank, hidden], ΔW_B [hidden, rank])
    
    The generated LoRA weights are:
        W' = W + α/r · ΔW_B @ ΔW_A
    
    Training:
        1. Collect (query, ideal_output) pairs
        2. Generate LoRA weights via hypernetwork
        3. Apply to model, generate output
        4. Loss = quality_loss + regularization
    
    This is highly experimental and requires significant training data.
    It is presented as a future research direction.
    """

    def __init__(
        self,
        input_dim: int = 1536,      # Query embedding dimension
        model_hidden_dim: int = 4096, # LLM hidden dimension
        lora_rank: int = 8,           # Rank of generated LoRA
        num_target_layers: int = 8,   # How many layers to adapt
        bottleneck_dim: int = 256,    # Shared encoder bottleneck
    ):
        super().__init__()
        self._lora_rank = lora_rank
        self._model_dim = model_hidden_dim
        self._num_layers = num_target_layers

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.GELU(),
        )

        # Per-layer heads: generate ΔW_A and ΔW_B for each layer
        self.layer_heads_A = nn.ModuleList([
            nn.Linear(bottleneck_dim, lora_rank * model_hidden_dim)
            for _ in range(num_target_layers)
        ])
        self.layer_heads_B = nn.ModuleList([
            nn.Linear(bottleneck_dim, model_hidden_dim * lora_rank)
            for _ in range(num_target_layers)
        ])

    def forward(
        self, query_embedding: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate LoRA weight pairs for each target layer.
        
        Input: query_embedding [embed_dim]
        Output: List of (ΔW_A, ΔW_B) per layer
            ΔW_A: [rank, model_hidden_dim]
            ΔW_B: [model_hidden_dim, rank]
        """
        # Shared encoding
        h = self.shared_encoder(query_embedding)

        lora_weights = []
        for i in range(self._num_layers):
            # Generate A and B matrices
            delta_a = self.layer_heads_A[i](h)
            delta_a = delta_a.view(self._lora_rank, self._model_dim)
            
            delta_b = self.layer_heads_B[i](h)
            delta_b = delta_b.view(self._model_dim, self._lora_rank)
            
            lora_weights.append((delta_a, delta_b))

        return lora_weights
```

---

## Task 4: Memory Bus Integration

### Sub-task 4.1: Bus Handler for Dynamic LoRA

```python
async def setup_weight_interface(bus, backend, registry, embedding_client):
    """Wire the Weight Adaptation Interface into the Memory Bus."""
    
    manager = AdapterManager(backend, registry)
    router = AdapterRouter(registry, embedding_client)
    await router.initialize_domain_embeddings()
    
    async def _weight_handler(memory_vectors):
        """
        Bus handler: select and load appropriate adapters.
        """
        if not memory_vectors:
            return
        
        # Use the first memory's text as query context
        query_context = " ".join(mv.source_text[:100] for mv in memory_vectors[:3])
        
        # Route to adapters
        selections = await router.route(query_context, top_k=2)
        
        for adapter_name, confidence in selections:
            if confidence > 0.5:
                await manager.load_adapter(adapter_name, scaling=confidence)
    
    bus.register_handler(InjectionChannel.DYNAMIC_LORA, _weight_handler)
    
    return manager, router
```

---

## Acceptance Criteria

1. `AdapterRegistry` discovers and catalogs available LoRA adapters
2. `AdapterManager` loads/unloads adapters with capacity management
3. `AdapterRouter` classifies queries and selects appropriate adapters
4. Multiple adapters can be loaded simultaneously (up to max_loaded)
5. Adapter swapping is atomic (old removed before new loaded)
6. `HyperNetwork` generates valid LoRA weight pairs from embeddings
7. Memory Bus handler for `DYNAMIC_LORA` channel is functional
8. Graceful degradation: if no adapters available, skip weight adaptation
9. GPU memory tracked for loaded adapters (integrated with Phase 7 hierarchy)

## Estimated Effort
- **Duration:** 4-6 weeks (including adapter training pipeline)
- **Complexity:** Very High
- **Risk:** High (adapter quality depends on training data; hypernetwork is experimental)

## Testing Strategy
1. Unit test adapter loading/unloading with a small model
2. Verify adapter routing selects correct domain
3. Integration test: load medical adapter → ask medical question → verify improved accuracy
4. Multi-adapter test: load 2 adapters, verify both influence output
5. Capacity test: exceed max_loaded, verify LRU eviction
6. Hypernetwork test: verify generated weights have correct shapes
