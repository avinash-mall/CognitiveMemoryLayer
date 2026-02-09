# Phase 3: Activation Interface — Hidden State Steering

## Overview

The Activation Interface is the **primary mechanism** for intrinsic memory integration. Unlike the Logit Interface (which only influences the final token distribution), activation steering injects memory directly into the model's **residual stream** — the hidden state representations flowing between transformer layers. This influences the model's entire reasoning process: its "mood," "topic focus," "reasoning style," and factual priors.

### Core Principle: Linear Representation Hypothesis
Research has established that high-level concepts in LLMs are represented as **linear directions** in activation space. A concept like "Paris" or "formal tone" corresponds to a specific vector direction. By adding a carefully computed **steering vector** to the hidden states during the forward pass, we can shift the model's behavior toward that concept — without modifying any weights.

### Why This Matters
- **Depth of influence:** Affects all subsequent layers' computations (not just the output)
- **Semantic richness:** Can encode complex concepts ("user prefers technical explanations") as single vectors
- **Composability:** Multiple steering vectors can be summed to express compound memories
- **Reversibility:** Remove the hook and the model returns to baseline behavior

### Theoretical Foundation
From the dynamical systems perspective, the LLM is a discrete dynamical system:
```
h_{l+1} = F_l(h_l) + memory_injection_l
```
Where `h_l` is the hidden state at layer `l`, `F_l` is the transformer layer function, and `memory_injection_l` is our steering vector. The injection acts as a **control signal** on the dynamical system.

### Dependencies
- Phase 1: Model Backend (activation hooks), Hook Manager, Memory Bus
- Phase 2: Logit Interface (fallback)
- Existing: Memory Retriever, Embedding Client

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Activation Interface                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Steering Vector Derivation Engine                    │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │                                                                      │    │
│  │  Method 1: Contrastive Direction Discovery (CDD)                     │    │
│  │    v = mean(h⁺) - mean(h⁻)   [concept vs. anti-concept]            │    │
│  │                                                                      │    │
│  │  Method 2: Mean-Centering / PCA                                      │    │
│  │    v = PCA_1(H_concept - H_baseline)  [principal direction]         │    │
│  │                                                                      │    │
│  │  Method 3: Identity V (Unembedding Rows)                             │    │
│  │    v = W_unembed[token_id]   [model's own semantic primes]          │    │
│  │                                                                      │    │
│  │  Method 4: Pre-trained Memory Encoder (Phase 6)                      │    │
│  │    v = HippocampalEncoder(memory_text)  [learned projection]        │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────┐   ┌─────────────────────────────────────────┐      │
│  │  Injection Scaler    │   │  Steering Vector Field (SVF)            │      │
│  │  ─────────────────  │   │  ─────────────────────────────────────  │      │
│  │  α = relevance *    │   │  Context-aware steering:                │      │
│  │      injection_str  │   │  v(h) = f(h, memory)                   │      │
│  │  norm preservation  │   │  Adapts direction to local geometry     │      │
│  │  cosine alignment   │   │  of the hidden state                   │      │
│  └─────────────────────┘   └─────────────────────────────────────────┘      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Layer Injection Manager                              │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  - Select optimal layers (middle third of model)                     │    │
│  │  - Register post_forward hooks via HookManager                       │    │
│  │  - Per-layer scaling (early layers: subtle, mid layers: strong)     │    │
│  │  - Safety guards: norm check, NaN trap, cosine divergence           │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Steering Vector Derivation Engine

### Description
Build the engine that converts memory content into steering vectors in the model's activation space. Multiple derivation methods are supported — the choice depends on available resources and the type of memory.

### Sub-task 1.1: Contrastive Direction Discovery (CDD)

**File:** `src/intrinsic/encoding/steering_vectors.py`

```python
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SteeringVector:
    """A computed steering vector with metadata."""
    vector: torch.Tensor          # [hidden_dim]
    layer_idx: int                # Target layer for injection
    method: str                   # "cdd", "pca", "identity_v", "encoder"
    concept: str                  # Human-readable description
    norm: float = 0.0
    source_memory_ids: List[str] = field(default_factory=list)


class ContrastiveDirectionDiscovery:
    """
    Derive steering vectors via Contrastive Direction Discovery (CDD).
    
    Algorithm:
    1. Collect positive examples (prompts containing the target concept)
    2. Collect negative examples (prompts without the concept)
    3. Run both sets through the model, capturing hidden states at target layer
    4. Compute: v = mean(h_positive) - mean(h_negative)
    5. Normalize to unit vector
    
    This produces a direction in activation space that represents the
    difference between "having the concept" and "not having the concept."
    
    Example:
    - Positive: "The user lives in Paris, France."
    - Negative: "The user lives in a city."
    - Result: vector encoding "Paris/France-ness"
    """

    def __init__(self, backend, tokenizer, num_contrast_pairs: int = 8):
        self._backend = backend
        self._tokenizer = tokenizer
        self._num_pairs = num_contrast_pairs
        self._captured_states: Dict[int, List[torch.Tensor]] = {}

    async def derive(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        target_layer: int,
        concept_name: str = "",
    ) -> SteeringVector:
        """
        Derive a steering vector from contrastive prompt pairs.
        
        positive_prompts: prompts that express the target concept
        negative_prompts: matched prompts without the concept
        target_layer: which layer to capture activations from
        """
        # Capture hidden states for positive examples
        pos_states = []
        for prompt in positive_prompts:
            h = await self._capture_hidden_state(prompt, target_layer)
            pos_states.append(h)

        # Capture hidden states for negative examples
        neg_states = []
        for prompt in negative_prompts:
            h = await self._capture_hidden_state(prompt, target_layer)
            neg_states.append(h)

        # Compute contrastive direction
        pos_mean = torch.stack(pos_states).mean(dim=0)  # [hidden_dim]
        neg_mean = torch.stack(neg_states).mean(dim=0)  # [hidden_dim]

        direction = pos_mean - neg_mean
        norm = torch.norm(direction).item()
        direction = direction / (norm + 1e-8)  # Normalize to unit vector

        return SteeringVector(
            vector=direction,
            layer_idx=target_layer,
            method="cdd",
            concept=concept_name,
            norm=norm,
        )

    async def _capture_hidden_state(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """Run a prompt through the model and capture hidden state at target layer."""
        captured = {}

        def _capture_hook(hidden_states: torch.Tensor, layer: int):
            # Take the mean across sequence positions (pool over tokens)
            captured["state"] = hidden_states.mean(dim=1).squeeze(0)  # [hidden_dim]
            return None  # Don't modify

        hook = self._backend.register_activation_hook(layer_idx, _capture_hook, "post_forward")
        
        try:
            # Tokenize and run forward pass
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._backend._device)
            with torch.no_grad():
                self._backend._model(**inputs)
        finally:
            hook.remove()

        return captured["state"].detach()

    async def derive_from_memory(
        self,
        memory_text: str,
        memory_type: str,
        target_layer: int,
    ) -> SteeringVector:
        """
        Automatically generate contrastive pairs from a memory record.
        
        For "User lives in Paris":
        - Positive: "The user lives in Paris, France. They are based in Paris."
        - Negative: "The user lives in a place. They are based somewhere."
        """
        # Generate contrastive pairs via template or LLM
        positive_prompts = self._generate_positive(memory_text, memory_type)
        negative_prompts = self._generate_negative(memory_text, memory_type)

        return await self.derive(
            positive_prompts, negative_prompts, target_layer,
            concept_name=memory_text[:50],
        )

    def _generate_positive(self, memory_text: str, memory_type: str) -> List[str]:
        """Generate positive prompts containing the concept."""
        templates = [
            f"Based on what I know: {memory_text}",
            f"Remembering that {memory_text}, the answer is",
            f"Given the fact: {memory_text}. Therefore,",
            f"I recall that {memory_text}.",
        ]
        return templates

    def _generate_negative(self, memory_text: str, memory_type: str) -> List[str]:
        """Generate negative prompts without the concept (generic baselines)."""
        templates = [
            "Based on what I know: something is true.",
            "Remembering that something happened, the answer is",
            "Given the fact: something is known. Therefore,",
            "I recall that something occurred.",
        ]
        return templates


class IdentityVDerivation:
    """
    Derive steering vectors using the model's own unembedding matrix rows ("Identity V").
    
    The unembedding matrix W_u has shape [vocab_size, hidden_dim].
    Each row W_u[token_id] represents the direction in hidden space that the model
    associates with producing that token. These are "semantic primes" — the model's
    own internal representation of each vocabulary entry.
    
    For a concept like "Paris", we can:
    1. Get the unembedding row for the token "Paris"
    2. Optionally combine with related tokens ("France", "French")
    3. Use this as a lightweight steering vector
    
    Advantages:
    - No forward passes needed (instant)
    - Uses the model's own representations
    - Naturally aligned with the model's output space
    
    Limitations:
    - Single tokens may be too narrow for complex concepts
    - Subword tokenization means multi-token concepts need aggregation
    """

    def __init__(self, backend, tokenizer):
        self._backend = backend
        self._tokenizer = tokenizer

    def derive(
        self,
        concept_tokens: List[str],
        target_layer: int,
        negative_tokens: Optional[List[str]] = None,
    ) -> SteeringVector:
        """
        Derive a steering vector from unembedding matrix rows.
        
        concept_tokens: ["Paris", "France"] — tokens representing the concept
        negative_tokens: optional contrasting tokens ["London", "England"]
        """
        unembed = self._backend.get_unembedding_matrix()
        if unembed is None:
            raise RuntimeError("Unembedding matrix not available")

        # Get concept direction
        pos_vecs = []
        for token in concept_tokens:
            ids = self._tokenizer.encode(token, add_special_tokens=False)
            for tid in ids:
                pos_vecs.append(unembed[tid])

        if not pos_vecs:
            raise ValueError(f"No valid tokens found for concepts: {concept_tokens}")

        pos_mean = torch.stack(pos_vecs).mean(dim=0)

        if negative_tokens:
            neg_vecs = []
            for token in negative_tokens:
                ids = self._tokenizer.encode(token, add_special_tokens=False)
                for tid in ids:
                    neg_vecs.append(unembed[tid])
            neg_mean = torch.stack(neg_vecs).mean(dim=0)
            direction = pos_mean - neg_mean
        else:
            direction = pos_mean

        norm = torch.norm(direction).item()
        direction = direction / (norm + 1e-8)

        return SteeringVector(
            vector=direction,
            layer_idx=target_layer,
            method="identity_v",
            concept=", ".join(concept_tokens),
            norm=norm,
        )


class MeanCenteringPCA:
    """
    Derive steering vectors using Mean-Centering + PCA.
    
    Algorithm:
    1. Collect hidden states for prompts containing the target concept
    2. Collect hidden states for baseline (neutral) prompts
    3. Compute difference: D = H_concept - H_baseline
    4. Run PCA on D, take first principal component
    5. This PC1 captures the direction of maximum variance
       attributable to the concept
    
    More robust than simple CDD when there's high variance
    in the prompt formulations.
    """

    def __init__(self, backend, tokenizer):
        self._backend = backend
        self._tokenizer = tokenizer

    async def derive(
        self,
        concept_prompts: List[str],
        baseline_prompts: List[str],
        target_layer: int,
        concept_name: str = "",
    ) -> SteeringVector:
        """Derive via PCA on the concept-baseline difference."""
        cdd = ContrastiveDirectionDiscovery(self._backend, self._tokenizer)

        # Collect states
        concept_states = []
        for prompt in concept_prompts:
            h = await cdd._capture_hidden_state(prompt, target_layer)
            concept_states.append(h)

        baseline_states = []
        for prompt in baseline_prompts:
            h = await cdd._capture_hidden_state(prompt, target_layer)
            baseline_states.append(h)

        # Compute difference matrix
        concept_matrix = torch.stack(concept_states)   # [N, hidden_dim]
        baseline_matrix = torch.stack(baseline_states)  # [M, hidden_dim]

        concept_centered = concept_matrix - concept_matrix.mean(dim=0)
        baseline_centered = baseline_matrix - baseline_matrix.mean(dim=0)

        # Difference: center concept states around baseline mean
        diff_matrix = concept_matrix - baseline_matrix.mean(dim=0)

        # PCA: SVD on the difference matrix
        U, S, Vh = torch.linalg.svd(diff_matrix, full_matrices=False)
        pc1 = Vh[0]  # First principal component [hidden_dim]

        # Ensure consistent sign (positive correlation with concept)
        concept_proj = (concept_matrix @ pc1).mean()
        baseline_proj = (baseline_matrix @ pc1).mean()
        if concept_proj < baseline_proj:
            pc1 = -pc1  # Flip if pointing away from concept

        norm = torch.norm(pc1).item()
        pc1 = pc1 / (norm + 1e-8)

        return SteeringVector(
            vector=pc1,
            layer_idx=target_layer,
            method="pca",
            concept=concept_name,
            norm=norm,
        )
```

---

## Task 2: Activation Injection Engine

### Description
The core injection mechanism: add steering vectors to the residual stream during the forward pass. Handles scaling, norm preservation, multi-layer injection, and composition of multiple memory vectors.

### Sub-task 2.1: `ActivationSteeringEngine`

**File:** `src/intrinsic/interfaces/activation.py`

```python
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from ..hooks.manager import HookManager
from ..backends.base import ModelBackend
from ..bus import MemoryVector

logger = logging.getLogger(__name__)


@dataclass
class InjectionConfig:
    """Per-layer injection configuration."""
    layer_idx: int
    strength: float = 1.0           # α multiplier
    norm_preservation: bool = True   # Scale vector to match hidden state norm
    injection_mode: str = "add"      # "add" | "project_and_add" | "slerp"
    apply_to: str = "all_tokens"     # "all_tokens" | "last_token" | "first_token"


class ActivationSteeringEngine:
    """
    Injects steering vectors into the model's residual stream.
    
    This is the primary Activation Interface implementation.
    
    The engine:
    1. Receives MemoryVectors with pre-computed steering vectors
    2. Composes multiple vectors (additive or attention-weighted)
    3. Scales for numerical stability (norm preservation)
    4. Registers hooks on target layers via HookManager
    5. During forward pass: adds scaled vectors to hidden states
    
    Injection formula:
        h'_l = h_l + α · norm_factor · Σ_i (r_i · v_i)
    
    Where:
    - h_l = original hidden state at layer l
    - α = injection strength
    - norm_factor = ||h_l|| / ||Σ v_i|| (norm preservation)
    - r_i = relevance score of memory i
    - v_i = steering vector for memory i
    """

    def __init__(
        self,
        hook_manager: HookManager,
        backend: ModelBackend,
        default_strength: float = 0.8,
        max_strength: float = 2.0,
        norm_preservation: bool = True,
    ):
        self._hook_manager = hook_manager
        self._backend = backend
        self._default_strength = default_strength
        self._max_strength = max_strength
        self._norm_preservation = norm_preservation
        
        # Active steering vectors per layer
        self._active_vectors: Dict[int, List[Tuple[torch.Tensor, float]]] = {}  # layer -> [(vector, weight)]
        self._hook_ids: List[str] = []  # Track registered hooks
        self._injection_count = 0

    def set_steering_vectors(
        self,
        memory_vectors: List[MemoryVector],
    ) -> None:
        """
        Configure the active steering vectors for the next forward pass.
        Called by the bus handler before generation.
        
        Each MemoryVector may target different layers.
        """
        self._active_vectors.clear()

        for mv in memory_vectors:
            if mv.steering_vector is None:
                continue

            for layer_idx in mv.target_layers:
                if layer_idx not in self._active_vectors:
                    self._active_vectors[layer_idx] = []

                weight = mv.relevance_score * mv.injection_strength * mv.decay_factor
                self._active_vectors[layer_idx].append(
                    (mv.steering_vector, weight)
                )

        logger.debug(
            f"Configured steering vectors: {sum(len(v) for v in self._active_vectors.values())} "
            f"vectors across {len(self._active_vectors)} layers"
        )

    def attach_hooks(self, layer_indices: Optional[List[int]] = None) -> None:
        """
        Register injection hooks on target layers.
        If no layers specified, uses the model's recommended injection layers.
        """
        if layer_indices is None:
            from ..inspector import ModelInspector
            inspector = ModelInspector(self._backend)
            layer_indices = inspector.recommended_injection_layers()

        for layer_idx in layer_indices:
            hook_id = self._hook_manager.register(
                layer_idx=layer_idx,
                hook_fn=self._create_injection_hook(layer_idx),
                hook_type="post_forward",
                priority=50,  # Middle priority (logit hooks are 100)
                name=f"activation_steering_L{layer_idx}",
            )
            self._hook_ids.append(hook_id)

        logger.info(f"Attached activation steering hooks on layers: {layer_indices}")

    def _create_injection_hook(self, target_layer: int) -> Callable:
        """Create the hook function for a specific layer."""

        def _inject(hidden_states: torch.Tensor, layer_idx: int) -> Optional[torch.Tensor]:
            """
            The actual injection function called during forward pass.
            
            hidden_states: [batch, seq_len, hidden_dim]
            """
            vectors_and_weights = self._active_vectors.get(target_layer, [])
            if not vectors_and_weights:
                return None  # No injection for this layer

            device = hidden_states.device
            dtype = hidden_states.dtype

            # Step 1: Compose all steering vectors for this layer
            composed = self._compose_vectors(vectors_and_weights, device, dtype)
            if composed is None:
                return None

            # Step 2: Scale for numerical stability
            scaled = self._scale_vector(
                composed, hidden_states, self._default_strength
            )

            # Step 3: Inject into hidden states
            # Add to all token positions (broadcast over seq_len)
            modified = hidden_states + scaled.unsqueeze(0).unsqueeze(0)
            # Shape: [batch, seq_len, hidden_dim] + [1, 1, hidden_dim] → broadcast

            self._injection_count += 1
            return modified

        return _inject

    def _compose_vectors(
        self,
        vectors_and_weights: List[Tuple[torch.Tensor, float]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Compose multiple steering vectors into a single injection vector.
        
        Strategy: Weighted sum
        v_composed = Σ_i (w_i · v_i) / Σ_i w_i
        
        This naturally handles composition: if two memories push in similar
        directions, they reinforce. If they conflict, they partially cancel.
        """
        if not vectors_and_weights:
            return None

        total_weight = 0.0
        composed = torch.zeros(vectors_and_weights[0][0].shape, device=device, dtype=dtype)

        for vector, weight in vectors_and_weights:
            composed += weight * vector.to(device=device, dtype=dtype)
            total_weight += abs(weight)

        if total_weight < 1e-8:
            return None

        composed = composed / total_weight  # Normalize by total weight
        return composed

    def _scale_vector(
        self,
        vector: torch.Tensor,
        hidden_states: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """
        Scale the steering vector for numerical stability.
        
        Norm preservation: scale vector so its norm matches
        a fraction (strength) of the hidden state norm.
        This prevents the injection from overwhelming the model's
        existing representations.
        
        Formula:
            scaled = α · (||h|| / ||v||) · v
        Where:
        - α = strength (0 to max_strength)
        - ||h|| = mean norm of hidden states across tokens
        - ||v|| = norm of the steering vector
        """
        strength = min(strength, self._max_strength)

        if self._norm_preservation:
            # Mean norm of hidden states: [batch, seq_len, hidden_dim] → scalar
            h_norm = hidden_states.norm(dim=-1).mean().item()
            v_norm = vector.norm().item()

            if v_norm > 1e-8:
                norm_factor = h_norm / v_norm
            else:
                return vector * 0  # Zero vector, nothing to inject

            return strength * norm_factor * vector
        else:
            return strength * vector

    def detach_hooks(self) -> None:
        """Remove all steering hooks."""
        for hook_id in self._hook_ids:
            self._hook_manager.unregister(hook_id)
        self._hook_ids.clear()
        self._active_vectors.clear()

    def clear_vectors(self) -> None:
        """Clear active vectors without removing hooks."""
        self._active_vectors.clear()

    def get_stats(self) -> Dict:
        return {
            "active_layers": list(self._active_vectors.keys()),
            "total_vectors": sum(len(v) for v in self._active_vectors.values()),
            "injection_count": self._injection_count,
            "hook_count": len(self._hook_ids),
        }
```

### Sub-task 2.2: Steering Vector Field (SVF) — Context-Aware Steering

```python
class SteeringVectorField:
    """
    Context-aware steering: the steering vector adapts based on 
    the current hidden state (local geometry).
    
    Standard steering uses a fixed vector v for all contexts.
    SVF computes v(h) = f(h, memory), where the steering direction
    is a function of the current hidden state.
    
    Implementation:
    v(h) = v_base + β · project(h, v_base) · v_orthogonal
    
    Where:
    - v_base = the pre-computed steering vector
    - project(h, v_base) = how aligned the current state is with the concept
    - v_orthogonal = component of h orthogonal to v_base
    - β = field strength parameter
    
    Intuition: If the model is already thinking about "Paris" (high projection),
    apply less steering (it's already there). If it's far from "Paris",
    steer more aggressively.
    """

    def __init__(self, field_strength: float = 0.3):
        self._field_strength = field_strength

    def compute_context_aware_vector(
        self,
        base_vector: torch.Tensor,   # [hidden_dim]
        hidden_state: torch.Tensor,   # [hidden_dim] (mean-pooled or last token)
    ) -> torch.Tensor:
        """
        Compute context-adapted steering vector.
        
        If the model's current state is already aligned with the memory concept,
        reduce steering (avoid over-correction). If misaligned, steer more.
        """
        # Project hidden state onto steering direction
        projection = torch.dot(hidden_state, base_vector) / (torch.norm(base_vector) ** 2 + 1e-8)

        # Adaptive scaling: less steering when already aligned
        # alignment ∈ [-1, 1] (cosine similarity)
        alignment = F.cosine_similarity(
            hidden_state.unsqueeze(0), base_vector.unsqueeze(0)
        ).item()

        # When alignment is high (already in-concept), reduce strength
        # When alignment is low (off-concept), increase strength
        adaptive_scale = 1.0 - abs(alignment) * self._field_strength

        return adaptive_scale * base_vector
```

---

## Task 3: Multi-Layer Injection Strategy

### Description
Research shows that different layers encode different types of information. The injection strategy must select which layers to target based on the type of memory being injected.

### Sub-task 3.1: Layer Selection Strategy

**File:** `src/intrinsic/interfaces/activation.py` (continued)

```python
class LayerSelectionStrategy:
    """
    Select optimal injection layers based on memory type and model architecture.
    
    Layer roles (for a 32-layer model):
    - Layers 0-7 (Early, 0-25%): Token-level, syntactic, positional
      → Inject: entity names, specific tokens, formatting
    
    - Layers 8-21 (Middle, 25-66%): Semantic, conceptual, relational
      → Inject: facts, preferences, knowledge, reasoning patterns
      → PRIMARY injection zone for most memories
    
    - Layers 22-31 (Late, 66-100%): Output formatting, next-token prediction
      → Inject: style preferences, tone, output format
      → Lighter touch — too deep can cause incoherence
    
    Research-backed:
    - Activation patching shows layers 12-20 (of 32) are most impactful for factual recall
    - Steering vector research uses layers 15-25 for behavioral steering
    - Too-early injection gets washed out; too-late injection causes artifacts
    """

    def __init__(self, num_layers: int):
        self._num_layers = num_layers

    def select_layers(self, memory_type: str, intensity: str = "normal") -> List[int]:
        """
        Select injection layers based on memory type.
        
        intensity: "light" (1-2 layers), "normal" (3-5 layers), "heavy" (5-8 layers)
        """
        n = self._num_layers
        
        # Define layer ranges by memory type
        LAYER_RANGES = {
            # Semantic facts → middle layers (strongest)
            "semantic_fact": (n // 4, 2 * n // 3),
            "knowledge": (n // 4, 2 * n // 3),
            
            # Preferences → mid-to-late layers (behavioral)
            "preference": (n // 3, 3 * n // 4),
            
            # Constraints → wide range (must be respected everywhere)
            "constraint": (n // 5, 3 * n // 4),
            
            # Episodic events → middle layers (narrative)
            "episodic_event": (n // 3, 2 * n // 3),
            
            # Procedures → mid-to-late (action planning)
            "procedure": (n // 3, 3 * n // 4),
            
            # Style/tone → late layers (output formatting)
            "style": (2 * n // 3, n - 1),
        }
        
        start, end = LAYER_RANGES.get(memory_type, (n // 3, 2 * n // 3))
        
        # Select layers based on intensity
        available = list(range(start, end + 1))
        if intensity == "light":
            count = min(2, len(available))
        elif intensity == "normal":
            count = min(4, len(available))
        elif intensity == "heavy":
            count = min(8, len(available))
        else:
            count = min(4, len(available))
        
        # Evenly space selected layers across the range
        if count >= len(available):
            return available
        
        step = len(available) / count
        selected = [available[int(i * step)] for i in range(count)]
        return selected

    def get_layer_weight(self, layer_idx: int, target_range: Tuple[int, int]) -> float:
        """
        Compute per-layer injection weight.
        Layers in the sweet spot of the range get higher weight.
        Uses a Gaussian centered on the middle of the range.
        """
        import math
        center = (target_range[0] + target_range[1]) / 2
        sigma = (target_range[1] - target_range[0]) / 4
        
        weight = math.exp(-0.5 * ((layer_idx - center) / sigma) ** 2)
        return weight
```

---

## Task 4: Memory Bus Integration

### Description
Wire the Activation Interface into the `IntrinsicMemoryBus` as a handler for the `ACTIVATION_STEERING` channel.

### Sub-task 4.1: Bus Handler

```python
# Integration code for activation interface

async def setup_activation_interface(bus, backend, hook_manager, tokenizer):
    """Wire the Activation Interface into the Memory Bus."""
    
    # Create steering engine
    engine = ActivationSteeringEngine(
        hook_manager=hook_manager,
        backend=backend,
        default_strength=0.8,
        norm_preservation=True,
    )
    
    # Attach hooks on recommended layers
    engine.attach_hooks()
    
    # Create derivation tools
    cdd = ContrastiveDirectionDiscovery(backend, tokenizer)
    identity_v = IdentityVDerivation(backend, tokenizer)
    layer_strategy = LayerSelectionStrategy(backend.get_model_spec().num_layers)
    
    async def _activation_handler(memory_vectors: List[MemoryVector]):
        """
        Bus handler: derive steering vectors for memories that don't have them,
        then configure the engine.
        """
        for mv in memory_vectors:
            if mv.steering_vector is None:
                # Derive steering vector on-the-fly
                # Use Identity V for speed (no forward passes needed)
                try:
                    sv = identity_v.derive(
                        concept_tokens=mv.source_text.split()[:3],
                        target_layer=layer_strategy.select_layers(
                            "semantic_fact", "normal"
                        )[0],
                    )
                    mv.steering_vector = sv.vector
                    mv.target_layers = layer_strategy.select_layers("semantic_fact")
                except Exception as e:
                    logger.warning(f"Failed to derive steering vector: {e}")
                    continue
        
        engine.set_steering_vectors(memory_vectors)
    
    bus.register_handler(InjectionChannel.ACTIVATION_STEERING, _activation_handler)
    
    return engine, cdd, identity_v
```

---

## Task 5: Steering Vector Cache & Pre-computation

### Description
Steering vector derivation (especially CDD) requires forward passes and is expensive. Build a cache that pre-computes and stores steering vectors for frequently accessed memories.

### Sub-task 5.1: `SteeringVectorCache`

**File:** `src/intrinsic/encoding/steering_vectors.py` (continued)

```python
import hashlib
import json
import os
from pathlib import Path


class SteeringVectorCache:
    """
    Cache pre-computed steering vectors to disk.
    
    Keyed by (memory_id, model_name, layer_idx, method).
    Avoids redundant forward passes for stable memories.
    
    Cache invalidation:
    - Memory content changes → invalidate
    - Model changes → invalidate all
    - Layer selection changes → invalidate affected
    """

    def __init__(self, cache_dir: str = ".cache/steering_vectors"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, SteeringVector] = {}  # In-memory LRU

    def _cache_key(
        self, memory_id: str, model_name: str, layer_idx: int, method: str
    ) -> str:
        raw = f"{memory_id}:{model_name}:{layer_idx}:{method}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(
        self, memory_id: str, model_name: str, layer_idx: int, method: str
    ) -> Optional[SteeringVector]:
        """Retrieve a cached steering vector."""
        key = self._cache_key(memory_id, model_name, layer_idx, method)

        # Check in-memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        cache_path = self._cache_dir / f"{key}.pt"
        if cache_path.exists():
            data = torch.load(cache_path, weights_only=True)
            sv = SteeringVector(
                vector=data["vector"],
                layer_idx=data["layer_idx"],
                method=data["method"],
                concept=data["concept"],
                norm=data["norm"],
            )
            self._memory_cache[key] = sv
            return sv

        return None

    def put(
        self, memory_id: str, model_name: str, sv: SteeringVector
    ) -> None:
        """Store a steering vector in cache."""
        key = self._cache_key(memory_id, model_name, sv.layer_idx, sv.method)
        self._memory_cache[key] = sv

        # Persist to disk
        cache_path = self._cache_dir / f"{key}.pt"
        torch.save({
            "vector": sv.vector,
            "layer_idx": sv.layer_idx,
            "method": sv.method,
            "concept": sv.concept,
            "norm": sv.norm,
        }, cache_path)

    def invalidate(self, memory_id: str) -> None:
        """Invalidate all cached vectors for a memory."""
        # Remove from in-memory cache
        keys_to_remove = [k for k, v in self._memory_cache.items() 
                          if memory_id in k]  # Simplified; real impl uses metadata
        for key in keys_to_remove:
            del self._memory_cache[key]

        # Remove from disk (scan for matching files)
        # In production: maintain a reverse index
        pass

    def invalidate_all(self) -> None:
        """Clear entire cache (e.g., after model change)."""
        self._memory_cache.clear()
        for f in self._cache_dir.glob("*.pt"):
            f.unlink()
```

---

## Acceptance Criteria

1. `ContrastiveDirectionDiscovery` produces meaningful steering vectors from contrastive prompts
2. `IdentityVDerivation` extracts unembedding rows and computes concept directions
3. `MeanCenteringPCA` produces principal direction from concept vs. baseline states
4. `ActivationSteeringEngine` injects vectors into hidden states with norm preservation
5. `SteeringVectorField` adapts injection based on current hidden state alignment
6. `LayerSelectionStrategy` maps memory types to appropriate layer ranges
7. `SteeringVectorCache` persists and retrieves vectors from disk
8. Safety guards prevent norm explosion and NaN propagation
9. Memory Bus handler for `ACTIVATION_STEERING` channel is functional
10. Multiple memories compose correctly via weighted vector summation

## Estimated Effort
- **Duration:** 3-4 weeks
- **Complexity:** High (requires careful numerical stability management)
- **Risk:** Medium (steering vector quality is model-dependent; needs empirical tuning)

## Testing Strategy
1. Unit test vector derivation methods with a small model (TinyLlama, Phi-3-mini)
2. Verify norm preservation: injected states have similar norms to originals
3. Behavioral test: inject "Paris" steering vector → model mentions Paris more
4. Ablation: vary injection strength and measure perplexity impact
5. Multi-vector composition test: verify two compatible vectors reinforce
6. Safety test: verify NaN/Inf recovery, norm explosion capping
