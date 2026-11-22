"""Phases I-1, I-3 & I-6: Model Access Layer & Residual-Stream Activation Steering.

Solves ActiveCML Risk #1 (Request concurrency and hook-state isolation):
PyTorch forward hooks are global to a model instance, so storing per-request
steering state on `self` or the module would contaminate concurrent requests.
Here, active steering vectors live in a `contextvars.ContextVar`, guaranteeing
strict per-coroutine/per-request isolation across concurrent async inferences.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActiveSteeringState:
    """Coroutine-local activation steering state read by PyTorch forward hooks."""

    steering_vector: list[float]
    scale: float
    target_layers: frozenset[int]


_ACTIVE_STEERING_VAR: contextvars.ContextVar[ActiveSteeringState | None] = contextvars.ContextVar(
    "cml_active_steering_state", default=None
)


class ActivationSteeringEngine:
    """Synthesizes steering vectors from retrieved memories and attaches isolated hooks."""

    def __init__(self, default_layers: list[int] | None = None) -> None:
        # Mid-to-upper transformer layers (e.g. layers 12, 16, 20) where semantic/factual
        # representations concentrate under the Linear Representation Hypothesis.
        self.default_layers = default_layers or [12, 16, 20]

    def synthesize_steering_vector(
        self,
        packet: Any,
        *,
        gate_strength: float = 0.7,
        target_dim: int | None = None,
    ) -> tuple[list[float], float, list[int]]:
        """Compute a confidence/relevance-weighted latent steering vector from memory embeddings."""
        weighted_vecs: list[tuple[float, list[float]]] = []
        for mem in getattr(packet, "all_memories", [])[:10]:
            rec = getattr(mem, "record", None)
            emb = getattr(rec, "embedding", None) if rec is not None else None
            if emb and isinstance(emb, list) and len(emb) > 0:
                rel = max(0.1, float(getattr(mem, "relevance_score", 0.5) or 0.5))
                conf = max(0.1, float(getattr(rec, "confidence", 0.5) or 0.5))
                meta = getattr(rec, "metadata", None) or {}
                somatic = meta.get("somatic_marker")
                arousal_boost = (
                    1.0 + 0.3 * float(somatic.get("arousal", 0.0) or 0.0)
                    if isinstance(somatic, dict)
                    else 1.0
                )
                w = rel * conf * arousal_boost
                weighted_vecs.append((w, emb))

        if not weighted_vecs:
            return [], 0.0, list(self.default_layers)

        dim = len(weighted_vecs[0][1])
        accum = [0.0] * dim
        total_w = 0.0
        for w, vec in weighted_vecs:
            if len(vec) != dim:
                continue
            total_w += w
            for i, val in enumerate(vec):
                accum[i] += w * float(val)

        if total_w <= 0:
            return [], 0.0, list(self.default_layers)

        centroid = [v / total_w for v in accum]
        norm = math.sqrt(sum(v * v for v in centroid))
        if norm > 1e-8:
            centroid = [v / norm for v in centroid]

        # Optional deterministic cyclic/orthogonal projection when target_dim != embedding dim
        if target_dim is not None and target_dim > 0 and target_dim != len(centroid):
            projected = [0.0] * target_dim
            src_len = len(centroid)
            for i in range(target_dim):
                sign = 1.0 if ((i // src_len) % 2 == 0) else -1.0
                projected[i] = sign * centroid[i % src_len]
            p_norm = math.sqrt(sum(v * v for v in projected))
            if p_norm > 1e-8:
                projected = [v / p_norm for v in projected]
            centroid = projected

        scale = round(min(1.5, 0.45 * gate_strength), 4)
        return centroid, scale, list(self.default_layers)

    @staticmethod
    def _resolve_model_layers(model: Any) -> list[Any]:
        """Introspect a HuggingFace / PyTorch transformer model to locate its decoder layers."""
        for path in (
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            ("layers",),
        ):
            curr = model
            for attr in path:
                curr = getattr(curr, attr, None)
                if curr is None:
                    break
            if curr is not None and hasattr(curr, "__len__") and hasattr(curr, "__getitem__"):
                return list(curr)
        return []

    @contextlib.contextmanager
    def steer_model(
        self,
        model: Any,
        steering_vector: list[float],
        *,
        scale: float = 0.3,
        layer_indices: list[int] | None = None,
    ) -> Iterator[None]:
        """Context manager attaching concurrency-safe residual-stream forward hooks to a PyTorch model."""
        if not steering_vector or scale <= 0.0 or model is None:
            yield
            return

        target_indices = frozenset(layer_indices or self.default_layers)
        state = ActiveSteeringState(
            steering_vector=steering_vector,
            scale=scale,
            target_layers=target_indices,
        )
        token = _ACTIVE_STEERING_VAR.set(state)
        handles: list[Any] = []

        try:
            layers = self._resolve_model_layers(model)
            for idx in target_indices:
                if 0 <= idx < len(layers) and hasattr(layers[idx], "register_forward_hook"):
                    layer_idx = idx

                    def _make_hook(l_idx: int):
                        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
                            active = _ACTIVE_STEERING_VAR.get()
                            if active is None or l_idx not in active.target_layers:
                                return output
                            try:
                                import torch

                                hidden = output[0] if isinstance(output, tuple) else output
                                if not isinstance(hidden, torch.Tensor):
                                    return output
                                d_model = hidden.shape[-1]
                                raw_vec = active.steering_vector
                                if len(raw_vec) != d_model:
                                    # Deterministic tile/truncate to match model hidden size
                                    tiled = [raw_vec[i % len(raw_vec)] for i in range(d_model)]
                                else:
                                    tiled = raw_vec
                                steer_t = torch.tensor(
                                    tiled, dtype=hidden.dtype, device=hidden.device
                                )
                                steer_t = steer_t / (steer_t.norm() + 1e-8)
                                steered = hidden + (active.scale * steer_t)
                                if isinstance(output, tuple):
                                    return (steered, *output[1:])
                                return steered
                            except Exception:
                                return output

                        return _hook

                    handles.append(layers[idx].register_forward_hook(_make_hook(layer_idx)))
            yield
        finally:
            for h in handles:
                with contextlib.suppress(Exception):
                    h.remove()
            _ACTIVE_STEERING_VAR.reset(token)
