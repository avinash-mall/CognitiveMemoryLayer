"""Prefrontal / Perirhinal Fine-Grained Cross-Encoder Discriminator.

Scores candidate (query, memory_text) pairs with full token-level cross-attention
using a GPU FP16 Cross-Encoder (default ``BAAI/bge-reranker-v2-m3``, auto-stepping
down to ``BAAI/bge-reranker-base`` or CPU/lexical-hybrid if CUDA free VRAM is tight
or when ``EMBEDDING_INTERNAL__PROVIDER=mock`` is active in unit tests).
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any

from ..utils.logging_config import get_logger
from ..utils.similarity import word_set

logger = get_logger(__name__)

_LOCK = threading.Lock()
_SINGLETON: CrossEncoderScorer | None = None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 40.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -40.0))
    return z / (1.0 + z)


def _lexical_overlap_score(query: str, text: str) -> float:
    """Fast deterministic fallback scorer when neural cross-encoder is disabled/mocked."""
    q_words = word_set(query)
    t_words = word_set(text)
    if not q_words or not t_words:
        return 0.0
    inter = len(q_words & t_words)
    recall = inter / len(q_words)
    jaccard = inter / len(q_words | t_words)
    return min(1.0, 0.72 * recall + 0.28 * jaccard)


class CrossEncoderScorer:
    """Thread-safe lazy Cross-Encoder scorer with VRAM-aware device placement."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.requested_model = model_name
        self.active_model_name: str | None = None
        self.device: str = "cpu"
        self._model: Any | None = None
        self._load_attempted = False
        self._load_lock = threading.Lock()

    @staticmethod
    def _is_mock_mode() -> bool:
        env_prov = os.environ.get("EMBEDDING_INTERNAL__PROVIDER", "").strip().lower()
        if env_prov == "mock":
            return True
        try:
            from ..core.config import get_settings

            s = get_settings()
            if (s.embedding_internal.provider or "").strip().lower() == "mock":
                return True
        except Exception:
            pass
        return False

    def _ensure_loaded(self) -> Any | None:
        if self._is_mock_mode():
            return None
        if self._load_attempted:
            return self._model
        with self._load_lock:
            if self._load_attempted:
                return self._model
            self._load_attempted = True
            try:
                import torch
                from sentence_transformers import CrossEncoder

                chosen_model = self.requested_model
                device = "cpu"
                automodel_args: dict[str, Any] = {}

                if torch.cuda.is_available():
                    free_bytes, _total_bytes = torch.cuda.mem_get_info()
                    # bge-reranker-v2-m3 FP16 needs ~1.15 GB; bge-reranker-base FP16 needs ~560 MB
                    if free_bytes >= 1_350_000_000:
                        device = "cuda"
                        automodel_args = {"torch_dtype": torch.float16}
                    elif free_bytes >= 650_000_000:
                        device = "cuda"
                        chosen_model = "BAAI/bge-reranker-base"
                        automodel_args = {"torch_dtype": torch.float16}
                    else:
                        device = "cpu"
                        chosen_model = "BAAI/bge-reranker-base"

                self._model = CrossEncoder(
                    chosen_model,
                    device=device,
                    max_length=512,
                    automodel_args=automodel_args,
                )
                self.active_model_name = chosen_model
                self.device = device
                logger.info(
                    "cross_encoder_loaded",
                    extra={"model": chosen_model, "device": device},
                )
            except Exception as exc:
                # If CUDA allocation failed due to external VRAM pressure, retry cleanly on CPU
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(
                        "BAAI/bge-reranker-base",
                        device="cpu",
                        max_length=512,
                    )
                    self.active_model_name = "BAAI/bge-reranker-base"
                    self.device = "cpu"
                    logger.info(
                        "cross_encoder_loaded_cpu_fallback",
                        extra={"model": "BAAI/bge-reranker-base", "reason": str(exc)[:120]},
                    )
                except Exception as cpu_exc:
                    logger.warning("cross_encoder_load_fallback", extra={"error": str(cpu_exc)})
                    self._model = None
            return self._model

    def score_pairs(
        self,
        query: str,
        texts: list[str],
        *,
        batch_size: int = 32,
        temperature: float = 1.0,
    ) -> list[float]:
        """Return calibrated [0.0, 1.0] cross-attention relevance scores for (query, text) pairs."""
        if not texts:
            return []
        if not query.strip():
            return [0.5] * len(texts)

        temp = max(0.25, float(temperature))
        model = self._ensure_loaded()
        if model is None:
            return [_lexical_overlap_score(query, t) for t in texts]

        try:
            import torch

            pairs = [(query[:512], (t or "")[:1024]) for t in texts]
            # Pass Identity activation so we get raw logits (sentence-transformers 5.x
            # applies Sigmoid by default, which would cause double-sigmoid squashing).
            raw_scores = model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                activation_fn=torch.nn.Identity(),
            )
            out: list[float] = []
            for s in raw_scores:
                val = float(s)
                out.append(_sigmoid(val / temp))
            return out
        except Exception as exc:
            logger.warning("cross_encoder_predict_fallback", extra={"error": str(exc)})
            return [_lexical_overlap_score(query, t) for t in texts]


def get_cross_encoder() -> CrossEncoderScorer:
    """Return process-wide singleton CrossEncoderScorer."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _LOCK:
        if _SINGLETON is None:
            model_name = "BAAI/bge-reranker-v2-m3"
            try:
                from ..core.config import get_settings

                model_name = getattr(get_settings().retrieval, "cross_encoder_model", model_name)
            except Exception:
                pass
            _SINGLETON = CrossEncoderScorer(model_name=model_name)
        return _SINGLETON
