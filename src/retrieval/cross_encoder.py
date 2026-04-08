"""Cross-encoder reranking for retrieved memories.

Cross-encoders attend jointly to query and document, catching relevance
signals that bi-encoders (embedding similarity) miss.  Expected: +3-7 F1
points overall.

Uses sentence-transformers CrossEncoder when available, falling back to
no-op when the model is not installed.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_cross_encoder_model = None
_cross_encoder_loading = False


def _get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3") -> Any:
    """Lazy-load the cross-encoder model (singleton)."""
    global _cross_encoder_model, _cross_encoder_loading

    if _cross_encoder_model is not None:
        return _cross_encoder_model

    if _cross_encoder_loading:
        return None

    _cross_encoder_loading = True
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder_model = CrossEncoder(model_name)
        logger.info("cross_encoder_loaded", model=model_name)
        return _cross_encoder_model
    except ImportError:
        logger.info(
            "cross_encoder_unavailable",
            reason="sentence-transformers not installed or model not found",
        )
        return None
    except Exception as exc:
        logger.warning("cross_encoder_load_failed", error=str(exc))
        return None
    finally:
        _cross_encoder_loading = False


def cross_encoder_rerank(
    query: str,
    documents: list[dict[str, Any]],
    text_key: str = "text",
    top_k: int | None = None,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> list[dict[str, Any]]:
    """Rerank documents using a cross-encoder model.

    Args:
        query: The search query.
        documents: List of result dicts, each having a text field.
        text_key: Key in each dict containing the document text.
        top_k: Max results to return (None = return all, re-sorted).
        model_name: HuggingFace model ID for the cross-encoder.

    Returns:
        Documents sorted by cross-encoder score (descending).
        Each doc gets a "cross_encoder_score" field added.
        Returns original list unchanged if model is unavailable.
    """
    if not documents or not query:
        return documents

    model = _get_cross_encoder(model_name)
    if model is None:
        return documents

    try:
        pairs = [[query, doc.get(text_key, "")] for doc in documents]
        scores = model.predict(pairs)

        for doc, score in zip(documents, scores, strict=True):
            doc["cross_encoder_score"] = float(score)

        documents.sort(key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True)

        if top_k is not None:
            return documents[:top_k]
        return documents
    except Exception as exc:
        logger.warning("cross_encoder_rerank_failed", error=str(exc))
        return documents
