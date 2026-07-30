"""Text and vector similarity primitives.

One home for the two measures that were copy-pasted across the consolidation,
forgetting, and retrieval paths. Pure Python on purpose: these run on small
inputs inside request handlers, and pulling numpy in for a dot product would
cost more at import than it saves at call time.

The empty-input contract is the load-bearing part. Every copy returned ``0.0``
for empty or mismatched input rather than raising or returning ``1.0``, and
callers treat the result as "not similar" — so a change here silently alters
deduplication, clustering, and reranking rather than failing a test.
"""

from __future__ import annotations


def word_set(text: str) -> frozenset[str]:
    """Lowercased whitespace-split token set.

    Whitespace-split, not regex: punctuation stays attached to its word, so
    ``"food."`` and ``"food"`` are different tokens. That is the behaviour every
    caller was already getting.
    """
    return frozenset(text.lower().split())


def jaccard(a: str | frozenset[str], b: str | frozenset[str]) -> float:
    """Jaccard overlap of two token sets. Returns 0.0 if either side is empty.

    Accepts pre-computed token sets so a caller comparing one text against many
    can hoist ``word_set`` out of the loop (the reranker does this per query).
    """
    s1 = a if isinstance(a, frozenset) else word_set(a)
    s2 = b if isinstance(b, frozenset) else word_set(b)
    if not s1 or not s2:
        return 0.0
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 0.0


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns 0.0 when either vector is empty, the lengths disagree, or either
    has zero magnitude — all of which mean "no usable signal" to every caller.
    """
    if not v1 or len(v1) != len(v2):
        return 0.0
    dot = sum(x * y for x, y in zip(v1, v2, strict=True))
    norm1 = sum(x * x for x in v1) ** 0.5
    norm2 = sum(y * y for y in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
