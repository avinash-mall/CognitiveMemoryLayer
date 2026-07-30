"""Cosine similarity is valid on [-1, 1], so the vector-search default must not filter.

`min_similarity` used to default to 0.0, which silently discarded every
negatively-correlated row. That is invisible with embedding models whose vectors are
effectively non-negative (nomic-embed), but with hashed/mock embeddings — what CI and
offline test runs use — legitimate matches vanished from the result set, and a write/read
round trip returned fewer memories than were stored.
"""

from __future__ import annotations

import inspect

from src.storage.base import MemoryStoreBase
from src.storage.postgres import PostgresMemoryStore


def _default_min_similarity(fn) -> float:
    return inspect.signature(fn).parameters["min_similarity"].default


def test_vector_search_default_admits_negative_similarity() -> None:
    """The 'unset' default must sit at the bottom of the cosine range, not the middle."""
    for fn in (PostgresMemoryStore.vector_search, MemoryStoreBase.vector_search):
        default = _default_min_similarity(fn)
        assert default <= -1.0, (
            f"{fn.__qualname__} defaults min_similarity to {default}; anything above -1.0 "
            "silently drops negatively-correlated results"
        )


def test_similarity_floor_keeps_negatively_correlated_rows() -> None:
    """Mirror the store's filter predicate against a realistic mock-embedding spread."""
    default = _default_min_similarity(PostgresMemoryStore.vector_search)
    # Observed with hashed mock embeddings for the query "profession job career":
    similarities = [0.0962, -0.0946]
    kept = [s for s in similarities if s >= default]
    assert len(kept) == len(similarities), (
        f"floor {default} dropped {len(similarities) - len(kept)} of {len(similarities)} rows"
    )
