"""BM25 sparse retrieval index for hybrid retrieval.

BM25 captures exact keyword matches that dense retrieval misses, particularly
for names and specific terms.  Combined with dense retrieval via Reciprocal
Rank Fusion (RRF), this yields +3-5 F1 points.

The index is built lazily per tenant and refreshed when the memory count
changes significantly.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]


@dataclass
class BM25Document:
    """A document in the BM25 index."""

    doc_id: str
    text: str
    tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """In-memory BM25 index for a single tenant's memories.

    Uses the Okapi BM25 algorithm with default parameters:
      k1 = 1.5 (term frequency saturation)
      b  = 0.75 (document length normalization)

    The index is intentionally simple and fast.  For production scale,
    consider replacing with rank_bm25 or a dedicated search engine.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._documents: list[BM25Document] = []
        self._doc_index: dict[str, int] = {}  # doc_id -> position
        self._df: dict[str, int] = {}  # term -> document frequency
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        self._n: int = 0
        self._built = False

    @property
    def size(self) -> int:
        return self._n

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document to the index. Call build() after adding all documents."""
        tokens = _tokenize(text)
        doc = BM25Document(
            doc_id=doc_id,
            text=text,
            tokens=tokens,
            metadata=metadata or {},
        )
        self._documents.append(doc)
        self._doc_index[doc_id] = len(self._documents) - 1
        self._built = False

    def build(self) -> None:
        """Build/rebuild the BM25 statistics. Must be called after adding documents."""
        self._n = len(self._documents)
        if self._n == 0:
            self._built = True
            return

        self._df = {}
        self._doc_lengths = []

        for doc in self._documents:
            self._doc_lengths.append(len(doc.tokens))
            seen_terms: set[str] = set()
            for token in doc.tokens:
                if token not in seen_terms:
                    self._df[token] = self._df.get(token, 0) + 1
                    seen_terms.add(token)

        self._avg_dl = sum(self._doc_lengths) / self._n if self._n > 0 else 1.0
        self._built = True

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Search the index with a query string. Returns ranked results."""
        if not self._built:
            self.build()
        if self._n == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        import math

        scores: list[float] = [0.0] * self._n

        for term in query_tokens:
            df = self._df.get(term, 0)
            if df == 0:
                continue

            # IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

            for i, doc in enumerate(self._documents):
                # Term frequency in this document
                tf = doc.tokens.count(term)
                if tf == 0:
                    continue

                dl = self._doc_lengths[i]
                # BM25 TF component
                tf_norm = (tf * (self._k1 + 1.0)) / (
                    tf + self._k1 * (1.0 - self._b + self._b * dl / self._avg_dl)
                )
                scores[i] += idf * tf_norm

        # Rank and return top_k
        scored = [(score, i) for i, score in enumerate(scores) if score > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, idx in scored[:top_k]:
            doc = self._documents[idx]
            results.append(
                {
                    "id": doc.doc_id,
                    "text": doc.text,
                    "score": score,
                    "metadata": doc.metadata,
                }
            )

        return results


def rrf_merge(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple retriever outputs.

    Each result_list is a ranked list of dicts. The `id_key` field
    identifies unique documents across lists.

    Args:
        result_lists: List of ranked result lists from different retrievers.
        k: RRF constant (default 60). Higher k reduces the impact of rank.
        id_key: Key in each result dict that uniquely identifies documents.

    Returns:
        Merged and re-ranked list of results.
    """
    scores: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_id = str(doc.get(id_key, id(doc)))
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "rrf_score": 0.0}
            scores[doc_id]["rrf_score"] += 1.0 / (k + rank + 1)

    merged = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )
    return [item["doc"] for item in merged]


class TenantBM25Manager:
    """Manages per-tenant BM25 indexes with lazy building and refresh."""

    def __init__(self) -> None:
        self._indexes: dict[str, BM25Index] = {}
        self._last_build: dict[str, float] = {}
        self._doc_counts: dict[str, int] = {}
        self._refresh_threshold = 0.2  # Rebuild if doc count changed >20%
        self._min_refresh_interval_s = 60.0

    def get_or_create(self, tenant_id: str) -> BM25Index:
        """Get the BM25 index for a tenant, creating if needed."""
        if tenant_id not in self._indexes:
            self._indexes[tenant_id] = BM25Index()
            self._doc_counts[tenant_id] = 0
        return self._indexes[tenant_id]

    def needs_refresh(self, tenant_id: str, current_count: int) -> bool:
        """Check if the tenant's index needs rebuilding."""
        if tenant_id not in self._last_build:
            return True

        elapsed = time.time() - self._last_build.get(tenant_id, 0)
        if elapsed < self._min_refresh_interval_s:
            return False

        old_count = self._doc_counts.get(tenant_id, 0)
        if old_count == 0:
            return current_count > 0

        change_ratio = abs(current_count - old_count) / old_count
        return change_ratio > self._refresh_threshold

    def mark_built(self, tenant_id: str) -> None:
        """Mark the index as freshly built."""
        self._last_build[tenant_id] = time.time()
        idx = self._indexes.get(tenant_id)
        if idx:
            self._doc_counts[tenant_id] = idx.size
