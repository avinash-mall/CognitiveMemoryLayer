"""Reciprocal Rank Fusion for merging ranked retrieval outputs.

Used to fuse the main retrieval pass with the HyDE pass in
`memory_retriever.retrieve`. This file used to also hold an in-memory BM25
index (`BM25Index`, `TenantBM25Manager`) built for a sparse second pathway
that was never wired into the retrieval plan — only the fusion step survived.
"""

from __future__ import annotations

from typing import Any


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
