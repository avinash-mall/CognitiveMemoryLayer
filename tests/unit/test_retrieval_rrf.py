"""Unit tests for Reciprocal Rank Fusion.

`rrf_merge` is the only survivor of the removed BM25 index and had no test of
its own; it is live on the HyDE merge path in `memory_retriever.retrieve`.
"""

from src.retrieval.rrf import rrf_merge


def test_empty_input_returns_empty():
    assert rrf_merge([]) == []
    assert rrf_merge([[], []]) == []


def test_single_list_preserves_order():
    docs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert [d["id"] for d in rrf_merge([docs])] == ["a", "b", "c"]


def test_doc_ranked_by_both_lists_wins():
    # "b" is 2nd in one list and 1st in the other; "a" is 1st in only one.
    # RRF: b = 1/62 + 1/61 = 0.0325, a = 1/61 = 0.0164 -> b outranks a.
    merged = rrf_merge([[{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "c"}]])
    assert [d["id"] for d in merged] == ["b", "a", "c"]


def test_dedupes_by_id_key():
    merged = rrf_merge([[{"id": "a"}], [{"id": "a"}]])
    assert len(merged) == 1


def test_lower_k_amplifies_rank_difference():
    lists = [[{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "a"}]]
    # Symmetric input: both docs score identically at any k, so order is stable
    # and the merge must not drop either document.
    assert len(rrf_merge(lists, k=1)) == 2
    assert len(rrf_merge(lists, k=60)) == 2


def test_custom_id_key():
    merged = rrf_merge([[{"uid": "x"}, {"uid": "y"}]], id_key="uid")
    assert [d["uid"] for d in merged] == ["x", "y"]


def test_missing_id_key_treats_docs_as_distinct():
    # Falls back to id(doc), so two payload-identical dicts are not merged.
    merged = rrf_merge([[{"text": "same"}], [{"text": "same"}]])
    assert len(merged) == 2
