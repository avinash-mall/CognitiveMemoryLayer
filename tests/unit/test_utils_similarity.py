"""Unit tests for the shared similarity primitives.

These pin the three axes that differed-or-could-differ between the four
copy-pasted implementations these replaced: tokenisation, casefolding, and the
empty-input return value. The empty-input case is the one that breaks silently —
callers read 0.0 as "not similar", so returning 1.0 or raising would change
dedup/clustering/reranking without failing anything obvious.
"""

import pytest

from src.utils.similarity import cosine_similarity, jaccard, word_set


class TestWordSet:
    def test_lowercases(self):
        assert word_set("Food HIKING") == frozenset({"food", "hiking"})

    def test_splits_on_whitespace_only_keeping_punctuation(self):
        # Deliberate: whitespace-split, not regex. "food." != "food".
        assert word_set("food. hiking") == frozenset({"food.", "hiking"})

    def test_collapses_duplicates(self):
        assert word_set("a a b") == frozenset({"a", "b"})

    def test_empty_text(self):
        assert word_set("") == frozenset()
        assert word_set("   ") == frozenset()


class TestJaccard:
    def test_identical_is_one(self):
        assert jaccard("vegan food", "vegan food") == 1.0

    def test_disjoint_is_zero(self):
        assert jaccard("vegan food", "kernel panic") == 0.0

    def test_partial_overlap(self):
        # {a,b} vs {b,c}: intersection 1, union 3
        assert jaccard("a b", "b c") == pytest.approx(1 / 3)

    def test_case_insensitive(self):
        assert jaccard("Vegan Food", "vegan food") == 1.0

    @pytest.mark.parametrize(("a", "b"), [("", "x"), ("x", ""), ("", "")])
    def test_empty_side_is_zero_not_one(self, a, b):
        assert jaccard(a, b) == 0.0

    def test_accepts_precomputed_frozensets(self):
        assert jaccard(word_set("a b"), word_set("b c")) == pytest.approx(1 / 3)

    def test_mixed_str_and_frozenset(self):
        assert jaccard("a b", word_set("b c")) == pytest.approx(1 / 3)


class TestCosineSimilarity:
    def test_identical_unit_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_is_negative(self):
        # Not clamped: callers that need [0,1] clamp themselves.
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_magnitude_invariant(self):
        assert cosine_similarity([3.0, 4.0], [6.0, 8.0]) == pytest.approx(1.0)

    def test_length_mismatch_is_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_empty_is_zero(self):
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0

    def test_zero_vector_is_zero_not_nan(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0
