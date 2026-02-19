"""Unit tests for chunker_require_llm feature flag (fail-fast when LLM required)."""

import pytest

from src.memory.working.manager import WorkingMemoryManager


def test_chunker_require_llm_raises_when_no_llm_and_fast_chunker_false():
    """When chunker_require_llm=true, use_fast_chunker=false, no llm_client -> raise."""
    with pytest.raises(ValueError, match="chunker_require_llm"):
        WorkingMemoryManager(
            llm_client=None,
            use_fast_chunker=False,
            chunker_require_llm=True,
        )


def test_chunker_require_llm_false_allows_rule_based_fallback():
    """When chunker_require_llm=false, no llm_client falls back to RuleBasedChunker."""
    mgr = WorkingMemoryManager(
        llm_client=None,
        use_fast_chunker=False,
        chunker_require_llm=False,
    )
    assert mgr._use_llm is False
    assert mgr.chunker is not None


def test_use_fast_chunker_true_never_requires_llm():
    """When use_fast_chunker=true, chunker_require_llm is irrelevant."""
    mgr = WorkingMemoryManager(
        llm_client=None,
        use_fast_chunker=True,
    )
    assert mgr._use_llm is False
