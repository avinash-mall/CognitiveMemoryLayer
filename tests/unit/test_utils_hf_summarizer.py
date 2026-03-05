"""Unit tests for Hugging Face summarizer backend wrapper."""

import sys
import types

import pytest

from src.utils.hf_summarizer import HuggingFaceSummarizer


@pytest.mark.asyncio
async def test_hf_summarizer_uses_pipeline_when_available(monkeypatch):
    class _FakePipeline:
        def __call__(self, text: str, **kwargs):
            _ = text
            _ = kwargs
            return [{"summary_text": "condensed output"}]

    fake_transformers = types.SimpleNamespace(
        pipeline=lambda *args, **kwargs: _FakePipeline(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    summarizer = HuggingFaceSummarizer(model="Falconsai/text_summarization")
    out = await summarizer.summarize("x " * 500, max_chars=120)
    assert out == "condensed output"


@pytest.mark.asyncio
async def test_hf_summarizer_falls_back_to_truncation_on_load_error(monkeypatch):
    def _raise_pipeline(*args, **kwargs):
        _ = args
        _ = kwargs
        raise RuntimeError("load failed")

    fake_transformers = types.SimpleNamespace(pipeline=_raise_pipeline)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    summarizer = HuggingFaceSummarizer(model="Falconsai/text_summarization")
    text = "long text " * 100
    out = await summarizer.summarize(text, max_chars=80)
    assert out.endswith("...")
    assert len(out) <= 80
