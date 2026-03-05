"""Unit tests for SUMMARIZER_INTERNAL env settings."""

import pytest

from src.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings_cache(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_summarizer_internal_defaults(monkeypatch):
    monkeypatch.delenv("SUMMARIZER_INTERNAL__PROVIDER", raising=False)
    monkeypatch.delenv("SUMMARIZER_INTERNAL__MODEL", raising=False)
    monkeypatch.delenv("SUMMARIZER_INTERNAL__TASK", raising=False)

    settings = get_settings()
    assert settings.summarizer_internal.provider == "huggingface"
    assert settings.summarizer_internal.model == "Falconsai/text_summarization"
    assert settings.summarizer_internal.task == "summarization"


def test_summarizer_internal_model_from_env(monkeypatch):
    monkeypatch.setenv("SUMMARIZER_INTERNAL__PROVIDER", "huggingface")
    monkeypatch.setenv("SUMMARIZER_INTERNAL__MODEL", "google/pegasus-xsum")
    monkeypatch.setenv("SUMMARIZER_INTERNAL__TASK", "summarization")
    monkeypatch.setenv("SUMMARIZER_INTERNAL__MAX_LENGTH", "64")

    settings = get_settings()
    assert settings.summarizer_internal.provider == "huggingface"
    assert settings.summarizer_internal.model == "google/pegasus-xsum"
    assert settings.summarizer_internal.task == "summarization"
    assert settings.summarizer_internal.max_length == 64
