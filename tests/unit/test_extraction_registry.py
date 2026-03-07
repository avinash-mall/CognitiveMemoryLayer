from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.extraction.registry import ExtractorRegistry


def test_registry_register_get_list_and_create() -> None:
    registry = ExtractorRegistry()

    @registry.register("entity", "demo")
    class _Extractor:
        def __init__(self, *, value: int) -> None:
            self.value = value

    assert registry.get("entity", "demo") is _Extractor
    assert registry.list_registered("entity") == {"entity": ["demo"]}
    instance = registry.create("entity", "demo", value=7)
    assert isinstance(instance, _Extractor)
    assert instance.value == 7


def test_registry_logs_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ExtractorRegistry()
    warning = MagicMock()
    monkeypatch.setattr("src.extraction.registry.logger.warning", warning)

    @registry.register("fact", "shared")
    class _First:
        pass

    @registry.register("fact", "shared")
    class _Second:
        pass

    assert registry.get("fact", "shared") is _Second
    warning.assert_called_once()


def test_registry_create_missing_raises_key_error() -> None:
    registry = ExtractorRegistry()
    with pytest.raises(KeyError, match="No extractor registered"):
        registry.create("memory_type", "missing")
