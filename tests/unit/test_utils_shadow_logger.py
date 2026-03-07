from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.utils import shadow_logger


def test_compare_returns_heuristic_result_when_disabled() -> None:
    logger = shadow_logger.ShadowModeLogger(enabled=False)
    model_fn = MagicMock(return_value="model")

    result = logger.compare(
        component="retriever",
        task="rerank",
        heuristic_fn=lambda: "heuristic",
        model_fn=model_fn,
    )

    assert result == "heuristic"
    model_fn.assert_not_called()
    assert logger.get_summary()["total"] == 0


def test_compare_skips_sample_when_random_above_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = shadow_logger.ShadowModeLogger(enabled=True, sample_rate=0.1)
    monkeypatch.setattr(shadow_logger.random, "random", lambda: 0.9)
    model_fn = MagicMock(return_value="model")

    result = logger.compare(
        component="retriever",
        task="rerank",
        heuristic_fn=lambda: "heuristic",
        model_fn=model_fn,
    )

    assert result == "heuristic"
    model_fn.assert_not_called()
    assert logger.get_summary()["total"] == 0


def test_compare_records_agreement_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow_logger.random, "random", lambda: 0.0)
    logger = shadow_logger.ShadowModeLogger(enabled=True, sample_rate=1.0)

    result = logger.compare(
        component="classifier",
        task="intent",
        heuristic_fn=lambda: {"intent": "fact"},
        model_fn=lambda: {"intent": "fact"},
    )

    summary = logger.get_summary()
    assert result == {"intent": "fact"}
    assert summary["total"] == 1
    assert summary["agreed"] == 1
    assert summary["agreement_rate"] == 1.0
    assert summary["by_component"]["classifier:intent"]["agreement_rate"] == 1.0


def test_compare_logs_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow_logger.random, "random", lambda: 0.0)
    info = MagicMock()
    monkeypatch.setattr(shadow_logger.logger, "info", info)
    logger = shadow_logger.ShadowModeLogger(enabled=True, sample_rate=1.0)

    logger.compare(
        component="write_gate",
        task="importance",
        heuristic_fn=lambda: {"importance": 0.2},
        model_fn=lambda: {"importance": 0.9},
    )

    info.assert_called_once()
    assert logger.get_summary()["agreement_rate"] == 0.0


def test_compare_uses_custom_agreement_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow_logger.random, "random", lambda: 0.0)
    logger = shadow_logger.ShadowModeLogger(enabled=True, sample_rate=1.0)

    logger.compare(
        component="retriever",
        task="top-k",
        heuristic_fn=lambda: {"score": 0.91},
        model_fn=lambda: {"score": 0.95},
        agreement_fn=lambda left, right: abs(left["score"] - right["score"]) < 0.1,
    )

    assert logger.get_summary()["agreed"] == 1


def test_compare_handles_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow_logger.random, "random", lambda: 0.0)
    debug = MagicMock()
    monkeypatch.setattr(shadow_logger.logger, "debug", debug)
    logger = shadow_logger.ShadowModeLogger(enabled=True, sample_rate=1.0)

    logger.compare(
        component="extractor",
        task="facts",
        heuristic_fn=lambda: "ok",
        model_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    debug.assert_called_once()
    assert logger.get_summary()["total"] == 1


def test_safe_serialize_handles_nested_structures() -> None:
    class _Obj:
        def __str__(self) -> str:
            return "obj"

    assert shadow_logger._safe_serialize([1, {"x": _Obj()}]) == [1, {"x": "obj"}]


def test_reset_clears_comparisons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow_logger.random, "random", lambda: 0.0)
    logger = shadow_logger.ShadowModeLogger(enabled=True, sample_rate=1.0)
    logger.compare(
        component="retriever",
        task="rank",
        heuristic_fn=lambda: 1,
        model_fn=lambda: 1,
    )

    logger.reset()

    assert logger.get_summary() == {"total": 0, "agreement_rate": 0.0, "by_component": {}}


def test_get_shadow_logger_returns_cached_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow_logger, "_shadow_logger", None)
    first = shadow_logger.get_shadow_logger()
    second = shadow_logger.get_shadow_logger()
    assert first is second
