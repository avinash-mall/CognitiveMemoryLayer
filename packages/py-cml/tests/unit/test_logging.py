"""Tests for logging configuration."""

import logging

import pytest

from cml import configure_logging
from cml.utils.logging import _redact, logger


def test_configure_logging_sets_level() -> None:
    """configure_logging sets logger level."""
    configure_logging("DEBUG")
    assert logger.level == logging.DEBUG
    configure_logging("WARNING")
    assert logger.level == logging.WARNING


def test_configure_logging_adds_stream_handler_when_none() -> None:
    """configure_logging adds a StreamHandler when no handlers are present."""
    logger.handlers.clear()
    configure_logging("INFO")
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert logger.level == logging.INFO


def test_configure_logging_custom_handler() -> None:
    """configure_logging attaches the provided handler."""
    logger.handlers.clear()
    handler = logging.NullHandler()
    configure_logging("DEBUG", handler=handler)
    assert handler in logger.handlers


def test_redact_short_value_returns_stars() -> None:
    """_redact returns *** for values at or below visible_chars."""
    assert _redact("abc", visible_chars=4) == "***"
    assert _redact("", visible_chars=4) == "***"
    assert _redact("abcd", visible_chars=4) == "***"


@pytest.mark.parametrize(
    "value, visible, expected",
    [
        ("supersecretkey", 4, "***tkey"),
        ("sk-ant-1234567890", 6, "***567890"),
    ],
)
def test_redact_long_value_shows_suffix(value: str, visible: int, expected: str) -> None:
    """_redact shows only the last N characters for longer values."""
    assert _redact(value, visible_chars=visible) == expected
