"""Logging configuration for cognitive-memory-layer."""

from __future__ import annotations

import logging

logger = logging.getLogger("cml")


def configure_logging(
    level: str = "WARNING",
    handler: logging.Handler | None = None,
) -> None:
    """Configure cognitive-memory-layer logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        handler: Custom handler. If not provided and logger has no handlers,
            a StreamHandler with a default formatter is added.

    Example:
        import cml
        cml.configure_logging("DEBUG")  # Enable debug logging
    """
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
    if handler is not None:
        logger.addHandler(handler)
    elif not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(h)


def _redact(value: str, visible_chars: int = 4) -> str:
    """Redact a secret, showing only the last N characters."""
    if len(value) <= visible_chars:
        return "***"
    return f"***{value[-visible_chars:]}"
