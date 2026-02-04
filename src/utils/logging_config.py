"""Structured logging configuration (Phase 10)."""
import logging
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structured logging for the application."""
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    level = getattr(logging, log_level.upper(), logging.INFO)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Get a configured structlog logger."""
    return structlog.get_logger(name)
