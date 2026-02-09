"""Tests for logging configuration."""

import logging

from cml import configure_logging
from cml.utils.logging import logger


def test_configure_logging_sets_level() -> None:
    """configure_logging sets logger level."""
    configure_logging("DEBUG")
    assert logger.level == logging.DEBUG
    configure_logging("WARNING")
    assert logger.level == logging.WARNING
