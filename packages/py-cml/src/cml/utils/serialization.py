"""Serialization utilities for cognitive-memory-layer."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID


class CMLJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for CML types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _serialize_value(value: Any) -> Any:
    """PY-BUG-02: Recursively serialize UUID/datetime/dict/list for JSON."""
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return serialize_for_api(value)
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    return value


def serialize_for_api(data: dict[str, Any]) -> dict[str, Any]:
    """Prepare a dict for API transmission.

    - Converts UUID to string
    - Converts datetime to ISO format
    - Removes None values
    - Recurses into nested dicts and lists (including UUID/datetime inside lists)
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        result[key] = _serialize_value(value)
    return result
