"""Serialization utilities for py-cml."""

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


def serialize_for_api(data: dict[str, Any]) -> dict[str, Any]:
    """Prepare a dict for API transmission.

    - Converts UUID to string
    - Converts datetime to ISO format
    - Removes None values
    - Recurses into nested dicts and lists (list elements: only recurse if dict)
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, UUID):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_for_api(value)
        elif isinstance(value, list):
            result[key] = [serialize_for_api(v) if isinstance(v, dict) else v for v in value]
        else:
            result[key] = value
    return result
