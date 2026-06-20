"""Helpers for parsing loose, LLM/JSON-sourced data.

Shared coercion utilities so extractors and the model-pack runtime do not each
carry their own copy. Pure stdlib — safe to import anywhere in ``src/``.
"""

from __future__ import annotations

from typing import Any, overload


def strip_markdown_fences(text: str) -> str:
    """Strip a leading/trailing markdown code fence (```` ```json `` … `` ``` ````) from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].startswith("```"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


@overload
def safe_float(value: Any, default: float) -> float: ...
@overload
def safe_float(value: Any, default: None = ...) -> float | None: ...
def safe_float(value: Any, default: float | None = None) -> float | None:
    """Coerce a loose value to ``float``, returning ``default`` on malformed input.

    LLMs and config/artifact metadata routinely emit non-numeric values (e.g.
    ``"high"``) where a float is expected; this guard avoids an unhandled raise.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@overload
def safe_int(value: Any, default: int) -> int: ...
@overload
def safe_int(value: Any, default: None = ...) -> int | None: ...
def safe_int(value: Any, default: int | None = None) -> int | None:
    """Coerce a loose value to ``int``, returning ``default`` on malformed input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
