"""Temporal resolution: convert relative time references to absolute dates.

Temporal reasoning is the category with the highest variance across systems
(18.4% to 73.8%).  The key insight: every memory must carry absolute timestamps,
and relative time expressions must be resolved at extraction time using the
session timestamp as anchor.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Patterns for relative time references and their approximate offsets
_RELATIVE_PATTERNS: list[tuple[str, timedelta]] = [
    (r"\byesterday\b", timedelta(days=-1)),
    (r"\bthe day before yesterday\b", timedelta(days=-2)),
    (r"\btomorrow\b", timedelta(days=1)),
    (r"\btoday\b", timedelta(days=0)),
    (r"\blast night\b", timedelta(days=-1)),
    (r"\blast week\b", timedelta(weeks=-1)),
    (r"\blast month\b", timedelta(days=-30)),
    (r"\blast year\b", timedelta(days=-365)),
    (r"\bnext week\b", timedelta(weeks=1)),
    (r"\bnext month\b", timedelta(days=30)),
    (r"\bnext year\b", timedelta(days=365)),
    (r"\bthis morning\b", timedelta(days=0)),
    (r"\bthis afternoon\b", timedelta(days=0)),
    (r"\bthis evening\b", timedelta(days=0)),
    (r"\bthis week\b", timedelta(days=0)),
    (r"\bthis month\b", timedelta(days=0)),
    (r"\bthis year\b", timedelta(days=0)),
]

# Patterns with numeric extraction
_NUMERIC_PATTERNS: list[tuple[str, str]] = [
    (r"\b(\d+)\s+days?\s+ago\b", "days"),
    (r"\b(\d+)\s+weeks?\s+ago\b", "weeks"),
    (r"\b(\d+)\s+months?\s+ago\b", "months"),
    (r"\b(\d+)\s+years?\s+ago\b", "years"),
    (r"\b(\d+)\s+hours?\s+ago\b", "hours"),
    (r"\bin\s+(\d+)\s+days?\b", "days_future"),
    (r"\bin\s+(\d+)\s+weeks?\b", "weeks_future"),
    (r"\bin\s+(\d+)\s+months?\b", "months_future"),
]

# Vague patterns with approximate offsets
_VAGUE_PATTERNS: list[tuple[str, timedelta]] = [
    (r"\brecently\b", timedelta(days=-10)),
    (r"\ba while ago\b", timedelta(days=-60)),
    (r"\ba few days ago\b", timedelta(days=-3)),
    (r"\ba few weeks ago\b", timedelta(weeks=-3)),
    (r"\ba few months ago\b", timedelta(days=-90)),
    (r"\ba couple of days ago\b", timedelta(days=-2)),
    (r"\ba couple of weeks ago\b", timedelta(weeks=-2)),
    (r"\ba couple of months ago\b", timedelta(days=-60)),
    (r"\bearlier this week\b", timedelta(days=-3)),
    (r"\bearlier this month\b", timedelta(days=-15)),
    (r"\bearlier this year\b", timedelta(days=-120)),
    (r"\bthe other day\b", timedelta(days=-3)),
    (r"\bnot long ago\b", timedelta(days=-14)),
]


def resolve_temporal_references(
    text: str,
    session_date: datetime,
) -> list[dict[str, Any]]:
    """Extract and resolve temporal references in text relative to session_date.

    Returns a list of dicts with:
      - "original": the matched text span
      - "resolved_date": the absolute datetime
      - "approximate": whether the resolution is approximate (vague references)
    """
    results: list[dict[str, Any]] = []
    text_lower = text.lower()

    # Exact relative patterns
    for pattern, delta in _RELATIVE_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": session_date + delta,
                    "approximate": False,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # Numeric relative patterns
    for pattern, unit in _NUMERIC_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            count = int(match.group(1))
            is_future = unit.endswith("_future")
            base_unit = unit.replace("_future", "")
            sign = 1 if is_future else -1

            if base_unit == "days":
                delta = timedelta(days=sign * count)
            elif base_unit == "weeks":
                delta = timedelta(weeks=sign * count)
            elif base_unit == "months":
                delta = timedelta(days=sign * count * 30)
            elif base_unit == "years":
                delta = timedelta(days=sign * count * 365)
            elif base_unit == "hours":
                delta = timedelta(hours=sign * count)
            else:
                continue

            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": session_date + delta,
                    "approximate": base_unit in ("months", "years"),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # Vague patterns
    for pattern, delta in _VAGUE_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": session_date + delta,
                    "approximate": True,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # Deduplicate by span position (keep most specific match)
    results.sort(key=lambda r: (r["start"], -(r["end"] - r["start"])))
    deduped: list[dict[str, Any]] = []
    covered: set[int] = set()
    for r in results:
        span = set(range(r["start"], r["end"]))
        if not span & covered:
            covered.update(span)
            deduped.append(r)

    return deduped


def annotate_text_with_dates(
    text: str,
    session_date: datetime,
) -> str:
    """Return the text with inline date annotations for resolved temporal references.

    Example: "I went there yesterday" -> "I went there yesterday [2026-04-07]"
    """
    refs = resolve_temporal_references(text, session_date)
    if not refs:
        return text

    # Sort by position descending so we can insert without offset issues
    refs.sort(key=lambda r: r["start"], reverse=True)
    result = text
    for ref in refs:
        date_str = ref["resolved_date"].strftime("%Y-%m-%d")
        approx = "~" if ref["approximate"] else ""
        annotation = f" [{approx}{date_str}]"
        insert_pos = ref["end"]
        # Adjust for case differences between original and lower-cased match
        result = result[:insert_pos] + annotation + result[insert_pos:]

    return result


def extract_event_date(
    text: str,
    session_date: datetime,
) -> datetime | None:
    """Extract the most likely event date from text.

    Returns the resolved date of the first temporal reference found,
    or None if no temporal reference is detected.
    """
    refs = resolve_temporal_references(text, session_date)
    if refs:
        return refs[0]["resolved_date"]
    return None
