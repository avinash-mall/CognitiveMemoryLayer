"""Hippocampal Time Cells: granularity-aware temporal resolution & calendar intervals.

Resolves relative time expressions ("yesterday", "last Tuesday", "last month",
"three years ago", "last summer") against the utterance's session timestamp while
preserving true calendar granularity ("2023-05-07" vs "April 2023" vs "2022")
so downstream prompts never hallucinate fake day-of-month numbers on month/year spans.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_WORD_NUMBERS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _shift_months(dt: datetime, delta_months: int) -> tuple[datetime, str]:
    total_m = (dt.year * 12 + (dt.month - 1)) + delta_months
    new_year, new_m0 = divmod(total_m, 12)
    new_month = new_m0 + 1
    first_of_month = dt.replace(year=new_year, month=new_month, day=1)
    return first_of_month, f"{_MONTH_NAMES[new_month]} {new_year}"


def _shift_years(dt: datetime, delta_years: int) -> tuple[datetime, str]:
    new_year = dt.year + delta_years
    first_of_year = dt.replace(year=new_year, month=1, day=1)
    return first_of_year, str(new_year)


# Exact day-level relative patterns
_DAY_PATTERNS: list[tuple[str, int]] = [
    (r"\bthe day before yesterday\b", -2),
    (r"\byesterday\b", -1),
    (r"\blast night\b", -1),
    (r"\btomorrow\b", 1),
    (r"\btoday\b", 0),
    (r"\bthis morning\b", 0),
    (r"\bthis afternoon\b", 0),
    (r"\bthis evening\b", 0),
]

# Week-level relative patterns
_WEEK_PATTERNS: list[tuple[str, int]] = [
    (r"\blast week\b", -1),
    (r"\bnext week\b", 1),
    (r"\bthis week\b", 0),
]

# Month/Year relative patterns (calendar-exact, NOT -30/-365 day approximations)
_MONTH_REL_PATTERNS: list[tuple[str, int]] = [
    (r"\blast month\b", -1),
    (r"\bnext month\b", 1),
    (r"\bthis month\b", 0),
]

_YEAR_REL_PATTERNS: list[tuple[str, int]] = [
    (r"\blast year\b", -1),
    (r"\bnext year\b", 1),
    (r"\bthis year\b", 0),
]

_SEASON_RE = re.compile(r"\b(last|this|next)\s+(spring|summer|fall|autumn|winter)\b", re.IGNORECASE)
_SEASON_MONTH = {"spring": 4, "summer": 7, "fall": 10, "autumn": 10, "winter": 1}

_WEEKDAY_RE = re.compile(
    r"\b(last|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

_COUNT_REL_RE = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(days?|weeks?|months?|years?|hours?)\s+(ago|from now|later)\b",
    re.IGNORECASE,
)

_IN_FUTURE_RE = re.compile(
    r"\bin\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

# Vague patterns: kept in resolve_temporal_references with approximate=True & granularity="vague",
# but intentionally excluded from stamping exact event_date displays on episodes.
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


def _parse_num(tok: str) -> int:
    tl = tok.lower()
    if tl.isdigit():
        return int(tl)
    return _WORD_NUMBERS.get(tl, 1)


def resolve_temporal_references(
    text: str,
    session_date: datetime,
) -> list[dict[str, Any]]:
    """Extract and resolve temporal references relative to ``session_date``.

    Returns dicts containing:
      - ``original``: matched text span
      - ``resolved_date``: representative ``datetime``
      - ``approximate``: bool
      - ``granularity``: ``"day" | "week" | "month" | "season" | "year" | "vague"``
      - ``display``: human/LLM-accurate calendar string (no fake day on month/year!)
    """
    results: list[dict[str, Any]] = []
    text_lower = text.lower()

    # 1. Exact weekday ("last Saturday", "next Tuesday")
    for match in _WEEKDAY_RE.finditer(text_lower):
        direction = match.group(1).lower()
        target_wd = _WEEKDAYS[match.group(2).lower()]
        if direction == "last":
            diff = (session_date.weekday() - target_wd) % 7 or 7
            dt = session_date - timedelta(days=diff)
        else:
            diff = (target_wd - session_date.weekday()) % 7 or 7
            dt = session_date + timedelta(days=diff)
        results.append(
            {
                "original": match.group(0),
                "resolved_date": dt,
                "approximate": False,
                "granularity": "day",
                "display": dt.strftime("%Y-%m-%d"),
                "start": match.start(),
                "end": match.end(),
            }
        )

    # 2. Exact day offsets ("yesterday", "tomorrow", "today")
    for pattern, day_off in _DAY_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            dt = session_date + timedelta(days=day_off)
            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": dt,
                    "approximate": False,
                    "granularity": "day",
                    "display": dt.strftime("%Y-%m-%d"),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # 3. Week offsets ("last week", "next week")
    for pattern, wk_off in _WEEK_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            dt = session_date + timedelta(weeks=wk_off)
            disp = (
                f"week of {dt.strftime('%Y-%m-%d')}"
                if wk_off != 0
                else f"week of {session_date.strftime('%Y-%m-%d')}"
            )
            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": dt,
                    "approximate": True,
                    "granularity": "week",
                    "display": disp,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # 4. Calendar-exact month offsets ("last month", "next month", "this month")
    for pattern, m_off in _MONTH_REL_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            dt, disp = _shift_months(session_date, m_off)
            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": dt,
                    "approximate": False,
                    "granularity": "month",
                    "display": disp,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # 5. Calendar-exact year offsets ("last year", "next year", "this year")
    for pattern, y_off in _YEAR_REL_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            dt, disp = _shift_years(session_date, y_off)
            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": dt,
                    "approximate": False,
                    "granularity": "year",
                    "display": disp,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # 6. Seasons ("last summer", "next spring")
    for match in _SEASON_RE.finditer(text_lower):
        rel, season = match.group(1).lower(), match.group(2).lower()
        yr = session_date.year + (-1 if rel == "last" else (1 if rel == "next" else 0))
        m_num = _SEASON_MONTH[season]
        dt = session_date.replace(year=yr, month=m_num, day=1)
        results.append(
            {
                "original": match.group(0),
                "resolved_date": dt,
                "approximate": True,
                "granularity": "season",
                "display": f"{season.capitalize()} {yr}",
                "start": match.start(),
                "end": match.end(),
            }
        )

    # 7. Numeric / word-number expressions ("3 months ago", "two years ago", "5 days ago")
    for regex, default_dir in ((_COUNT_REL_RE, None), (_IN_FUTURE_RE, "future")):
        for match in regex.finditer(text_lower):
            count = _parse_num(match.group(1))
            unit = match.group(2).lower().rstrip("s")
            if default_dir == "future":
                sign = 1
            else:
                dir_tok = match.group(3).lower()
                sign = -1 if dir_tok == "ago" else 1

            if unit == "hour":
                dt = session_date + timedelta(hours=sign * count)
                gran, approx, disp = "day", False, dt.strftime("%Y-%m-%d")
            elif unit == "day":
                dt = session_date + timedelta(days=sign * count)
                gran, approx, disp = "day", False, dt.strftime("%Y-%m-%d")
            elif unit == "week":
                dt = session_date + timedelta(weeks=sign * count)
                gran, approx, disp = (
                    "week",
                    True,
                    f"{_MONTH_NAMES[dt.month]} {dt.year} (~{dt.strftime('%Y-%m-%d')})",
                )
            elif unit == "month":
                dt, disp = _shift_months(session_date, sign * count)
                gran, approx = "month", True
            elif unit == "year":
                dt, disp = _shift_years(session_date, sign * count)
                gran, approx = "year", True
            else:
                continue

            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": dt,
                    "approximate": approx,
                    "granularity": gran,
                    "display": disp,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # 8. Vague expressions (never used to stamp exact event_date_display)
    for pattern, delta in _VAGUE_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            dt = session_date + delta
            results.append(
                {
                    "original": match.group(0),
                    "resolved_date": dt,
                    "approximate": True,
                    "granularity": "vague",
                    "display": "",
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # Deduplicate overlapping spans (keep earliest, longest match)
    results.sort(key=lambda r: (r["start"], -(r["end"] - r["start"])))
    deduped: list[dict[str, Any]] = []
    covered: set[int] = set()
    for r in results:
        span = set(range(r["start"], r["end"]))
        if not span & covered:
            covered.update(span)
            deduped.append(r)

    return deduped


def extract_event_info(
    text: str,
    session_date: datetime,
) -> dict[str, Any] | None:
    """Return the first non-vague Hippocampal Time Cell resolution (date + granularity + display)."""
    for ref in resolve_temporal_references(text, session_date):
        if ref.get("granularity") != "vague":
            return ref
    return None
