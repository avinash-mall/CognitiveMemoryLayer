"""Hippocampal Time Cell Engine: deterministic event timeline & duration deltas.

Prevents calendar-arithmetic errors in downstream LLMs on temporal reasoning queries
by sorting retrieved episodes on resolved event dates (falling back to spoken dates)
and pre-computing exact day/month/year intervals across anchored events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_iso_dt(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None)
    if not isinstance(raw, str) or len(raw.strip()) < 4:
        return None
    s = raw.strip()
    try:
        if len(s) == 4 and s.isdigit():
            return datetime(int(s), 1, 1)
        return datetime.fromisoformat(s[:19].replace("Z", ""))
    except ValueError:
        return None


def _format_delta(d1: datetime, d2: datetime) -> str:
    days = abs((d2.date() - d1.date()).days)
    if days == 0:
        return "same day"
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''}"
    if days < 60:
        wks = round(days / 7.0, 1)
        return f"{days} days (~{wks:g} weeks)"
    months = abs((d2.year - d1.year) * 12 + (d2.month - d1.month))
    if months < 24:
        return f"~{months} month{'s' if months != 1 else ''} ({days} days)"
    years = round(months / 12.0, 1)
    return f"~{years:g} years ({months} months)"


class HippocampalTimeCellEngine:
    """Synthesizes deterministic chronological ordering & duration helpers for MemoryPacket."""

    @staticmethod
    def build_timeline_block(episodes: list[Any], max_items: int = 8) -> str:
        """Render a compact deterministic Event Timeline & Span block for temporal queries."""
        anchored: list[tuple[datetime, str, str, str]] = []
        for ep in episodes:
            rec = getattr(ep, "record", ep)
            meta = getattr(rec, "metadata", None) or {}
            said_dt = _parse_iso_dt(getattr(rec, "timestamp", None))
            ev_dt = _parse_iso_dt(meta.get("event_date")) or said_dt
            if ev_dt is None:
                continue
            disp = str(meta.get("event_date_display") or "").strip()
            if not disp:
                ev_raw = str(meta.get("event_date") or "")[:10]
                disp = ev_raw if ev_raw else ev_dt.strftime("%Y-%m-%d")
            said_str = said_dt.strftime("%Y-%m-%d") if said_dt else ""
            label = (
                f"{disp} (spoken {said_str})"
                if said_str and disp != said_str
                else (disp or said_str)
            )
            spk = str(meta.get("speaker") or "").strip()
            txt = str(getattr(rec, "text", "") or "").strip().replace("\n", " ")[:110]
            anchored.append((ev_dt, label, spk, txt))

        if len(anchored) < 2:
            return ""

        anchored.sort(key=lambda x: x[0])
        selected = anchored[:max_items]
        lines = ["## Computed Event Timeline & Calendar Deltas\n"]
        for idx, (_dt, label, spk, txt) in enumerate(selected, start=1):
            spk_p = f"{spk}: " if spk else ""
            lines.append(f"{idx}. [{label}] {spk_p}{txt}\n")

        earliest_dt, earliest_lbl, _, _ = selected[0]
        latest_dt, latest_lbl, _, _ = selected[-1]
        if latest_dt > earliest_dt:
            span_txt = _format_delta(earliest_dt, latest_dt)
            lines.append(f"- Overall span ({earliest_lbl} -> {latest_lbl}): {span_txt}\n")
        return "".join(lines) + "\n"
