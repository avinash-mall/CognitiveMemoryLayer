"""PII redaction before storage."""

import re
from dataclasses import dataclass


@dataclass
class RedactionResult:
    original_text: str
    redacted_text: str
    redactions: list[tuple[str, str, int, int]]  # (type, original, start, end)
    has_redactions: bool


class PIIRedactor:
    """Redacts PII from text before storage."""

    PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(self, additional_patterns: dict | None = None) -> None:
        self.patterns = {**self.PATTERNS}
        if additional_patterns:
            self.patterns.update(additional_patterns)
        self._compiled = {
            name: re.compile(pattern, re.IGNORECASE) for name, pattern in self.patterns.items()
        }

    def redact(self, text: str) -> RedactionResult:
        matches: list[tuple[int, int, str, str]] = []  # start, end, pii_type, original
        for pii_type, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), pii_type, match.group()))

        # Merge overlapping ranges to avoid garbled output (LOW-11)
        if matches:
            matches.sort(key=lambda x: (x[0], -x[1]))
            merged: list[tuple[int, int, str, str]] = [matches[0]]
            for start, end, pii_type, original in matches[1:]:
                prev_start, prev_end, prev_type, _prev_orig = merged[-1]
                if start <= prev_end:
                    # Overlapping; extend the previous range
                    if end > prev_end:
                        merged[-1] = (prev_start, end, prev_type, text[prev_start:end])
                else:
                    merged.append((start, end, pii_type, original))
            matches = merged

        # Sort by start descending so we can replace without offset issues
        matches.sort(key=lambda x: x[0], reverse=True)
        redacted = text
        redactions: list[tuple[str, str, int, int]] = []
        for start, end, pii_type, original in matches:
            redactions.append((pii_type, original, start, end))
            redacted = redacted[:start] + f"[{pii_type}_REDACTED]" + redacted[end:]

        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            redactions=redactions,
            has_redactions=len(redactions) > 0,
        )
