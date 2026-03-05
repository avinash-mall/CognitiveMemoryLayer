"""PII redaction before storage."""

import re
from dataclasses import dataclass

from ...utils.modelpack import get_modelpack_runtime
from ...utils.ner import extract_pii_spans


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
        "PHONE": r"(?<!\w)(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}(?!\w)",
        "PHONE_INTL": r"(?<!\w)\+\d{1,3}[\s.\-]?(?:\(?\d{1,4}\)?[\s.\-]?){2,5}\d{2,4}(?!\w)",
        "ADDRESS_US": (
            r"\b\d{1,6}[A-Za-z]?\s+[A-Za-z0-9][A-Za-z0-9\s\.\-]{1,50}?\s+"
            r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|"
            r"way|terrace|ter|place|pl|parkway|pkwy)\b"
            r"(?:,\s*[A-Za-z\.\-\s]+)?(?:,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
        ),
        "ADDRESS_UK": (
            r"\b\d{1,4}[A-Za-z]?\s+[A-Za-z][A-Za-z0-9\s\.\-]{1,50}?\s+"
            r"(?:road|rd|street|st|lane|ln|avenue|ave|close|cl|drive|dr)\b"
            r"(?:,\s*[A-Za-z\.\-\s]+)?(?:,\s*[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})?"
        ),
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
        self.modelpack = get_modelpack_runtime()

    def redact(
        self,
        text: str,
        additional_spans: list[tuple[int, int, str]] | None = None,
        include_ner: bool = False,
    ) -> RedactionResult:
        """Redact PII from text. additional_spans: [(start, end, pii_type), ...] from LLM."""
        matches: list[tuple[int, int, str, str]] = []  # start, end, pii_type, original
        for pii_type, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), pii_type, match.group()))

        # --- model path: merge model PII spans as union with regex baseline ---
        try:
            if getattr(self.modelpack, "has_task_model", lambda _: False)("pii_span_detection"):
                span_pred = self.modelpack.predict_spans("pii_span_detection", text)
                if span_pred is not None:
                    for start, end, pii_type in span_pred.spans:
                        if 0 <= start < end <= len(text):
                            matches.append((start, end, pii_type, text[start:end]))
        except Exception:
            pass

        if include_ner:
            for start, end, pii_type in extract_pii_spans(text):
                if 0 <= start < end <= len(text):
                    matches.append((start, end, pii_type, text[start:end]))
        if additional_spans:
            for start, end, pii_type in additional_spans:
                if 0 <= start < end <= len(text):
                    matches.append((start, end, pii_type, text[start:end]))

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
