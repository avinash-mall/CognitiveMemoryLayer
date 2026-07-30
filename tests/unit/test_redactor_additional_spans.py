"""Unit tests for use_llm_pii_redaction feature flag (LLM spans merged with regex)."""

from src.memory.hippocampal.redactor import PIIRedactor


def test_redactor_accepts_additional_spans():
    """PIIRedactor.redact accepts additional_spans and merges with regex."""
    redactor = PIIRedactor()
    text = "Contact me at user@example.com or call 555-123-4567"
    # additional_spans: [(start, end, pii_type), ...]
    # Simulate LLM-found span for a custom pattern
    additional = [(10, 25, "EMAIL")]
    result = redactor.redact(text, additional_spans=additional)
    assert result.has_redactions
    assert "[EMAIL_REDACTED]" in result.redacted_text or "REDACTED" in result.redacted_text


def test_redactor_without_additional_spans_unchanged_behavior():
    """Without additional_spans, redactor behaves as before."""
    redactor = PIIRedactor()
    text = "My email is test@example.com"
    result = redactor.redact(text)
    assert result.has_redactions
    assert "test@example.com" not in result.redacted_text
