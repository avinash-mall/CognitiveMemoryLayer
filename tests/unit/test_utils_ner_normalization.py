"""Unit tests for deterministic NER normalization helpers."""

from src.utils.ner import extract_pii_spans, normalize_entity_name, normalize_scope_values


def test_normalize_entity_name_alias_location():
    assert normalize_entity_name("NYC", "LOCATION") == "new york city"
    assert normalize_entity_name("U.S.A.", "LOCATION") == "united states"


def test_normalize_entity_name_coreference():
    assert normalize_entity_name("I", "PERSON") == "user"
    assert normalize_entity_name("myself", "PERSON") == "user"


def test_normalize_scope_values_deduplicates_aliases():
    scopes = normalize_scope_values(["NYC", "New York City", "  new york  "])
    assert scopes == ["new york city"]


def test_extract_pii_spans_detects_regional_phone_and_address():
    text = "Call me at +1 (212) 555-9988 or meet at 221B Baker Street, London."
    spans = extract_pii_spans(text)
    labels = {span[2] for span in spans}
    assert "PHONE" in labels or "PHONE_INTL" in labels
    assert "ADDRESS_US" in labels or "ADDRESS_UK" in labels
