"""Unit tests for Phase 4: neocortical schemas and schema manager."""

from src.memory.neocortical.schemas import (
    DEFAULT_FACT_SCHEMAS,
    FactCategory,
    FactSchema,
    SemanticFact,
)
from src.memory.neocortical.schema_manager import SchemaManager


class TestFactCategory:
    def test_identity_value(self):
        assert FactCategory.IDENTITY.value == "identity"

    def test_preference_value(self):
        assert FactCategory.PREFERENCE.value == "preference"


class TestFactSchema:
    def test_schema_has_category_and_key_pattern(self):
        schema = FactSchema(
            category=FactCategory.PREFERENCE,
            key_pattern="user:preference:cuisine",
            value_type="list",
        )
        assert schema.category == FactCategory.PREFERENCE
        assert schema.multi_valued is False

    def test_default_fact_schemas_has_identity(self):
        assert "user:identity:name" in DEFAULT_FACT_SCHEMAS
        assert DEFAULT_FACT_SCHEMAS["user:identity:name"].value_type == "string"


class TestSemanticFact:
    def test_semantic_fact_minimal(self):
        f = SemanticFact(
            id="f1",
            tenant_id="t1",
            context_tags=[],
            category=FactCategory.IDENTITY,
            key="user:identity:name",
            subject="user",
            predicate="name",
            value="Alice",
            value_type="str",
        )
        assert f.confidence == 0.8
        assert f.is_current is True


class TestSchemaManager:
    def test_get_schema_exact(self):
        mgr = SchemaManager()
        schema = mgr.get_schema("user:identity:name")
        assert schema is not None
        assert schema.category == FactCategory.IDENTITY

    def test_get_schema_wildcard(self):
        mgr = SchemaManager()
        schema = mgr.get_schema("user:relationship:spouse")
        assert schema is not None
        assert schema.category == FactCategory.RELATIONSHIP

    def test_get_schema_missing(self):
        mgr = SchemaManager()
        assert mgr.get_schema("unknown:key") is None

    def test_get_schemas_for_category(self):
        mgr = SchemaManager()
        schemas = mgr.get_schemas_for_category(FactCategory.PREFERENCE)
        assert len(schemas) >= 1
        assert any("preference" in k for k in schemas)

    def test_validate_key(self):
        mgr = SchemaManager()
        assert mgr.validate_key("user:identity:name") is True
        assert mgr.validate_key("user:relationship:alice") is True
        assert mgr.validate_key("random:key") is False
