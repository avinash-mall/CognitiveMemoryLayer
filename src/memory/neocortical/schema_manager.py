"""Schema management for neocortical fact types."""

from .schemas import DEFAULT_FACT_SCHEMAS, FactCategory, FactSchema


class SchemaManager:
    """Manages fact schemas and validation for the neocortical store."""

    def __init__(self, schemas: dict[str, FactSchema] | None = None):
        self.schemas = schemas or DEFAULT_FACT_SCHEMAS

    def get_schema(self, key: str) -> FactSchema | None:
        """Get schema for a key (exact or wildcard match)."""
        if key in self.schemas:
            return self.schemas[key]
        for pattern, schema in self.schemas.items():
            if "*" in pattern:
                prefix = pattern.split("*")[0]
                if key.startswith(prefix):
                    return schema
        return None

    def get_schemas_for_category(self, category: FactCategory) -> dict[str, FactSchema]:
        """Return all schemas in a category."""
        return {k: v for k, v in self.schemas.items() if v.category == category}

    def register_schema(self, key_pattern: str, schema: FactSchema) -> None:
        """Register a new schema."""
        self.schemas[key_pattern] = schema

    def validate_key(self, key: str) -> bool:
        """Check if key matches any known schema pattern."""
        return self.get_schema(key) is not None
