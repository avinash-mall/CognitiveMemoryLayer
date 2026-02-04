"""Semantic fact and schema definitions for neocortical store."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class FactCategory(str, Enum):
    """Category of semantic fact."""

    IDENTITY = "identity"
    LOCATION = "location"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    OCCUPATION = "occupation"
    TEMPORAL = "temporal"
    ATTRIBUTE = "attribute"
    CUSTOM = "custom"


@dataclass
class FactSchema:
    """
    Schema definition for a type of fact.
    Defines validation rules and display properties.
    """

    category: FactCategory
    key_pattern: str
    value_type: str  # "string", "number", "date", "list", "object"
    required: bool = False
    multi_valued: bool = False
    temporal: bool = False
    validators: List[str] = field(default_factory=list)
    display_name: str = ""
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class SemanticFact:
    """A structured semantic fact in the neocortical store."""

    id: str
    tenant_id: str
    user_id: str
    category: FactCategory
    key: str
    subject: str
    predicate: str
    value: Any
    value_type: str
    confidence: float = 0.8
    evidence_count: int = 1
    evidence_ids: List[str] = field(default_factory=list)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    is_current: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    supersedes_id: Optional[str] = None


DEFAULT_FACT_SCHEMAS: Dict[str, FactSchema] = {
    "user:identity:name": FactSchema(
        category=FactCategory.IDENTITY,
        key_pattern="user:identity:name",
        value_type="string",
        display_name="User's Name",
        description="The user's preferred name",
        examples=["John", "Dr. Smith"],
    ),
    "user:location:current_city": FactSchema(
        category=FactCategory.LOCATION,
        key_pattern="user:location:current_city",
        value_type="string",
        temporal=True,
        display_name="Current City",
        description="Where the user currently lives",
        examples=["Paris", "New York"],
    ),
    "user:preference:cuisine": FactSchema(
        category=FactCategory.PREFERENCE,
        key_pattern="user:preference:cuisine",
        value_type="list",
        multi_valued=True,
        display_name="Food Preferences",
        description="Types of cuisine the user likes/dislikes",
        examples=["vegetarian", "Italian", "no seafood"],
    ),
    "user:relationship:*": FactSchema(
        category=FactCategory.RELATIONSHIP,
        key_pattern="user:relationship:{person}",
        value_type="object",
        display_name="Relationship",
        description="User's relationship with someone",
        examples=["spouse: Jane", "colleague: Bob"],
    ),
}
