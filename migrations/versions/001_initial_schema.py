"""Initial schema: event_log, memory_records, and semantic_facts with pgvector.

This is a consolidated migration combining all schema evolution into a single
initial migration for new project deployments.

Revision ID: 001
Revises:
Create Date: 2026-02-06

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID

# pgvector for embedding column and HNSW index
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # type: ignore

from src.core.config import get_settings

# Embedding vector dimension driven by EMBEDDING__DIMENSIONS in .env / config.
_EMBEDDING_DIM = get_settings().embedding.dimensions

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # -------------------------------------------------------------------------
    # event_log table
    # -------------------------------------------------------------------------
    op.create_table(
        "event_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("scope_id", sa.String(100), nullable=False),  # Renamed from user_id
        sa.Column("agent_id", sa.String(100), nullable=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("operation", sa.String(20), nullable=True),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("memory_ids", ARRAY(UUID(as_uuid=True)), server_default=sa.text("'{}'::uuid[]")),
        sa.Column("parent_event_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("ip_address", sa.String(50), nullable=True),
        sa.Column("user_agent", sa.String(500), nullable=True),
    )
    op.create_index(
        "ix_event_log_tenant_scope_time",
        "event_log",
        ["tenant_id", "scope_id", "created_at"],
    )

    # -------------------------------------------------------------------------
    # memory_records table (holistic tenant-based, with context_tags)
    # -------------------------------------------------------------------------
    embedding_type = Vector(_EMBEDDING_DIM) if Vector else JSON
    op.create_table(
        "memory_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("agent_id", sa.String(100), nullable=True),
        sa.Column("type", sa.String(30), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("key", sa.String(200), nullable=True),
        sa.Column("embedding", embedding_type, nullable=True),
        sa.Column("entities", JSON, server_default=sa.text("'[]'::json")),
        sa.Column("relations", JSON, server_default=sa.text("'[]'::json")),
        sa.Column("metadata", JSON, server_default=sa.text("'{}'::json")),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("written_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("valid_from", sa.DateTime(), nullable=True),
        sa.Column("valid_to", sa.DateTime(), nullable=True),
        sa.Column("confidence", sa.Float(), server_default="0.5"),
        sa.Column("importance", sa.Float(), server_default="0.5"),
        sa.Column("access_count", sa.Integer(), server_default="0"),
        sa.Column("last_accessed_at", sa.DateTime(), nullable=True),
        sa.Column("decay_rate", sa.Float(), server_default="0.01"),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("labile", sa.Boolean(), server_default="false"),
        sa.Column("provenance", JSON, nullable=False),
        sa.Column("version", sa.Integer(), server_default="1"),
        sa.Column("supersedes_id", UUID(as_uuid=True), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        # Holistic memory columns (from migrations 003-004)
        sa.Column("namespace", sa.String(100), nullable=True),
        sa.Column("context_tags", ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("source_session_id", sa.String(100), nullable=True),
    )

    # Indexes for memory_records
    op.create_index(
        "ix_memory_tenant_status",
        "memory_records",
        ["tenant_id", "status"],
    )
    op.create_index(
        "ix_memory_tenant_type",
        "memory_records",
        ["tenant_id", "type"],
    )
    op.create_index(
        "ix_memory_tenant_key",
        "memory_records",
        ["tenant_id", "key"],
    )
    op.create_index(
        "ix_memory_content_hash",
        "memory_records",
        ["content_hash"],
    )
    op.create_index(
        "ix_memory_tenant_namespace",
        "memory_records",
        ["tenant_id", "namespace", "status"],
    )
    op.create_index(
        "ix_memory_records_namespace",
        "memory_records",
        ["namespace"],
    )

    # GIN index for context_tags
    op.execute("""
        CREATE INDEX ix_memory_context_tags_gin
        ON memory_records
        USING GIN (context_tags)
    """)

    # Vector index for similarity search (only if pgvector is available)
    if Vector is not None:
        op.execute("""
            CREATE INDEX ix_memory_embedding_hnsw
            ON memory_records
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """)

    # -------------------------------------------------------------------------
    # semantic_facts table (holistic tenant-based, with context_tags)
    # -------------------------------------------------------------------------
    op.create_table(
        "semantic_facts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("category", sa.String(30), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("subject", sa.String(200), nullable=False),
        sa.Column("predicate", sa.String(200), nullable=False),
        sa.Column("value", JSON, nullable=False),
        sa.Column("value_type", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), server_default="0.8"),
        sa.Column("evidence_count", sa.Integer(), server_default="1"),
        sa.Column("evidence_ids", ARRAY(sa.String()), server_default=sa.text("'{}'")),
        sa.Column("valid_from", sa.DateTime(), nullable=True),
        sa.Column("valid_to", sa.DateTime(), nullable=True),
        sa.Column("is_current", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("version", sa.Integer(), server_default="1"),
        sa.Column("supersedes_id", UUID(as_uuid=True), nullable=True),
        # Holistic memory column (from migration 004)
        sa.Column("context_tags", ARRAY(sa.String()), nullable=False, server_default="{}"),
    )

    # Indexes for semantic_facts
    op.create_index(
        "ix_semantic_facts_tenant_key",
        "semantic_facts",
        ["tenant_id", "key", "is_current"],
    )
    op.create_index(
        "ix_semantic_facts_tenant_category",
        "semantic_facts",
        ["tenant_id", "category", "is_current"],
    )

    # GIN index for context_tags
    op.execute("""
        CREATE INDEX ix_semantic_facts_context_tags_gin
        ON semantic_facts
        USING GIN (context_tags)
    """)


def downgrade() -> None:
    # Drop semantic_facts
    op.drop_index("ix_semantic_facts_context_tags_gin", table_name="semantic_facts", if_exists=True)
    op.drop_index("ix_semantic_facts_tenant_category", table_name="semantic_facts")
    op.drop_index("ix_semantic_facts_tenant_key", table_name="semantic_facts")
    op.drop_table("semantic_facts")

    # Drop memory_records
    op.drop_index("ix_memory_embedding_hnsw", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_context_tags_gin", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_records_namespace", table_name="memory_records")
    op.drop_index("ix_memory_tenant_namespace", table_name="memory_records")
    op.drop_index("ix_memory_content_hash", table_name="memory_records")
    op.drop_index("ix_memory_tenant_key", table_name="memory_records")
    op.drop_index("ix_memory_tenant_type", table_name="memory_records")
    op.drop_index("ix_memory_tenant_status", table_name="memory_records")
    op.drop_table("memory_records")

    # Drop event_log
    op.drop_index("ix_event_log_tenant_scope_time", table_name="event_log")
    op.drop_table("event_log")
