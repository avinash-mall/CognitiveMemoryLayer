"""Initial schema: event_log and memory_records with pgvector.

Revision ID: 001
Revises:
Create Date: 2026-02-03

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID

# pgvector for embedding column and HNSW index
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # type: ignore

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Event log table
    op.create_table(
        "event_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("user_id", sa.String(100), nullable=False),
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
        "ix_event_log_tenant_user_time",
        "event_log",
        ["tenant_id", "user_id", "created_at"],
    )

    # Memory records table
    embedding_type = Vector(1536) if Vector else JSON
    op.create_table(
        "memory_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("user_id", sa.String(100), nullable=False),
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
    )

    op.create_index(
        "ix_memory_tenant_user_status",
        "memory_records",
        ["tenant_id", "user_id", "status"],
    )
    op.create_index(
        "ix_memory_tenant_user_type",
        "memory_records",
        ["tenant_id", "user_id", "type"],
    )
    op.create_index(
        "ix_memory_tenant_user_key",
        "memory_records",
        ["tenant_id", "user_id", "key"],
    )
    op.create_index(
        "ix_memory_content_hash",
        "memory_records",
        ["content_hash"],
    )

    # Vector index for similarity search (only if pgvector is available)
    if Vector is not None:
        op.execute(
            """
            CREATE INDEX ix_memory_embedding_hnsw
            ON memory_records
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )


def downgrade() -> None:
    op.drop_index("ix_memory_embedding_hnsw", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_content_hash", table_name="memory_records")
    op.drop_index("ix_memory_tenant_user_key", table_name="memory_records")
    op.drop_index("ix_memory_tenant_user_type", table_name="memory_records")
    op.drop_index("ix_memory_tenant_user_status", table_name="memory_records")
    op.drop_table("memory_records")
    op.drop_index("ix_event_log_tenant_user_time", table_name="event_log")
    op.drop_table("event_log")
