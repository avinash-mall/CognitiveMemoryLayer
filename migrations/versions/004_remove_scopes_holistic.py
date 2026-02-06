"""Remove scopes; add context_tags and source_session_id (holistic memory).

Revision ID: 004
Revises: 003
Create Date: 2026-02-05

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSON

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ----- memory_records -----
    # Add new columns
    op.add_column(
        "memory_records",
        sa.Column("context_tags", ARRAY(sa.String()), nullable=False, server_default="{}"),
    )
    op.add_column(
        "memory_records",
        sa.Column("source_session_id", sa.String(100), nullable=True),
    )

    # Backfill: derive context_tags from scope, source_session_id from session_id or scope_id
    op.execute("""
        UPDATE memory_records
        SET context_tags = array_append(ARRAY[]::text[], scope)
        WHERE context_tags = '{}' OR context_tags IS NULL
    """)
    op.execute("""
        UPDATE memory_records
        SET source_session_id = COALESCE(session_id, scope_id)
        WHERE source_session_id IS NULL AND (session_id IS NOT NULL OR scope_id IS NOT NULL)
    """)

    # Drop old indexes that reference scope/scope_id
    op.drop_index("ix_memory_tenant_scope", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_tenant_scope_type", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_tenant_scope_key", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_records_scope_id", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_records_session_id", table_name="memory_records", if_exists=True)

    # Drop scope, scope_id, session_id. Also drop user_id if present (from 001) for holistic tenant-only.
    op.drop_column("memory_records", "scope")
    op.drop_column("memory_records", "scope_id")
    op.drop_column("memory_records", "session_id")
    conn_mem = op.get_bind()
    insp_mem = sa.inspect(conn_mem)
    mem_cols = [c["name"] for c in insp_mem.get_columns("memory_records")]
    if "user_id" in mem_cols:
        op.drop_column("memory_records", "user_id")

    # GIN index for context_tags
    op.execute("""
        CREATE INDEX ix_memory_context_tags_gin
        ON memory_records
        USING GIN (context_tags)
    """)
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

    # ----- semantic_facts -----
    # Add context_tags
    op.add_column(
        "semantic_facts",
        sa.Column("context_tags", ARRAY(sa.String()), nullable=False, server_default="{}"),
    )
    op.execute("""
        UPDATE semantic_facts
        SET context_tags = ARRAY['world']::text[]
        WHERE context_tags = '{}' OR context_tags IS NULL
    """)

    # Drop scope_id-based indexes (002 had user_id; if DB has scope_id from a prior change we drop it)
    op.drop_index("ix_semantic_facts_tenant_scope_key", table_name="semantic_facts", if_exists=True)
    op.drop_index("ix_semantic_facts_tenant_scope_category", table_name="semantic_facts", if_exists=True)
    op.drop_index("ix_semantic_facts_tenant_user_key", table_name="semantic_facts", if_exists=True)
    op.drop_index("ix_semantic_facts_tenant_user_category", table_name="semantic_facts", if_exists=True)

    # Rename or drop scope_id: 002 created user_id. Models use scope_id - so DB might have user_id.
    # Check and drop the column that partitions by user/scope (we use tenant only now).
    conn_sf = op.get_bind()
    insp_sf = sa.inspect(conn_sf)
    cols = [c["name"] for c in insp_sf.get_columns("semantic_facts")]
    if "scope_id" in cols:
        op.drop_column("semantic_facts", "scope_id")
    elif "user_id" in cols:
        op.drop_column("semantic_facts", "user_id")

    op.execute("""
        CREATE INDEX ix_semantic_facts_context_tags_gin
        ON semantic_facts
        USING GIN (context_tags)
    """)
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


def downgrade() -> None:
    # semantic_facts: restore scope_id/user_id and indexes
    op.add_column(
        "semantic_facts",
        sa.Column("scope_id", sa.String(100), nullable=False, server_default="default"),
    )
    op.drop_index("ix_semantic_facts_context_tags_gin", table_name="semantic_facts", if_exists=True)
    op.drop_index("ix_semantic_facts_tenant_key", table_name="semantic_facts", if_exists=True)
    op.drop_index("ix_semantic_facts_tenant_category", table_name="semantic_facts", if_exists=True)
    op.create_index(
        "ix_semantic_facts_tenant_scope_key",
        "semantic_facts",
        ["tenant_id", "scope_id", "key", "is_current"],
    )
    op.create_index(
        "ix_semantic_facts_tenant_scope_category",
        "semantic_facts",
        ["tenant_id", "scope_id", "category", "is_current"],
    )
    op.drop_column("semantic_facts", "context_tags")

    # memory_records: restore scope, scope_id, session_id
    op.add_column(
        "memory_records",
        sa.Column("scope", sa.String(20), nullable=False, server_default="session"),
    )
    op.add_column(
        "memory_records",
        sa.Column("scope_id", sa.String(100), nullable=False, server_default="default"),
    )
    op.add_column(
        "memory_records",
        sa.Column("session_id", sa.String(100), nullable=True),
    )
    op.execute("UPDATE memory_records SET scope_id = source_session_id WHERE source_session_id IS NOT NULL")
    op.drop_index("ix_memory_context_tags_gin", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_tenant_status", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_tenant_type", table_name="memory_records", if_exists=True)
    op.drop_index("ix_memory_tenant_key", table_name="memory_records", if_exists=True)
    op.create_index("ix_memory_tenant_scope", "memory_records", ["tenant_id", "scope", "scope_id", "status"])
    op.create_index("ix_memory_tenant_scope_type", "memory_records", ["tenant_id", "scope_id", "type"])
    op.create_index("ix_memory_tenant_scope_key", "memory_records", ["tenant_id", "scope_id", "key"])
    op.drop_column("memory_records", "context_tags")
    op.drop_column("memory_records", "source_session_id")
