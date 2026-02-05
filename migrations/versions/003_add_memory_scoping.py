"""Add memory scoping (scope, scope_id, session_id, namespace).

Revision ID: 003
Revises: 002
Create Date: 2026-02-04

"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to memory_records
    op.add_column(
        "memory_records",
        sa.Column("scope", sa.String(20), nullable=False, server_default="user"),
    )
    op.add_column(
        "memory_records",
        sa.Column("scope_id", sa.String(100), nullable=True),
    )
    op.add_column(
        "memory_records",
        sa.Column("session_id", sa.String(100), nullable=True),
    )
    op.add_column(
        "memory_records",
        sa.Column("namespace", sa.String(100), nullable=True),
    )

    # Backfill existing data: scope='user', scope_id=user_id
    op.execute(
        "UPDATE memory_records SET scope = 'user', scope_id = user_id WHERE scope_id IS NULL"
    )

    # Make user_id nullable
    op.alter_column(
        "memory_records",
        "user_id",
        existing_type=sa.String(100),
        nullable=True,
    )

    # Create new indexes
    op.create_index(
        "ix_memory_tenant_scope",
        "memory_records",
        ["tenant_id", "scope", "scope_id", "status"],
    )
    op.create_index(
        "ix_memory_tenant_session",
        "memory_records",
        ["tenant_id", "session_id", "status"],
    )
    op.create_index(
        "ix_memory_tenant_namespace",
        "memory_records",
        ["tenant_id", "namespace", "status"],
    )

    # Create index on scope_id for lookups (scope_id was added with index=True in model)
    op.create_index(
        "ix_memory_records_scope_id",
        "memory_records",
        ["scope_id"],
    )
    op.create_index(
        "ix_memory_records_session_id",
        "memory_records",
        ["session_id"],
    )
    op.create_index(
        "ix_memory_records_namespace",
        "memory_records",
        ["namespace"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_records_namespace", table_name="memory_records")
    op.drop_index("ix_memory_records_session_id", table_name="memory_records")
    op.drop_index("ix_memory_records_scope_id", table_name="memory_records")
    op.drop_index("ix_memory_tenant_namespace", table_name="memory_records")
    op.drop_index("ix_memory_tenant_session", table_name="memory_records")
    op.drop_index("ix_memory_tenant_scope", table_name="memory_records")

    op.alter_column(
        "memory_records",
        "user_id",
        existing_type=sa.String(100),
        nullable=False,
    )

    op.drop_column("memory_records", "namespace")
    op.drop_column("memory_records", "session_id")
    op.drop_column("memory_records", "scope_id")
    op.drop_column("memory_records", "scope")
