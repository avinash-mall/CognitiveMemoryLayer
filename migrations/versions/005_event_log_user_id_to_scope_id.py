"""Rename event_log.user_id to scope_id for API consistency.

Revision ID: 005
Revises: 004
Create Date: 2026-02-05

"""
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "event_log",
        "user_id",
        new_column_name="scope_id",
    )
    op.drop_index("ix_event_log_tenant_user_time", table_name="event_log", if_exists=True)
    op.create_index(
        "ix_event_log_tenant_scope_time",
        "event_log",
        ["tenant_id", "scope_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_event_log_tenant_scope_time", table_name="event_log", if_exists=True)
    op.create_index(
        "ix_event_log_tenant_user_time",
        "event_log",
        ["tenant_id", "scope_id", "created_at"],
    )
    op.alter_column(
        "event_log",
        "scope_id",
        new_column_name="user_id",
    )
