"""Rename event_log.user_id to scope_id to match EventLogModel.

Revision ID: 005
Revises: 004
Create Date: 2026-02-06

"""
from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index(
        "ix_event_log_tenant_user_time",
        table_name="event_log",
        if_exists=True,
    )
    op.alter_column(
        "event_log",
        "user_id",
        new_column_name="scope_id",
        existing_type=sa.String(100),
    )
    op.create_index(
        "ix_event_log_tenant_scope_time",
        "event_log",
        ["tenant_id", "scope_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_event_log_tenant_scope_time",
        table_name="event_log",
        if_exists=True,
    )
    op.alter_column(
        "event_log",
        "scope_id",
        new_column_name="user_id",
        existing_type=sa.String(100),
    )
    op.create_index(
        "ix_event_log_tenant_user_time",
        "event_log",
        ["tenant_id", "user_id", "created_at"],
    )
