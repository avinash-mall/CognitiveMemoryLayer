"""Add semantic_facts table.

Revision ID: 002
Revises: 001
Create Date: 2026-02-03

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "semantic_facts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("user_id", sa.String(100), nullable=False),
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
    )
    op.create_index(
        "ix_semantic_facts_tenant_user_key",
        "semantic_facts",
        ["tenant_id", "user_id", "key", "is_current"],
    )
    op.create_index(
        "ix_semantic_facts_tenant_user_category",
        "semantic_facts",
        ["tenant_id", "user_id", "category", "is_current"],
    )


def downgrade() -> None:
    op.drop_index("ix_semantic_facts_tenant_user_category", table_name="semantic_facts")
    op.drop_index("ix_semantic_facts_tenant_user_key", table_name="semantic_facts")
    op.drop_table("semantic_facts")
