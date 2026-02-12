"""Add dashboard_jobs table for consolidation/forgetting job history.

Revision ID: 002
Revises: 001
Create Date: 2026-02-12

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dashboard_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("job_type", sa.String(30), nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("user_id", sa.String(100), nullable=True),
        sa.Column("dry_run", sa.Boolean(), server_default="false"),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("result", JSON, nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_dashboard_jobs_tenant_type", "dashboard_jobs", ["tenant_id", "job_type"])
    op.create_index("ix_dashboard_jobs_status", "dashboard_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_dashboard_jobs_status", table_name="dashboard_jobs")
    op.drop_index("ix_dashboard_jobs_tenant_type", table_name="dashboard_jobs")
    op.drop_table("dashboard_jobs")
