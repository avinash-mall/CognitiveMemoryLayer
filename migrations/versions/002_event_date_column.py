"""Promote episodic event_date from metadata JSON to an indexed column.

Temporal resolution has been extracting *when the described event happened* — as
distinct from when the turn was written — since ``bb3a4a5``, and storing it only in
``memory_records.metadata['event_date']``. It was never a column, never indexed, and
read in exactly one place (``packet_builder``, for rendering). Nothing filtered, sorted
or ranged on it: the planner's time filter and ``vector_search`` both hit ``timestamp``,
which is turn time. So the system extracted event time correctly and then could not
query by it — the write-only bug class this codebase has now removed three times.

The backfill is the point of the migration. 23,463 of 549,582 rows carry an event_date
in metadata (4.3%), every one of them ISO-prefixed, and re-deriving them would mean
re-running the LLM extractor over the whole corpus.

Revision ID: 002
Revises: 001

"""

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memory_records", sa.Column("event_date", sa.DateTime(), nullable=True))

    # Values are written as ISO strings, both date-only ("2023-10-24") and full
    # ("2023-06-15T21:38:00"); ::timestamp accepts either. The regex guard skips
    # anything else rather than failing the migration on one malformed row — a cast
    # error here would abort the whole upgrade.
    op.execute(
        r"""
        UPDATE memory_records
           SET event_date = (metadata->>'event_date')::timestamp
         WHERE metadata->>'event_date' ~ '^\d{4}-\d{2}-\d{2}'
        """
    )

    # Composite with tenant_id because every read is tenant-scoped; a bare event_date
    # index would not be usable by the planner's filter.
    op.create_index(
        "ix_memory_tenant_event_date",
        "memory_records",
        ["tenant_id", "event_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_tenant_event_date", table_name="memory_records")
    op.drop_column("memory_records", "event_date")
