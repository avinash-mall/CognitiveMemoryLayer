"""CML v3.0 Whole-Brain schema: Dentate Gyrus GIN FTS, Hyperedges, DMN Self, Procedural Skills.

Adds:
1. ``ix_memory_records_fts`` functional GIN index on ``memory_records`` for sub-5ms
   Dentate Gyrus sparse lexical pattern separation (BM25 / ``ts_rank_cd``).
2. ``hyperedge_engrams`` table for N-ary Cortical Hypergraph binding & CSR PageRank reload.
3. ``dmn_self_profiles`` table for Conway Tier-3 persistent Conceptual-Self schemas.
4. ``procedural_skills`` table for Basal Ganglia / Cerebellar skill playbooks & RPE weights.

Revision ID: 003
Revises: 002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Dentate Gyrus functional GIN full-text index on memory_records
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_memory_records_fts
        ON memory_records
        USING gin (to_tsvector('english', coalesce(text, '')))
        """
    )

    # 2. N-ary Cortical Hypergraph Engrams
    op.create_table(
        "hyperedge_engrams",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("engram_key", sa.String(200), nullable=False),
        sa.Column("actors", postgresql.JSON(), nullable=False),
        sa.Column("entities", postgresql.JSON(), nullable=False),
        sa.Column("event_summary", sa.Text(), nullable=False),
        sa.Column("temporal_anchor", sa.String(120), nullable=True),
        sa.Column("location", sa.String(200), nullable=True),
        sa.Column("causal_state", sa.Text(), nullable=True),
        sa.Column("prospective_cues", postgresql.JSON(), nullable=False),
        sa.Column("sensory_descriptors", postgresql.JSON(), nullable=False),
        sa.Column("episode_ids", postgresql.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True, server_default="0.82"),
        sa.Column("synaptic_weight", sa.Float(), nullable=True, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_hyperedge_engrams_tenant_id", "hyperedge_engrams", ["tenant_id"])
    op.create_index("ix_hyperedge_engrams_engram_key", "hyperedge_engrams", ["engram_key"])
    op.create_index("ix_hyperedge_tenant_key", "hyperedge_engrams", ["tenant_id", "engram_key"])

    # 3. Conway Tier-3 Persistent DMN Self Profiles
    op.create_table(
        "dmn_self_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("subject", sa.String(150), nullable=False),
        sa.Column("profile_json", postgresql.JSON(), nullable=False),
        sa.Column("evidence_count", sa.Integer(), nullable=True, server_default="1"),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_dmn_self_profiles_tenant_id", "dmn_self_profiles", ["tenant_id"])
    op.create_index("ix_dmn_self_profiles_subject", "dmn_self_profiles", ["subject"])
    op.create_index(
        "ix_dmn_self_tenant_subject",
        "dmn_self_profiles",
        ["tenant_id", "subject"],
        unique=True,
    )

    # 4. Basal Ganglia / Cerebellar Procedural Skills
    op.create_table(
        "procedural_skills",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("skill_key", sa.String(200), nullable=False),
        sa.Column("trigger_pattern", sa.Text(), nullable=False),
        sa.Column("steps", postgresql.JSON(), nullable=False),
        sa.Column("pitfalls", postgresql.JSON(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=True, server_default="1"),
        sa.Column("failure_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("synaptic_weight", sa.Float(), nullable=True, server_default="1.0"),
        sa.Column("embedding_json", postgresql.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_procedural_skills_tenant_id", "procedural_skills", ["tenant_id"])
    op.create_index("ix_procedural_skills_skill_key", "procedural_skills", ["skill_key"])
    op.create_index(
        "ix_procedural_skills_tenant_key",
        "procedural_skills",
        ["tenant_id", "skill_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("procedural_skills")
    op.drop_table("dmn_self_profiles")
    op.drop_table("hyperedge_engrams")
    op.execute("DROP INDEX IF EXISTS ix_memory_records_fts")
