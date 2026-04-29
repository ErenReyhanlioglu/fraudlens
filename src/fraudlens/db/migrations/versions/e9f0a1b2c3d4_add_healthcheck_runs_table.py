"""add_healthcheck_runs_table

Revision ID: e9f0a1b2c3d4
Revises: c8a3f1d29b04
Create Date: 2026-04-29 00:00:00.000000

Adds a healthcheck_runs table that records the outcome of every *_healthcheck.py
probe run (pass/fail counts per section, elapsed time, tested transaction IDs).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e9f0a1b2c3d4"
down_revision: str | Sequence[str] | None = "c8a3f1d29b04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "healthcheck_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("script_name", sa.String(64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("elapsed_ms", sa.Float(), nullable=False),
        sa.Column("total_checks", sa.Integer(), nullable=False),
        sa.Column("passed_checks", sa.Integer(), nullable=False),
        sa.Column("failed_checks", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("all_passed", sa.Boolean(), nullable=False),
        sa.Column(
            "check_details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "transaction_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_healthcheck_runs_script_name", "healthcheck_runs", ["script_name"])


def downgrade() -> None:
    op.drop_index("ix_healthcheck_runs_script_name", table_name="healthcheck_runs")
    op.drop_table("healthcheck_runs")
