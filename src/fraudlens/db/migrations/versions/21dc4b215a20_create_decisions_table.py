"""create_decisions_table

Revision ID: 21dc4b215a20
Revises:
Create Date: 2026-04-24 15:05:20.688091

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "21dc4b215a20"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "decisions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("transaction_id", sa.UUID(), nullable=False),
        sa.Column("fraud_probability", sa.Float(), nullable=False),
        sa.Column("risk_tier", sa.String(length=16), nullable=False),
        sa.Column("triage_action", sa.String(length=16), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column(
            "shap_values",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("agent_used", sa.String(length=32), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column(
            "regulatory_citations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("processing_time_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_decisions_transaction_id"),
        "decisions",
        ["transaction_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_decisions_transaction_id"), table_name="decisions")
    op.drop_table("decisions")
