"""add_sar_report_column

Revision ID: c8a3f1d29b04
Revises: b7e2d4f1a039
Create Date: 2026-04-28 00:00:00.000000

Adds a nullable JSONB sar_report column populated only for escalated decisions.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c8a3f1d29b04"
down_revision: str | Sequence[str] | None = "b7e2d4f1a039"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "decisions",
        sa.Column("sar_report", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("decisions", "sar_report")
