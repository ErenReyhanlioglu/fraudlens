"""expand_decisions_table

Revision ID: b7e2d4f1a039
Revises: 21dc4b215a20
Create Date: 2026-04-25 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b7e2d4f1a039"
down_revision: str | Sequence[str] | None = "21dc4b215a20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("decisions", sa.Column("model_name", sa.String(length=64), nullable=True))
    op.add_column("decisions", sa.Column("decision_hint", sa.String(length=32), nullable=True))
    op.add_column("decisions", sa.Column("confidence", sa.Float(), nullable=True))
    op.add_column("decisions", sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"))
    op.add_column("decisions", sa.Column("red_flags", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"))
    op.add_column("decisions", sa.Column("tools_called", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"))
    op.add_column("decisions", sa.Column("tool_trace", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"))


def downgrade() -> None:
    op.drop_column("decisions", "tool_trace")
    op.drop_column("decisions", "tools_called")
    op.drop_column("decisions", "red_flags")
    op.drop_column("decisions", "evidence")
    op.drop_column("decisions", "confidence")
    op.drop_column("decisions", "decision_hint")
    op.drop_column("decisions", "model_name")
