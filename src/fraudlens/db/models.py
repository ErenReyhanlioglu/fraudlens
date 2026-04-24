"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from fraudlens.db.session import Base


class Decision(Base):
    """Persisted fraud decision for a single transaction.

    Created immediately after XGBoost scoring; `reasoning` and
    `regulatory_citations` are backfilled by the agent layer.
    """

    __tablename__ = "decisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    triage_action: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    shap_values: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    agent_used: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    regulatory_citations: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list
    )
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        onupdate=lambda: datetime.now(UTC),
    )