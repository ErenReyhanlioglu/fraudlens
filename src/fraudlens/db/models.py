"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
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

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    triage_action: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    shap_values: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    agent_used: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Investigation result fields (null for auto-approved transactions)
    decision_hint: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    red_flags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    tools_called: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    tool_trace: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    regulatory_citations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # SAR report — only populated when outcome = ESCALATE; NULL otherwise.
    sar_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)

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


class HealthcheckRun(Base):
    """One healthcheck script execution — summary of all section results.

    Written by each *_healthcheck.py script at the end of every run so past
    probe history is queryable from the database.
    """

    __tablename__ = "healthcheck_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    script_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_checks: Mapped[int] = mapped_column(Integer, nullable=False)
    passed_checks: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_checks: Mapped[int] = mapped_column(Integer, nullable=False)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False)
    all_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    check_details: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    transaction_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
