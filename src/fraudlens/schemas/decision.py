"""Pydantic v2 schemas for fraud decisions and agent outputs."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TriageAction(StrEnum):
    APPROVE = "approve"
    INVESTIGATE = "investigate"
    ESCALATE = "escalate"


class AgentType(StrEnum):
    NONE = "none"
    INVESTIGATION = "investigation"  # Haiku — p 0.3–0.7
    CRITICAL = "critical"  # Sonnet — p ≥ 0.7


class DecisionOutcome(StrEnum):
    APPROVE = "approve"
    DECLINE = "decline"
    ESCALATE = "escalate"  # triggers SAR generation
    MANUAL_REVIEW = "manual_review"


class Regulatorycitation(BaseModel):
    """Single regulatory reference surfaced by RAG.

    `article` and `page` are both optional to accommodate different source types:
    PDF-backed RAG fills `page`, structured legal corpora can populate `article`.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    article: str | None = None
    page: int | None = None
    excerpt: str
    relevance_score: Annotated[float, Field(ge=0.0, le=1.0)]


class DecisionCreate(BaseModel):
    """Internal schema used to persist a new decision row."""

    transaction_id: uuid.UUID
    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    risk_tier: RiskTier
    triage_action: TriageAction
    outcome: DecisionOutcome
    shap_values: dict[str, float]
    agent_used: AgentType = AgentType.NONE
    reasoning: str | None = None
    regulatory_citations: list[Regulatorycitation] = Field(default_factory=list)
    processing_time_ms: float


class DecisionRead(BaseModel):
    """Full decision record returned from GET /decisions/{id}."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    transaction_id: uuid.UUID
    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    risk_tier: RiskTier
    triage_action: TriageAction
    outcome: DecisionOutcome
    shap_values: dict[str, float]
    agent_used: AgentType
    model_name: str | None
    decision_hint: str | None
    confidence: float | None
    reasoning: str | None
    evidence: list[str]
    red_flags: list[str]
    tools_called: list[str]
    tool_trace: list[dict]
    regulatory_citations: list[Regulatorycitation]
    sar_report: dict | None = None
    processing_time_ms: float
    created_at: datetime
    updated_at: datetime | None


class FraudDecision(BaseModel):
    """Synthesized fraud decision produced by the Decision Synthesizer.

    Combines the agent's `InvestigationResult` with the triage tier into a
    final outcome. Created deterministically (no LLM call) so the mapping is
    auditable and reproducible.
    """

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    outcome: DecisionOutcome
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    ml_score: Annotated[float, Field(ge=0.0, le=1.0)]
    agent_used: AgentType
    decision_hint: str
    evidence: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    regulatory_citations: list[Regulatorycitation] = Field(default_factory=list)
    reasoning: str
    tools_called: list[str] = Field(default_factory=list)
