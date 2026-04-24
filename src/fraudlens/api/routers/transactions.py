"""POST /transactions and GET /decisions/{id} endpoints."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from fraudlens.api.deps import extractor, scorer
from fraudlens.db.models import Decision
from fraudlens.db.session import get_db
from fraudlens.schemas.decision import (
    AgentType,
    DecisionOutcome,
    DecisionRead,
    RiskTier,
    TriageAction,
)
from fraudlens.schemas.transaction import TransactionRequest, TransactionResponse

router = APIRouter(tags=["transactions"])


# ---------------------------------------------------------------------------
# Triage routing thresholds (business rule, NOT AI)
# ---------------------------------------------------------------------------

_APPROVE_THRESHOLD = 0.3
_ESCALATE_THRESHOLD = 0.7


def _triage(prob: float) -> tuple[RiskTier, TriageAction]:
    if prob < _APPROVE_THRESHOLD:
        return RiskTier.LOW, TriageAction.APPROVE
    if prob < _ESCALATE_THRESHOLD:
        return RiskTier.MEDIUM, TriageAction.INVESTIGATE
    return RiskTier.HIGH, TriageAction.ESCALATE


# ---------------------------------------------------------------------------
# POST /transactions
# ---------------------------------------------------------------------------


@router.post("/transactions", response_model=TransactionResponse, status_code=201)
async def submit_transaction(
    payload: TransactionRequest,
    db: AsyncSession = Depends(get_db),  
    raw_mode: bool = Query(default=False, description="Bypass InferenceExtractor and score payload.raw_features directly"),  
) -> TransactionResponse:
    """Score a transaction and persist the decision.

    The XGBoost model runs synchronously in a thread executor (<50 ms).
    Set raw_mode=true and populate raw_features with IEEE-CIS feature values
    to bypass the InferenceExtractor (useful for demos and integration tests).
    """
    t0 = time.perf_counter()

    if raw_mode:
        if not payload.raw_features:
            raise HTTPException(
                status_code=422,
                detail="raw_features must be provided when raw_mode=true",
            )
        prob, shap_features = await scorer.score_raw_async(payload.raw_features)
    else:
        feature_row = extractor.transform(payload)
        prob, shap_features = await scorer.score_async(feature_row)
    risk_tier, triage_action = _triage(prob)

    # Immediate outcome for auto-approve; agents will overwrite for others.
    outcome = (
        DecisionOutcome.APPROVE
        if triage_action is TriageAction.APPROVE
        else DecisionOutcome.ESCALATE
        if triage_action is TriageAction.ESCALATE
        else DecisionOutcome.MANUAL_REVIEW
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    decision_id = uuid.uuid4()
    decision = Decision(
        id=decision_id,
        transaction_id=payload.transaction_id,
        fraud_probability=prob,
        risk_tier=risk_tier.value,
        triage_action=triage_action.value,
        outcome=outcome.value,
        shap_values={f.feature: f.contribution for f in shap_features},
        agent_used=AgentType.NONE.value,
        processing_time_ms=elapsed_ms,
        regulatory_citations=[],
    )
    db.add(decision)
    # session commits via get_db dependency on clean exit

    return TransactionResponse(
        transaction_id=payload.transaction_id,
        decision_id=decision_id,
        received_at=datetime.now(UTC),
        fraud_probability=prob,
        risk_tier=risk_tier.value,
        triage_action=triage_action.value,
        shap_top_features=shap_features,
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# GET /decisions/{id}
# ---------------------------------------------------------------------------


@router.get("/decisions/{decision_id}", response_model=DecisionRead)
async def get_decision(
    decision_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),  
) -> DecisionRead:
    """Fetch a previously created decision by its UUID."""
    result = await db.get(Decision, decision_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    return DecisionRead.model_validate(result)