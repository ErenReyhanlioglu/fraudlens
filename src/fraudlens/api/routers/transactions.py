"""POST /transactions and GET /decisions/{id} endpoints."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from fraudlens.agents.graph import run_fraud_investigation
from fraudlens.api.deps import extractor, scorer
from fraudlens.core.config import Settings, get_settings
from fraudlens.core.job_store import create_job, update_job
from fraudlens.db.models import Decision
from fraudlens.db.session import get_db
from fraudlens.ml.feature_extractor import enrich_with_context
from fraudlens.ml.shap_vocab import annotate_shap
from fraudlens.schemas.decision import (
    AgentType,
    DecisionOutcome,
    DecisionRead,
    RiskTier,
    TriageAction,
)
from fraudlens.schemas.transaction import TransactionRequest, TransactionResponse

logger = structlog.get_logger(__name__)

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
# Submit response schema
# ---------------------------------------------------------------------------


class TransactionSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Background pipeline task
# ---------------------------------------------------------------------------


async def _run_pipeline_background(
    job_id: str,
    payload: TransactionRequest,
    raw_mode: bool,
    settings: Settings,
) -> None:
    """Execute the full fraud pipeline and write progress to Redis."""
    from fraudlens.db.session import AsyncSessionFactory  # noqa: PLC0415

    t0 = time.perf_counter()

    try:
        await update_job(job_id, status="scoring", stage_label="Running XGBoost fraud scorer")

        if raw_mode:
            prob, shap_features = await scorer.score_raw_async(payload.raw_features)
        else:
            feature_row = extractor.transform(payload)
            prob, shap_features = await scorer.score_async(feature_row)
        risk_tier, triage_action = _triage(prob)

        await update_job(job_id, status="routing", stage_label="Determining risk tier")

        outcome = (
            DecisionOutcome.APPROVE
            if triage_action is TriageAction.APPROVE
            else DecisionOutcome.ESCALATE
            if triage_action is TriageAction.ESCALATE
            else DecisionOutcome.MANUAL_REVIEW
        )

        shap_dict = {f.feature: f.contribution for f in shap_features}
        decision_id = uuid.uuid4()

        investigation_result = None
        fraud_decision = None
        sar_report = None
        token_usage = None

        if triage_action in (TriageAction.INVESTIGATE, TriageAction.ESCALATE):
            agent_type = (
                AgentType.INVESTIGATION
                if triage_action is TriageAction.INVESTIGATE
                else AgentType.CRITICAL
            )

            shap_signals = annotate_shap(
                [{"feature": f.feature, "shap": f.contribution} for f in shap_features]
            )
            if raw_mode and payload.raw_features:
                banking_context = enrich_with_context(payload.raw_features)
            else:
                banking_context: dict[str, Any] = {
                    "amount": payload.amount,
                    "timestamp": payload.timestamp,
                    "transaction_type": payload.transaction_type,
                    "sender_account_id": payload.sender_account_id,
                    "merchant_id": payload.merchant_id,
                    "ip_address": payload.ip_address,
                    "device_fingerprint": payload.device_fingerprint,
                    "sender_country": payload.sender_country,
                    "receiver_country": payload.receiver_country,
                    "currency": payload.currency,
                    "channel": payload.channel,
                }
            agent_context = {
                **banking_context,
                "ml_score": prob,
                "shap_signals": shap_signals,
                "transaction_id": str(payload.transaction_id),
            }

            if triage_action is TriageAction.INVESTIGATE:
                await update_job(
                    job_id,
                    status="investigating",
                    stage_label="Investigation Agent active",
                )
            else:
                await update_job(
                    job_id,
                    status="critical",
                    stage_label="Critical Agent active — deep investigation",
                )

            try:
                state = await run_fraud_investigation(
                    transaction_id=str(payload.transaction_id),
                    fraud_probability=prob,
                    shap_values=shap_dict,
                    transaction_context=agent_context,
                    triage_action=triage_action.value,
                    job_id=job_id,
                )
                investigation_result = state.get("investigation_result")
                fraud_decision = state.get("fraud_decision")
                sar_report = state.get("sar_report")
                token_usage = state.get("token_usage")
            except Exception:
                logger.exception("agent_dispatch_failed", transaction_id=str(payload.transaction_id))

            await update_job(
                job_id,
                status="synthesizing",
                stage_label="Synthesizing final decision",
            )

            if fraud_decision is not None and fraud_decision.outcome is DecisionOutcome.ESCALATE:
                await update_job(
                    job_id,
                    status="sar",
                    stage_label="Generating SAR report",
                )
        else:
            agent_type = AgentType.NONE

        processing_time_ms = (time.perf_counter() - t0) * 1000

        full_response = TransactionResponse(
            transaction_id=payload.transaction_id,
            decision_id=decision_id,
            received_at=datetime.now(UTC),
            fraud_probability=prob,
            risk_tier=risk_tier.value,
            triage_action=triage_action.value,
            shap_top_features=shap_features,
            processing_time_ms=processing_time_ms,
            investigation=investigation_result,
            fraud_decision=fraud_decision,
            sar_report=sar_report,
            token_usage=token_usage,
        )

        async with AsyncSessionFactory() as db:
            decision = Decision(
                id=decision_id,
                transaction_id=payload.transaction_id,
                fraud_probability=prob,
                risk_tier=risk_tier.value,
                triage_action=triage_action.value,
                outcome=outcome.value,
                shap_values=shap_dict,
                agent_used=agent_type.value,
                processing_time_ms=processing_time_ms,
                regulatory_citations=[],
            )
            if investigation_result is not None:
                decision.decision_hint = investigation_result.decision_hint.value
                decision.confidence = investigation_result.confidence
                decision.reasoning = investigation_result.reasoning_summary
                decision.evidence = list(investigation_result.evidence)
                decision.red_flags = list(investigation_result.red_flags)
                decision.tools_called = list(investigation_result.tools_called)
                decision.tool_trace = [dict(t) for t in investigation_result.tool_trace]
            if fraud_decision is not None:
                decision.outcome = fraud_decision.outcome.value
                decision.regulatory_citations = [
                    c.model_dump() for c in fraud_decision.regulatory_citations
                ]
                decision.model_name = settings.anthropic_model_haiku
            if sar_report is not None:
                decision.sar_report = sar_report.model_dump(mode="json")
            db.add(decision)
            await db.commit()

        await update_job(
            job_id,
            status="done",
            stage_label="Investigation complete",
            result=full_response.model_dump(mode="json"),
            current_tool=None,
        )
        logger.info("pipeline_complete", job_id=job_id, elapsed_ms=processing_time_ms)

    except Exception as exc:
        logger.exception("pipeline_background_failed", job_id=job_id)
        await update_job(job_id, status="error", error=str(exc))


# ---------------------------------------------------------------------------
# POST /transactions
# ---------------------------------------------------------------------------


@router.post("/transactions", response_model=TransactionSubmitResponse, status_code=202)
async def submit_transaction(
    payload: TransactionRequest,
    background_tasks: BackgroundTasks,
    raw_mode: bool = Query(default=False, description="Bypass InferenceExtractor and score payload.raw_features directly"),  # noqa: B008
) -> TransactionSubmitResponse:
    """Accept a transaction and immediately return a job_id for polling.

    The fraud investigation pipeline runs in the background. Poll
    GET /api/v1/jobs/{job_id} until status is 'done' or 'error'.
    Set raw_mode=true and populate raw_features with IEEE-CIS feature values
    to bypass the InferenceExtractor (useful for demos and integration tests).
    """
    if raw_mode and not payload.raw_features:
        raise HTTPException(
            status_code=422,
            detail="raw_features must be provided when raw_mode=true",
        )

    settings = get_settings()
    job_id = str(uuid.uuid4())
    await create_job(job_id)

    background_tasks.add_task(_run_pipeline_background, job_id, payload, raw_mode, settings)

    logger.info("transaction_queued", job_id=job_id, transaction_id=str(payload.transaction_id))
    return TransactionSubmitResponse(
        job_id=job_id,
        status="queued",
        message=f"Investigation started. Poll /api/v1/jobs/{job_id} for status.",
    )


# ---------------------------------------------------------------------------
# GET /decisions/{id}
# ---------------------------------------------------------------------------


@router.get("/decisions/{decision_id}", response_model=DecisionRead)
async def get_decision(
    decision_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> DecisionRead:
    """Fetch a previously created decision by its UUID."""
    result = await db.get(Decision, decision_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    return DecisionRead.model_validate(result)
