"""Decision Synthesizer — deterministic mapping from InvestigationResult to FraudDecision.

No LLM call is made here: the outcome is a pure function of the agent's
`decision_hint`, the upstream `triage_action`, and the agent's confidence.
This keeps the final outcome auditable and reproducible across runs.

Regulatory citations surfaced by the `regulatory_policy_rag` tool are parsed
from the agent's tool_trace and converted into structured `Regulatorycitation`
records. Parse failures are swallowed and an empty list is returned.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from fraudlens.schemas.decision import (
    AgentType,
    DecisionOutcome,
    FraudDecision,
    Regulatorycitation,
    TriageAction,
)
from fraudlens.schemas.investigation import DecisionHint, InvestigationResult

logger = structlog.get_logger(__name__)

_LEGITIMATE_ESCALATE_CONFIDENCE_OVERRIDE = 0.6


def _map_outcome(hint: DecisionHint, triage_action: str, confidence: float) -> DecisionOutcome:
    """Map (hint, triage_action, confidence) to a final outcome.

    Mapping rules:
        likely_legitimate + INVESTIGATE                    → APPROVE
        likely_legitimate + ESCALATE + conf > 0.6          → APPROVE
        likely_legitimate + ESCALATE + conf <= 0.6         → MANUAL_REVIEW
        suspicious        + INVESTIGATE                    → DECLINE
        suspicious        + ESCALATE                       → ESCALATE (triggers SAR)
        inconclusive      + INVESTIGATE                    → MANUAL_REVIEW
        inconclusive      + ESCALATE                       → ESCALATE (triggers SAR)

    Defensive default is MANUAL_REVIEW so unknown combinations route to a human.
    """
    is_escalate = triage_action == TriageAction.ESCALATE.value

    if hint is DecisionHint.LIKELY_LEGITIMATE:
        if not is_escalate:
            return DecisionOutcome.APPROVE
        return (
            DecisionOutcome.APPROVE
            if confidence > _LEGITIMATE_ESCALATE_CONFIDENCE_OVERRIDE
            else DecisionOutcome.MANUAL_REVIEW
        )

    if hint is DecisionHint.SUSPICIOUS:
        return DecisionOutcome.ESCALATE if is_escalate else DecisionOutcome.DECLINE

    if hint is DecisionHint.INCONCLUSIVE:
        return DecisionOutcome.ESCALATE if is_escalate else DecisionOutcome.MANUAL_REVIEW

    return DecisionOutcome.MANUAL_REVIEW


def _parse_rag_citations(tool_trace: list[dict[str, Any]]) -> list[Regulatorycitation]:
    """Extract Regulatorycitation entries from regulatory_policy_rag tool results.

    Each call's result is JSON with an `excerpts` list of
    `{text, citation, source, page, relevance_score}` objects. Failures (bad
    JSON, missing keys, validation errors) are logged and skipped — never raised.
    """
    citations: list[Regulatorycitation] = []
    for entry in tool_trace:
        if entry.get("tool") != "regulatory_policy_rag":
            continue
        raw = entry.get("result", "")
        if not raw:
            continue
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("rag_citation_parse_failed", error=str(exc))
            continue

        for ex in data.get("excerpts", []) or []:
            try:
                citations.append(
                    Regulatorycitation(
                        source=str(ex.get("source", "unknown")),
                        article=None,
                        page=int(ex["page"]) if ex.get("page") is not None else None,
                        excerpt=str(ex.get("text", "")),
                        relevance_score=float(ex.get("relevance_score", 0.0)),
                    )
                )
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug("rag_citation_invalid_entry", error=str(exc))
                continue

    # De-duplicate by (source, page, excerpt) preserving first-seen order
    seen: set[tuple[str, int | None, str]] = set()
    unique: list[Regulatorycitation] = []
    for c in citations:
        key = (c.source, c.page, c.excerpt[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


async def synthesize_decision(
    investigation_result: InvestigationResult,
    fraud_probability: float,
    triage_action: str,
    transaction_id: str,
) -> FraudDecision:
    """Combine an InvestigationResult with triage context into a FraudDecision.

    Args:
        investigation_result: Output of the Investigation or Critical Agent.
        fraud_probability: XGBoost score that drove triage.
        triage_action: TriageAction value (INVESTIGATE or ESCALATE).
        transaction_id: UUID string of the transaction being decided.

    Returns:
        FraudDecision with deterministic outcome and parsed regulatory citations.
        Falls back to MANUAL_REVIEW outcome on unrecognized hint/action combos.
    """
    agent_used = (
        AgentType.CRITICAL
        if triage_action == TriageAction.ESCALATE.value
        else AgentType.INVESTIGATION
    )

    outcome = _map_outcome(
        investigation_result.decision_hint,
        triage_action,
        investigation_result.confidence,
    )

    citations = _parse_rag_citations(list(investigation_result.tool_trace))

    logger.info(
        "decision_synthesized",
        transaction_id=transaction_id,
        triage_action=triage_action,
        decision_hint=investigation_result.decision_hint.value,
        outcome=outcome.value,
        confidence=investigation_result.confidence,
        citations=len(citations),
    )

    return FraudDecision(
        transaction_id=transaction_id,
        outcome=outcome,
        confidence=investigation_result.confidence,
        ml_score=fraud_probability,
        agent_used=agent_used,
        decision_hint=investigation_result.decision_hint.value,
        evidence=list(investigation_result.evidence),
        red_flags=list(investigation_result.red_flags),
        regulatory_citations=citations,
        reasoning=investigation_result.reasoning_summary,
        tools_called=list(investigation_result.tools_called),
    )
