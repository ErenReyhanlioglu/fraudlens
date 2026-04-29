"""SAR Generator — produces a structured Suspicious Activity Report.

Runs only when the Decision Synthesizer outputs `escalate`. Uses claude-haiku-4-5
with Anthropic prompt caching applied to the system template (the section
headers stay constant across runs so caching them saves cost on every call).

Failures never propagate: a deterministic fallback assembles a minimal but
valid SARReport from the existing investigation evidence so a SAR is always
produced for escalated cases.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from fraudlens.core.config import Settings
from fraudlens.schemas.decision import FraudDecision
from fraudlens.schemas.investigation import InvestigationResult
from fraudlens.schemas.sar import SARReport

logger = structlog.get_logger(__name__)

_SAR_SYSTEM_PROMPT = """You are an AML/CFT compliance officer at a Turkish bank regulated by BDDK and FATF.
You are drafting a Suspicious Activity Report (SAR) — known in Turkey as a Şüpheli İşlem Bildirimi (ŞİB) —
for filing with MASAK (the Turkish Financial Intelligence Unit).

Input fields available to you:
- transaction_context: raw transaction data (amount, currency, channel, timestamps)
- investigation_result: agent findings (evidence, red_flags, reasoning_summary)
- fraud_decision: final outcome, regulatory_citations (source, page, excerpt)

Output MUST follow this exact structure:

1. Customer Information — Account identifiers, KYC fields, risk profile.
2. Transaction Details — Amount, currency, channel, counterparties, timestamps.
3. Suspicious Indicators — Concrete red flags (circular flow, sanctions match, velocity anomaly, etc.).
4. Investigation Summary — Concise narrative of the agent's reasoning and the evidence chain.
5. Regulatory Trigger References — Statutory references that mandate the filing
   (e.g. FATF Recommendation 13, MASAK Article 4, BDDK regulations).
6. Recommended Action — File STR, freeze account, enhanced due diligence, etc.

Rules:
- Be specific and factual. Cite exact numbers, dates, and identifiers from the input data.
- Suspicious indicators must be observable facts from the investigation, not speculation.
- Regulatory triggers must reference real provisions found in the investigation evidence
  or known FATF/MASAK requirements.
- Recommended action must be a clear next step a compliance officer can execute today.
- Include only PII that is present in the input data. Do not infer or fabricate identifiers.
- Draft the report in English for internal review purposes.
  A Turkish translation for MASAK filing is handled separately.
- Output is for institutional record-keeping; tone is formal and concise."""


def _build_human_message(
    fraud_decision: FraudDecision,
    transaction_context: dict[str, Any],
    investigation_result: InvestigationResult,
) -> str:
    """Assemble the user-side prompt with all inputs the LLM needs."""
    return (
        f"Generate a SAR for the following escalated transaction.\n\n"
        f"Transaction ID: {fraud_decision.transaction_id}\n"
        f"ML fraud probability: {fraud_decision.ml_score:.4f}\n"
        f"Agent decision hint: {fraud_decision.decision_hint}\n"
        f"Agent confidence: {fraud_decision.confidence:.2f}\n"
        f"Final outcome: {fraud_decision.outcome.value}\n\n"
        f"Transaction context:\n{transaction_context}\n\n"
        f"Investigation evidence:\n"
        + "\n".join(f"- {e}" for e in investigation_result.evidence)
        + f"\n\nRed flags:\n"
        + "\n".join(f"- {rf}" for rf in investigation_result.red_flags)
        + f"\n\nAgent reasoning:\n{investigation_result.reasoning_summary}\n\n"
        f"Regulatory citations surfaced by RAG:\n"
        + (
            "\n".join(
                f"- {c.source}, p.{c.page}: {c.excerpt[:160]}..."
                for c in fraud_decision.regulatory_citations[:5]
            )
            or "- (none)"
        )
    )


def _fallback_sar(
    fraud_decision: FraudDecision,
    transaction_context: dict[str, Any],
    investigation_result: InvestigationResult,
    agent_model: str,
) -> SARReport:
    """Deterministic fallback used when the LLM call or parse fails.

    Composes a valid SARReport from the inputs we already have so the pipeline
    never returns no-SAR for an escalated decision.
    """
    triggers: list[str] = []
    for c in fraud_decision.regulatory_citations[:10]:
        page = f", p.{c.page}" if c.page is not None else ""
        triggers.append(f"{c.source}{page}")

    return SARReport(
        transaction_id=fraud_decision.transaction_id,
        customer_info={
            "sender_account_id": transaction_context.get("sender_account_id"),
            "sender_country": transaction_context.get("sender_country"),
        },
        transaction_details={
            "amount": transaction_context.get("amount"),
            "currency": transaction_context.get("currency"),
            "channel": transaction_context.get("channel"),
            "receiver_account_id": transaction_context.get("receiver_account_id"),
            "receiver_country": transaction_context.get("receiver_country"),
            "ml_score": fraud_decision.ml_score,
        },
        suspicious_indicators=list(investigation_result.red_flags),
        investigation_summary=investigation_result.reasoning_summary,
        regulatory_triggers=triggers or ["FATF Recommendation 13 (suspicious transaction reporting)"],
        recommended_action="File STR with MASAK and conduct enhanced due diligence on sender account.",
        generated_at=datetime.now(UTC),
        agent_model=agent_model,
    )


async def generate_sar_report(
    fraud_decision: FraudDecision,
    transaction_context: dict[str, Any],
    investigation_result: InvestigationResult,
    settings: Settings,
) -> SARReport:
    """Generate a structured SAR for an escalated decision.

    Args:
        fraud_decision: The synthesized decision (must have outcome=ESCALATE).
        transaction_context: Full transaction request fields (banking + ML signals).
        investigation_result: The agent's structured investigation output.
        settings: Application settings carrying anthropic_api_key + model name.

    Returns:
        SARReport. Falls back to a deterministic SAR composed from inputs on
        any LLM or validation failure — exceptions are never propagated.
    """
    model_name = settings.anthropic_model_haiku
    log = logger.bind(transaction_id=fraud_decision.transaction_id, model=model_name)
    log.info("sar_generator_start")

    llm = ChatAnthropic(
        model=model_name,
        api_key=settings.anthropic_api_key.get_secret_value(),
        temperature=0,
        max_retries=2,
    )

    # Anthropic prompt caching — only the system template is marked ephemeral.
    # The same template is reused for every SAR, so the cache hit rate is high.
    system_message = SystemMessage(
        content=[
            {
                "type": "text",
                "text": _SAR_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )

    human_message = HumanMessage(
        content=_build_human_message(fraud_decision, transaction_context, investigation_result)
    )

    structured_llm = llm.with_structured_output(SARReport)

    try:
        raw = await structured_llm.ainvoke([system_message, human_message])
    except Exception as exc:
        log.warning("sar_llm_call_failed", error=str(exc))
        return _fallback_sar(fraud_decision, transaction_context, investigation_result, model_name)

    try:
        report: SARReport = (
            raw if isinstance(raw, SARReport) else SARReport.model_validate(raw)
        )
    except ValidationError as exc:
        log.warning("sar_structured_output_invalid", error=str(exc))
        return _fallback_sar(fraud_decision, transaction_context, investigation_result, model_name)

    # Override two fields that should always reflect server-side truth.
    report = report.model_copy(
        update={
            "transaction_id": fraud_decision.transaction_id,
            "agent_model": model_name,
            "generated_at": datetime.now(UTC),
        }
    )
    log.info("sar_generator_done", indicators=len(report.suspicious_indicators))
    return report
