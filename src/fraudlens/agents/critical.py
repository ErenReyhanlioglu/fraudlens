"""Critical Agent — LangGraph ReAct agent for high-risk transactions (p >= 0.7).

Model: claude-haiku-4-5 (same model, stricter prompt and more tools than Investigation Agent).
Tools (8): explain_ml_score, get_customer_history, check_merchant_reputation,
           get_geolocation_context, find_similar_patterns,
           deep_network_analysis, regulatory_policy_rag, adverse_media_search.
Output: InvestigationResult (Pydantic, structured output + retry on failure).
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from fraudlens.agents.tools.adverse_media_search import adverse_media_search
from fraudlens.agents.tools.customer_history import get_customer_history
from fraudlens.agents.tools.explain_ml_score import make_explain_ml_score_tool
from fraudlens.agents.tools.geolocation import get_geolocation_context
from fraudlens.agents.tools.merchant_rep import check_merchant_reputation
from fraudlens.agents.tools.network_analysis import deep_network_analysis
from fraudlens.agents.tools.regulatory_rag import regulatory_policy_rag
from fraudlens.agents.tools.similar_patterns import find_similar_patterns
from fraudlens.core.config import get_settings
from fraudlens.schemas.investigation import DecisionHint, InvestigationResult

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are a senior AML/fraud investigator and compliance officer at a Turkish bank
regulated by BDDK and FATF. You have been escalated a CRITICAL RISK transaction (fraud probability >= 0.7).

You will receive:
- transaction context: account IDs, channel, country, time signals, customer average amount
- ml_score: XGBoost fraud probability [0-1]
- shap_signals: list of {feature, shap, meaning} for interpretable ML drivers (may be empty)

Transaction type codes: W=web purchase, H=hotel/travel, C=cash, S=subscription, R=recurring payment

YOUR GOAL: Conduct a thorough investigation to determine if this transaction should be ESCALATED
for SAR filing, is SUSPICIOUS but does not warrant SAR, or is INCONCLUSIVE.
At this risk tier, err on the side of caution — SUSPICIOUS is preferred over INCONCLUSIVE.

MANDATORY CHECKS (always run for critical-tier transactions):
1. explain_ml_score — if shap_signals is empty
2. get_customer_history — always run; behavioral baseline is mandatory at this tier
3. adverse_media_search — always run; sanctions/PEP check is a compliance requirement
4. deep_network_analysis — always run; detect layering/smurfing in transaction graph
5. regulatory_policy_rag — always run; critical tier requires a regulatory basis for every verdict

CONDITIONAL CHECKS (run based on findings):
6. check_merchant_reputation — if merchant_id present AND network_analysis or customer_history shows red flags
7. get_geolocation_context — if ip_address present AND ml_score > 0.75 or network risk is high
8. find_similar_patterns — last resort if verdict is still inconclusive after 6+ tool calls

VERDICT RULES:
- suspicious: ANY of: sanctions match, PEP flag, circular fund flow, ml_score > 0.80 with 2+ red flags
- inconclusive: conflicting signals where some tools show clean results despite high ML score
- likely_legitimate: all mandatory checks clean AND ml_score 0.70–0.75 with no behavioral anomalies

REGULATORY CITATION REQUIREMENT:
You MUST call regulatory_policy_rag and include the citation (source + page) in your reasoning_summary
and evidence fields. Every critical-tier verdict requires a documented regulatory basis.

CONFIDENCE SCORE RULES (you must output a float 0.0–1.0):
- suspicious with 3+ red flags confirmed by tools: 0.85–0.95
- suspicious with 1–2 red flags: 0.65–0.85
- inconclusive with conflicting signals: 0.40–0.60
- likely_legitimate with all clean mandatory checks: 0.70–0.90
Never output confidence=0.0 unless every tool call returned an error.

EFFICIENCY: Critical tier requires thoroughness. Run 5–7 tools. Never stop at 2–3 for critical transactions.

Always end with a detailed 3–5 sentence summary explaining your reasoning and regulatory basis."""

_MAX_STRUCTURED_RETRIES = 2


def _extract_tool_trace(messages: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract ordered tool calls and results from LangGraph message history.

    Correlates AIMessage tool_calls with ToolMessage results via tool_call_id.

    Returns:
        tools_called: Ordered list of tool names.
        tool_trace: Ordered list of {tool, args, result} dicts.
    """
    tool_trace: list[dict[str, Any]] = []
    calls_by_id: dict[str, int] = {}

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                idx = len(tool_trace)
                calls_by_id[tc["id"]] = idx
                tool_trace.append({"tool": tc["name"], "args": tc.get("args", {}), "result": ""})
        if hasattr(msg, "tool_call_id") and msg.tool_call_id in calls_by_id:
            idx = calls_by_id[msg.tool_call_id]
            limit = 15000 if tool_trace[idx]["tool"] == "regulatory_policy_rag" else 3000
            tool_trace[idx]["result"] = str(msg.content)[:limit]

    tools_called = [tc["tool"] for tc in tool_trace]
    return tools_called, tool_trace


async def run_critical_agent(
    transaction_id: str,
    fraud_probability: float,
    shap_values: dict[str, float],
    transaction_context: str,
) -> InvestigationResult:
    """Run the Critical Agent for a single high-risk transaction.

    Args:
        transaction_id: The transaction UUID string.
        fraud_probability: XGBoost score (expected >= 0.7).
        shap_values: Feature name → SHAP contribution from the scorer.
        transaction_context: JSON-serialized transaction fields for agent context.

    Returns:
        InvestigationResult with decision_hint leaning toward SUSPICIOUS at this tier.
        Falls back to SUSPICIOUS (not INCONCLUSIVE) on unrecoverable errors.
    """
    settings = get_settings()

    llm = ChatAnthropic(
        model=settings.anthropic_model_haiku,
        api_key=settings.anthropic_api_key.get_secret_value(),
        temperature=0,
        max_retries=2,
    )

    tools = [
        make_explain_ml_score_tool(shap_values),
        get_customer_history,
        check_merchant_reputation,
        get_geolocation_context,
        find_similar_patterns,
        deep_network_analysis,
        regulatory_policy_rag,
        adverse_media_search,
    ]

    agent = create_agent(llm, tools)

    # Anthropic prompt caching — the system prompt is identical across runs.
    system_message = SystemMessage(
        content=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    )
    human_message = HumanMessage(
        content=(
            f"Transaction ID: {transaction_id}\n"
            f"ML Fraud Probability: {fraud_probability:.4f}\n"
            f"Transaction details:\n{transaction_context}\n\n"
            "CRITICAL RISK transaction. Conduct a full investigation using all mandatory tools "
            "and provide your compliance verdict with regulatory citations where applicable."
        )
    )

    log = logger.bind(transaction_id=transaction_id, fraud_probability=fraud_probability)
    log.info("critical_agent_start")

    try:
        agent_output = await agent.ainvoke({"messages": [system_message, human_message]})
    except Exception:
        log.exception("critical_agent_failed")
        return _fallback_result(transaction_id)

    messages = agent_output.get("messages", [])
    tools_called, tool_trace = _extract_tool_trace(messages)

    final_text = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
            final_text = str(msg.content)
            break

    log.info("critical_agent_done", tools_called=tools_called)

    structured_llm = llm.with_structured_output(InvestigationResult)
    parse_prompt = (
        "Extract a structured investigation result from the following CRITICAL TIER "
        "fraud investigation summary.\n\n"
        f"Investigation summary:\n{final_text}\n\n"
        "Rules:\n"
        "- decision_hint: one of 'suspicious', 'inconclusive', or 'likely_legitimate'\n"
        "- confidence: float 0.0–1.0 reflecting verdict certainty. "
        "Use 0.85–0.95 for suspicious with 3+ confirmed red flags, "
        "0.65–0.85 for suspicious with 1–2 red flags, "
        "0.40–0.60 for inconclusive, 0.70–0.90 for likely_legitimate. "
        "NEVER return 0.0 unless all tools failed.\n"
        "- evidence: list of specific facts from the investigation (tool outputs, scores, dates). "
        "If regulatory_policy_rag was called, include the cited regulation as an evidence item.\n"
        "- red_flags: list of short risk signal labels (e.g. 'circular_fund_flow', 'sanctions_match', "
        "'sanctions_list_hit', 'pep_flag', 'structuring_pattern')\n"
        "- reasoning_summary: 3–5 sentence narrative explaining the verdict, the evidence chain, "
        "and any regulatory basis. If a regulation was cited, name it explicitly.\n"
        "- do NOT include tool_trace or tools_called — those are set programmatically\n"
    )

    result: InvestigationResult | None = None
    for attempt in range(_MAX_STRUCTURED_RETRIES + 1):
        try:
            raw = await structured_llm.ainvoke(
                [
                    SystemMessage(content="You extract structured data from fraud investigation narratives. Be precise."),
                    HumanMessage(content=parse_prompt),
                ]
            )
            result = raw if isinstance(raw, InvestigationResult) else InvestigationResult.model_validate(raw)
            break
        except (ValidationError, Exception) as exc:
            if attempt == _MAX_STRUCTURED_RETRIES:
                log.warning("structured_output_parse_failed", error=str(exc))
                return _fallback_result(
                    transaction_id,
                    reasoning=final_text,
                    tools_called=tools_called,
                    tool_trace=tool_trace,
                )
            log.debug("structured_output_retry", attempt=attempt + 1)

    if result is None:
        return _fallback_result(transaction_id, reasoning=final_text, tools_called=tools_called, tool_trace=tool_trace)

    return result.model_copy(update={"tools_called": tools_called, "tool_trace": tool_trace})


def _fallback_result(
    transaction_id: str,
    reasoning: str = "",
    tools_called: list[str] | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
) -> InvestigationResult:
    """Return a safe SUSPICIOUS result when the critical agent or parser fails.

    High-risk tier defaults to SUSPICIOUS (not INCONCLUSIVE) to ensure
    escalation on error rather than under-reporting.
    """
    return InvestigationResult(
        decision_hint=DecisionHint.SUSPICIOUS,
        confidence=0.0,
        evidence=[f"Critical agent execution error for transaction {transaction_id}; manual review required."],
        red_flags=["critical_agent_error", "high_risk_tier_auto_escalate"],
        tools_called=tools_called or [],
        reasoning_summary=reasoning or "Critical investigation could not complete; defaulting to escalation for safety.",
        tool_trace=tool_trace or [],
    )
