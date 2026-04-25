"""Investigation Agent — LangGraph ReAct agent for medium-risk transactions.

Model: claude-haiku-4-5 (cost-conscious, high-volume ~30% of transactions).
Tools: explain_ml_score, get_customer_history, check_merchant_reputation,
       get_geolocation_context, find_similar_patterns.
Output: InvestigationResult (Pydantic, structured output + retry on failure).

tool_trace and tools_called are populated from the raw LangGraph message history
(not via LLM extraction) so they are always accurate regardless of LLM output quality.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from fraudlens.agents.tools.customer_history import get_customer_history
from fraudlens.agents.tools.explain_ml_score import make_explain_ml_score_tool
from fraudlens.agents.tools.geolocation import get_geolocation_context
from fraudlens.agents.tools.merchant_rep import check_merchant_reputation
from fraudlens.agents.tools.similar_patterns import find_similar_patterns
from fraudlens.core.config import get_settings
from fraudlens.schemas.investigation import DecisionHint, InvestigationResult

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are a senior AML/fraud investigator at a Turkish bank regulated by BDDK and FATF.

A transaction has been flagged by the XGBoost model as MEDIUM RISK (fraud probability 0.3–0.7).

You will receive:
- transaction context: account IDs, channel, country, time signals, customer average amount
- ml_score: XGBoost fraud probability [0-1]
- shap_signals: list of {feature, shap, meaning} for interpretable ML drivers (may be empty — rely on context then)

Transaction type codes: W=web purchase, H=hotel/travel, C=cash, S=subscription, R=recurring payment
Note: customer_avg_amount shows the customer's historical average, not the exact transaction amount.

YOUR GOAL: Determine if this transaction is likely_legitimate, suspicious, or inconclusive.
Use the minimum number of tools needed — do not call tools unnecessarily.

TOOL SELECTION RULES (follow strictly):

explain_ml_score:
  - Call ONLY if shap_signals is empty.
  - Skip if shap_signals is present — it already summarizes the ML reasoning.

get_customer_history:
  - Call if: ml_score > 0.45
  - Call if: shap_signals mentions amount anomaly, address count, or transaction velocity
  - Call if: is_night=true AND ml_score > 0.40
  - Skip if: ml_score < 0.35 AND shap_signals has no behavioral red flags

check_merchant_reputation:
  - Call if: merchant_id is present in context
  - Skip if: merchant_id is missing

get_geolocation_context:
  - Call if: ip_address is present AND (ml_score > 0.50 OR shap_signals mentions device/location signal)
  - Skip if: ip_address is missing

find_similar_patterns:
  - Call ONLY if: 2+ tools already returned red flags AND verdict is still uncertain
  - This is your last resort — avoid unnecessary calls

VERDICT RULES:
- likely_legitimate: ml_score 0.3–0.5 AND tools show normal behavior AND no red flags
- suspicious: ml_score > 0.55 AND 2+ tools return red flags (high address count, velocity, unknown merchant, night transaction)
- inconclusive: conflicting signals (low ml_score but tool red flags, or high ml_score but clean history)

TIME SIGNALS (use in verdict reasoning):
- is_night=true: transaction between 00:00–06:00, elevates suspicion
- is_weekend=true: mildly relevant, note if combined with other signals
- hour_of_day: use to assess if timing is unusual for this customer

EFFICIENCY: A good investigator reaches a verdict in 2–3 tool calls, not 5.
If evidence is clear after 2 tools, stop and deliver your verdict.

Always end with a concise 2–3 sentence summary explaining your reasoning."""

_MAX_STRUCTURED_RETRIES = 2


def _extract_tool_trace(messages: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract ordered tool calls and their results from the LangGraph message history.

    Correlates AIMessage tool_calls with ToolMessage results via tool_call_id.
    More reliable than asking the LLM to report which tools it used.

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
                tool_trace.append({
                    "tool": tc["name"],
                    "args": tc.get("args", {}),
                    "result": "",
                })
        if hasattr(msg, "tool_call_id") and msg.tool_call_id in calls_by_id:
            idx = calls_by_id[msg.tool_call_id]
            tool_trace[idx]["result"] = str(msg.content)[:600]

    tools_called = [tc["tool"] for tc in tool_trace]
    return tools_called, tool_trace


async def run_investigation_agent(
    transaction_id: str,
    fraud_probability: float,
    shap_values: dict[str, float],
    transaction_context: str,
) -> InvestigationResult:
    """Run the Investigation Agent for a single medium-risk transaction.

    Args:
        transaction_id: The transaction UUID string.
        fraud_probability: XGBoost score (expected 0.3–0.7).
        shap_values: Feature name → SHAP contribution from the scorer.
        transaction_context: JSON-serialized transaction fields for agent context.

    Returns:
        InvestigationResult with decision_hint, confidence, evidence, red_flags,
        tools_called, and tool_trace (populated from message history).
        Falls back to INCONCLUSIVE on unrecoverable agent errors.
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
    ]

    agent = create_agent(llm, tools, system_prompt=_SYSTEM_PROMPT)

    human_message = (
        f"Transaction ID: {transaction_id}\n"
        f"ML Fraud Probability: {fraud_probability:.4f}\n"
        f"Transaction details:\n{transaction_context}\n\n"
        "Please investigate this transaction using all relevant tools and provide your verdict."
    )

    log = logger.bind(transaction_id=transaction_id, fraud_probability=fraud_probability)
    log.info("investigation_agent_start")

    try:
        agent_output = await agent.ainvoke({"messages": [HumanMessage(content=human_message)]})
    except Exception:
        log.exception("investigation_agent_failed")
        return _fallback_result(transaction_id)

    messages = agent_output.get("messages", [])
    tools_called, tool_trace = _extract_tool_trace(messages)

    # Extract the agent's final narrative from the last AI message without tool calls.
    final_text = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
            final_text = str(msg.content)
            break

    log.info("investigation_agent_done", tools_called=tools_called)

    # Parse the agent's narrative into a structured result via a dedicated LLM call.
    structured_llm = llm.with_structured_output(InvestigationResult)
    parse_prompt = (
        "Extract a structured investigation result from the following fraud investigation summary.\n\n"
        f"Investigation summary:\n{final_text}\n\n"
        "Note: do NOT include tool_trace or tools_called in your output — "
        "those fields will be set programmatically."
    )

    result: InvestigationResult | None = None
    for attempt in range(_MAX_STRUCTURED_RETRIES + 1):
        try:
            raw = await structured_llm.ainvoke([
                SystemMessage(content="You extract structured data from fraud investigation narratives. Be precise."),
                HumanMessage(content=parse_prompt),
            ])
            result = raw if isinstance(raw, InvestigationResult) else InvestigationResult.model_validate(raw)
            break
        except (ValidationError, Exception) as exc:
            if attempt == _MAX_STRUCTURED_RETRIES:
                log.warning("structured_output_parse_failed", error=str(exc))
                return _fallback_result(transaction_id, reasoning=final_text, tools_called=tools_called, tool_trace=tool_trace)
            log.debug("structured_output_retry", attempt=attempt + 1)

    if result is None:
        return _fallback_result(transaction_id, reasoning=final_text, tools_called=tools_called, tool_trace=tool_trace)

    # Override tools_called and tool_trace with accurate message-history data.
    return result.model_copy(update={"tools_called": tools_called, "tool_trace": tool_trace})


def _fallback_result(
    transaction_id: str,
    reasoning: str = "",
    tools_called: list[str] | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
) -> InvestigationResult:
    """Return a safe INCONCLUSIVE result when the agent or parser fails."""
    return InvestigationResult(
        decision_hint=DecisionHint.INCONCLUSIVE,
        confidence=0.0,
        evidence=[f"Agent execution error for transaction {transaction_id}; manual review required."],
        red_flags=[],
        tools_called=tools_called or [],
        reasoning_summary=reasoning or "Investigation could not be completed; defaulting to manual review.",
        tool_trace=tool_trace or [],
    )
