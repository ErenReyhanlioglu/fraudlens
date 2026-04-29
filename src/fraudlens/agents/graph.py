"""Top-level LangGraph state graph for fraud investigation routing.

Pipeline: investigate|critical → synthesize → sar → END

The state carries the transaction inputs, the agent's InvestigationResult,
the deterministic FraudDecision produced by the synthesizer, and (for
escalated cases only) the SARReport produced by the SAR Generator.

The public entry point now returns the full state so callers can read every
artefact without redoing the routing in the API layer.
"""

from __future__ import annotations

import json
from typing import Any, NotRequired, TypedDict

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from fraudlens.agents.critical import run_critical_agent
from fraudlens.agents.investigation import run_investigation_agent
from fraudlens.agents.sar_generator import generate_sar_report
from fraudlens.agents.synthesizer import synthesize_decision
from fraudlens.core.config import get_settings
from fraudlens.schemas.decision import DecisionOutcome, FraudDecision, TriageAction
from fraudlens.schemas.investigation import InvestigationResult
from fraudlens.schemas.sar import SARReport

logger = structlog.get_logger(__name__)


class FraudInvestigationState(TypedDict):
    """Typed state threaded through the fraud investigation graph."""

    transaction_id: str
    fraud_probability: float
    shap_values: dict[str, float]
    transaction_context: dict[str, Any]
    triage_action: str
    investigation_result: NotRequired[InvestigationResult]
    fraud_decision: NotRequired[FraudDecision]
    sar_report: NotRequired[SARReport]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def _node_investigate(state: FraudInvestigationState, config: RunnableConfig) -> FraudInvestigationState:  # noqa: ARG001
    """Run the Investigation Agent (claude-haiku-4-5, 5 tools)."""
    result = await run_investigation_agent(
        transaction_id=state["transaction_id"],
        fraud_probability=state["fraud_probability"],
        shap_values=state["shap_values"],
        transaction_context=json.dumps(state["transaction_context"], default=str),
    )
    return {**state, "investigation_result": result}


async def _node_critical(state: FraudInvestigationState, config: RunnableConfig) -> FraudInvestigationState:  # noqa: ARG001
    """Run the Critical Agent (claude-haiku-4-5, 8 tools + RAG)."""
    result = await run_critical_agent(
        transaction_id=state["transaction_id"],
        fraud_probability=state["fraud_probability"],
        shap_values=state["shap_values"],
        transaction_context=json.dumps(state["transaction_context"], default=str),
    )
    return {**state, "investigation_result": result}


async def _node_synthesize(state: FraudInvestigationState, config: RunnableConfig) -> FraudInvestigationState:  # noqa: ARG001
    """Combine the InvestigationResult with triage context into a FraudDecision.

    Deterministic mapping — no LLM call. See synthesizer module for rules.
    """
    investigation_result = state["investigation_result"]
    decision = await synthesize_decision(
        investigation_result=investigation_result,
        fraud_probability=state["fraud_probability"],
        triage_action=state["triage_action"],
        transaction_id=state["transaction_id"],
    )
    return {**state, "fraud_decision": decision}


async def _node_sar(state: FraudInvestigationState, config: RunnableConfig) -> FraudInvestigationState:  # noqa: ARG001
    """Generate a SAR only when the synthesized outcome is ESCALATE.

    Failures inside the generator are absorbed (the generator itself returns
    a deterministic fallback SAR), so this node never raises.
    """
    decision = state.get("fraud_decision")
    if decision is None or decision.outcome is not DecisionOutcome.ESCALATE:
        return state

    settings = get_settings()
    try:
        report = await generate_sar_report(
            fraud_decision=decision,
            transaction_context=state["transaction_context"],
            investigation_result=state["investigation_result"],
            settings=settings,
        )
    except Exception:
        logger.exception("sar_node_unexpected_failure", transaction_id=state["transaction_id"])
        return state

    return {**state, "sar_report": report}


def _route(state: FraudInvestigationState) -> str:
    """Route to the correct agent node based on triage_action."""
    if state["triage_action"] == TriageAction.ESCALATE:
        return "critical"
    return "investigate"


# ---------------------------------------------------------------------------
# Build the compiled graph (module-level singleton)
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    builder: StateGraph = StateGraph(FraudInvestigationState)
    builder.add_node("investigate", _node_investigate)
    builder.add_node("critical", _node_critical)
    builder.add_node("synthesize", _node_synthesize)
    builder.add_node("sar", _node_sar)
    builder.add_conditional_edges(
        START, _route, {"investigate": "investigate", "critical": "critical"}
    )
    builder.add_edge("investigate", "synthesize")
    builder.add_edge("critical", "synthesize")
    builder.add_edge("synthesize", "sar")
    builder.add_edge("sar", END)
    return builder.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point called from the API layer
# ---------------------------------------------------------------------------


async def run_fraud_investigation(
    transaction_id: str,
    fraud_probability: float,
    shap_values: dict[str, float],
    transaction_context: dict[str, Any],
    triage_action: str,
) -> FraudInvestigationState:
    """Entry point for the fraud investigation graph.

    Args:
        transaction_id: UUID string of the transaction being investigated.
        fraud_probability: XGBoost score from the scoring pipeline.
        shap_values: Feature name → SHAP contribution dict (top 10).
        transaction_context: Serialisable dict representation of TransactionRequest.
        triage_action: TriageAction value (INVESTIGATE or ESCALATE).

    Returns:
        FraudInvestigationState containing investigation_result, fraud_decision
        and (only for escalated cases) sar_report. Callers read whichever
        artefact they need.
    """
    initial_state: FraudInvestigationState = {
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "shap_values": shap_values,
        "transaction_context": transaction_context,
        "triage_action": triage_action,
    }
    return await _graph.ainvoke(initial_state)
