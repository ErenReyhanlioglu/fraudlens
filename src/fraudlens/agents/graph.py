"""Top-level LangGraph state graph for fraud investigation routing.

Accepts a typed FraudInvestigationState, routes to the appropriate agent
(Investigation or Critical based on risk_tier), and returns the populated state
with an InvestigationResult attached.
"""

from __future__ import annotations

import json
from typing import Any, NotRequired, TypedDict

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from fraudlens.agents.critical import run_critical_agent
from fraudlens.agents.investigation import run_investigation_agent
from fraudlens.schemas.decision import TriageAction
from fraudlens.schemas.investigation import InvestigationResult

logger = structlog.get_logger(__name__)


class FraudInvestigationState(TypedDict):
    """Typed state threaded through the fraud investigation graph."""

    transaction_id: str
    fraud_probability: float
    shap_values: dict[str, float]
    transaction_context: dict[str, Any]
    triage_action: str
    investigation_result: NotRequired[InvestigationResult]


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
    builder.add_conditional_edges(START, _route, {"investigate": "investigate", "critical": "critical"})
    builder.add_edge("investigate", END)
    builder.add_edge("critical", END)
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
) -> InvestigationResult:
    """Entry point for the fraud investigation graph.

    Args:
        transaction_id: UUID string of the transaction being investigated.
        fraud_probability: XGBoost score from the scoring pipeline.
        shap_values: Feature name → SHAP contribution dict (top 10).
        transaction_context: Serialisable dict representation of TransactionRequest.
        triage_action: TriageAction value (INVESTIGATE or ESCALATE).

    Returns:
        InvestigationResult with decision_hint, confidence, evidence, and red_flags.
    """
    initial_state: FraudInvestigationState = {
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "shap_values": shap_values,
        "transaction_context": transaction_context,
        "triage_action": triage_action,
    }
    output: FraudInvestigationState = await _graph.ainvoke(initial_state)
    return output["investigation_result"]
