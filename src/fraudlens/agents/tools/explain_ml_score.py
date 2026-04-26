"""Tool: ML score explanation via SHAP (values injected at agent-creation time)."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool


def make_explain_ml_score_tool(shap_values: dict[str, float]) -> BaseTool:
    """Create an explain_ml_score tool pre-loaded with SHAP values for this transaction.

    SHAP values are injected from the scoring pipeline at agent-creation time,
    avoiding a DB round-trip during agent execution. The LLM calls this tool
    with the current transaction_id; the id parameter is accepted but the
    pre-loaded values are always returned.

    Args:
        shap_values: Feature name → SHAP contribution mapping from FraudScorer.

    Returns:
        A LangChain BaseTool instance bound to the provided SHAP values.
    """
    sorted_features = sorted(
        [{"feature": k, "shap_contribution": round(v, 6)} for k, v in shap_values.items()],
        key=lambda x: abs(x["shap_contribution"]),
        reverse=True,
    )

    @tool
    def explain_ml_score(transaction_id: str) -> str:  # noqa: ARG001
        """Explain why the XGBoost model assigned the current fraud probability.

        Returns the top SHAP feature contributions for this transaction.
        Positive values push the score toward fraud; negative values push toward
        legitimate. Call this first to understand what the ML model found suspicious.

        Args:
            transaction_id: The transaction ID to explain (must match the current transaction).

        Returns:
            JSON with top_features list (feature name + shap_contribution),
            sorted by absolute impact descending.
        """
        return json.dumps(
            {
                "transaction_id": transaction_id,
                "top_features": sorted_features[:10],
                "interpretation": ("Positive shap_contribution → increases fraud probability. Negative → decreases it (pushes toward legitimate)."),
            }
        )

    return explain_ml_score
