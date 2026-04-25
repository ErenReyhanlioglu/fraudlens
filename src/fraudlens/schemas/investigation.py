"""Pydantic v2 schemas for agent investigation results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionHint(StrEnum):
    """High-level signal from the investigation agent to the decision synthesizer."""

    LIKELY_LEGITIMATE = "likely_legitimate"
    SUSPICIOUS = "suspicious"
    INCONCLUSIVE = "inconclusive"


class InvestigationResult(BaseModel):
    """Structured output produced by the Investigation Agent after tool-assisted analysis.

    Validated with Pydantic strict mode; the agent is retried on parse failure.
    tool_trace and tools_called are populated directly from the LangGraph message
    history (not via LLM extraction) so they are always accurate.
    """

    model_config = ConfigDict(frozen=True)

    decision_hint: DecisionHint
    confidence: float = Field(ge=0.0, le=1.0, description="Agent confidence in its decision_hint (0–1).")
    evidence: list[str] = Field(min_length=1, description="Supporting facts gathered from tools.")
    red_flags: list[str] = Field(default_factory=list, description="Specific fraud signals found.")
    tools_called: list[str] = Field(default_factory=list, description="Names of tools invoked during investigation.")
    reasoning_summary: str = Field(min_length=10, description="Concise narrative of the agent's reasoning.")
    tool_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered tool invocations from message history: [{tool, args, result}].",
    )
