"""Tool: similar fraud pattern lookup (mock — Qdrant hybrid retrieval in Week 5)."""

from __future__ import annotations

import json

from langchain_core.tools import tool

_PATTERNS = ["card_testing", "account_takeover", "structuring"]
_RISK_LEVELS = {0: "none", 1: "low", 2: "high", 3: "high"}


@tool
def find_similar_patterns(transaction_id: str, k: int = 5) -> str:
    """Find historically similar fraud patterns from the case database.

    Week 5: replaces this mock with Qdrant hybrid BM25+dense retrieval.
    Currently returns deterministic results seeded from transaction_id.

    Args:
        transaction_id: Current transaction ID used as retrieval seed.
        k: Number of similar historical cases to retrieve (1–10, default 5).

    Returns:
        JSON with match_count, top_match_score, patterns list, and risk_level.
    """
    hash_val = sum(ord(c) for c in transaction_id) % 100
    match_count = hash_val % 4
    top_match_score = round(0.5 + (hash_val % 50) / 100, 2)
    patterns = _PATTERNS[:match_count]
    risk_level = _RISK_LEVELS[match_count]

    return json.dumps({
        "transaction_id": transaction_id,
        "match_count": match_count,
        "top_match_score": top_match_score if match_count > 0 else None,
        "patterns": patterns,
        "risk_level": risk_level,
    })
