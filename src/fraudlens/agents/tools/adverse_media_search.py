"""Tool: adverse media and sanctions check (mock — deterministic from customer_id hash)."""

from __future__ import annotations

import json

from langchain_core.tools import tool

_SANCTION_LISTS = ["OFAC SDN", "EU Consolidated", "UN Security Council"]
_ADVERSE_CATEGORIES = ["fraud", "money_laundering", "corruption", "terrorism_financing"]


@tool
def adverse_media_search(customer_id: str, full_name: str = "") -> str:
    """Search adverse media coverage and global sanctions lists for a customer.

    Checks OFAC SDN, EU Consolidated, and UN Security Council lists for
    entity matches. Also scans adverse media signals (fraud, money laundering,
    corruption, terrorism financing). A match on any sanctions list is an
    immediate red flag requiring escalation.

    Args:
        customer_id: Customer account ID or entity identifier.
        full_name: Optional full name of the customer for name-matching.

    Returns:
        JSON with sanctions_match (bool), sanctions_list_hits, adverse_media_hits,
        pep_flag (politically exposed person), and overall_risk_level.
    """
    seed = sum(ord(c) for c in customer_id + full_name) % 1000

    # Only ~2% of customers hit a sanctions list (seed < 20)
    sanctions_match = seed < 20
    sanctions_list_hits: list[str] = []
    if sanctions_match:
        list_idx = seed % len(_SANCTION_LISTS)
        sanctions_list_hits = [_SANCTION_LISTS[list_idx]]

    # ~8% show adverse media (seed 20–99)
    adverse_hit_count = 0
    adverse_categories: list[str] = []
    if 20 <= seed < 100:
        adverse_hit_count = 1 + (seed % 3)
        adverse_categories = _ADVERSE_CATEGORIES[: min(adverse_hit_count, len(_ADVERSE_CATEGORIES))]

    # ~5% are PEPs (seed % 20 == 0 and not already sanctions)
    pep_flag = (not sanctions_match) and (seed % 20 == 0)

    risk_signals: list[str] = []
    if sanctions_match:
        risk_signals.append("sanctions_list_hit")
    if adverse_hit_count > 0:
        risk_signals.append("adverse_media")
    if pep_flag:
        risk_signals.append("pep")

    risk_level = "critical" if sanctions_match else ("high" if risk_signals else "low")

    return json.dumps(
        {
            "customer_id": customer_id,
            "sanctions_match": sanctions_match,
            "sanctions_list_hits": sanctions_list_hits,
            "adverse_media_hit_count": adverse_hit_count,
            "adverse_media_categories": adverse_categories,
            "pep_flag": pep_flag,
            "risk_signals": risk_signals,
            "overall_risk_level": risk_level,
            "note": "Mock data — real OFAC/EU API integration pending.",
        }
    )
