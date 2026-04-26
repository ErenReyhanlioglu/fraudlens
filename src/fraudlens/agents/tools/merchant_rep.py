"""Tool: merchant reputation check (mock)."""

from __future__ import annotations

import json

from langchain_core.tools import tool

_INDUSTRIES = ["retail", "travel", "marketplace", "gambling", "crypto_exchange"]


def _parse_merchant_int(merchant_id: str) -> int:
    """Extract integer from MERCH-NNNN format; fall back to char sum."""
    parts = merchant_id.split("-")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return sum(ord(c) for c in merchant_id) % 10000


@tool
def check_merchant_reputation(merchant_id: str) -> str:
    """Check the fraud risk reputation of a merchant.

    Use this whenever merchant_id is present in the transaction.
    A risk_score above 0.7 or high chargeback rates are strong fraud signals.

    Args:
        merchant_id: The merchant's identifier string.

    Returns:
        JSON with risk_score (0–1), chargeback_rate_pct, flags list,
        industry_category, and months_on_platform.
    """
    m = _parse_merchant_int(merchant_id)

    risk_score = round((m % 100) / 100, 2)
    chargeback_rate = round((m % 20) / 10, 2)

    flags: list[str] = []
    if risk_score > 0.7:
        flags.append("high_fraud_rate_history")
    elif risk_score > 0.5:
        flags.append("excessive_chargebacks")

    return json.dumps(
        {
            "merchant_id": merchant_id,
            "risk_score": risk_score,
            "chargeback_rate_pct": chargeback_rate,
            "flags": flags,
            "industry_category": _INDUSTRIES[m % 5],
            "months_on_platform": 1 + (m % 84),
            "dispute_count_90d": m % 20,
        }
    )
