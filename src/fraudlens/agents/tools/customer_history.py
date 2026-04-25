"""Tool: customer transaction history lookup (mock — PostgreSQL in Week 5)."""

from __future__ import annotations

import json

from langchain_core.tools import tool


def _parse_card1(customer_id: str) -> int:
    """Extract integer from ACC-NNNN format; fall back to char sum."""
    parts = customer_id.split("-")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return sum(ord(c) for c in customer_id) % 10000


@tool
def get_customer_history(customer_id: str, days: int = 30) -> str:
    """Retrieve recent transaction history for a customer from the bank's core system.

    Use this tool to establish a behavioral baseline for the customer.
    High prior_suspicious_flags or sudden country diversification are red flags.

    Args:
        customer_id: The customer's account ID or sender_account_id.
        days: Lookback window in days (1–90, default 30).

    Returns:
        JSON string with transaction_count, average_amount_usd, prior_suspicious_flags,
        countries_transacted, account_age_days, and last_password_change_days_ago.
    """
    days = max(1, min(days, 90))
    c = _parse_card1(customer_id)

    prior_flags = c % 4
    tx_count = 10 + (c % 40)
    avg_amount = 50 + (c % 500)
    failed_logins = c % 5

    mod3 = c % 3
    if mod3 == 0:
        countries = ["TR"]
    elif mod3 == 1:
        countries = ["TR", "US"]
    else:
        countries = ["CN", "RU", "UA"]

    return json.dumps({
        "customer_id": customer_id,
        "lookback_days": days,
        "transaction_count": tx_count,
        "average_transaction_amount_usd": float(avg_amount),
        "max_single_transaction_usd": float(avg_amount * 3),
        "prior_suspicious_flags": prior_flags,
        "countries_transacted": countries,
        "account_age_days": 180 + (c % 3000),
        "last_password_change_days_ago": 30 + (c % 300),
        "failed_login_attempts_30d": failed_logins,
    })
