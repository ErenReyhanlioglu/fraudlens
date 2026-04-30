"""GET /observability/cost — daily LLM cost summary."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from fraudlens.observability.cost_tracker import get_daily_costs

router = APIRouter(tags=["observability"])


@router.get("/observability/cost")
async def get_cost_summary() -> dict[str, Any]:
    """Return today's and the last 7 days' estimated LLM costs (USD).

    Costs are accumulated via Redis by the cost_tracker module.
    Missing days (no traffic) are returned as 0.0.
    """
    daily = await get_daily_costs(days=7)
    total = sum(d["cost_usd"] for d in daily)
    return {
        "today": daily[0] if daily else {"date": "unknown", "cost_usd": 0.0},
        "last_7_days": daily,
        "total_7d_usd": round(total, 4),
    }
