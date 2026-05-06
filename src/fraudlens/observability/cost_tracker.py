"""Per-request token cost tracking backed by Redis daily accumulators.

Pricing is read from Settings so it can be overridden per environment.
Daily totals live at Redis key ``cost:YYYY-MM-DD`` as a float (USD string).
GET /observability/cost returns today + last 7 days.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import redis.asyncio as aioredis
import structlog

from fraudlens.core.config import get_settings

logger = structlog.get_logger(__name__)


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str = "") -> float:
    """Compute USD cost for a single LLM call.

    Falls back to Haiku pricing for unknown models. Prices are per million tokens.
    """
    settings = get_settings()
    cfg = settings.llm_pricing

    model_lower = model.lower()
    prices = cfg.get("sonnet", cfg["haiku"]) if "sonnet" in model_lower else cfg["haiku"]

    return (
        input_tokens * prices["input_per_m"] + output_tokens * prices["output_per_m"]
    ) / 1_000_000


def _today_key() -> str:
    return "cost:" + datetime.now(UTC).strftime("%Y-%m-%d")


def _date_key(date: datetime) -> str:
    return "cost:" + date.strftime("%Y-%m-%d")


async def _get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def accumulate_cost(cost_usd: float) -> None:
    """Add cost_usd to today's Redis accumulator. Fire-and-forget; errors are logged only."""
    if cost_usd <= 0:
        return
    try:
        r = await _get_redis()
        await r.incrbyfloat(_today_key(), cost_usd)
        await r.expire(_today_key(), 60 * 60 * 24 * 10)  # keep 10 days
        await r.aclose()
    except Exception as exc:
        logger.warning("cost_accumulate_failed", error=str(exc))


async def get_daily_costs(days: int = 7) -> list[dict[str, Any]]:
    """Return daily cost totals for today + the previous `days` days.

    Each entry: ``{date: "YYYY-MM-DD", cost_usd: float}``.
    Missing days (no traffic) are returned as 0.0.
    """
    today = datetime.now(UTC)
    dates = [today - timedelta(days=i) for i in range(days)]

    try:
        r = await _get_redis()
        keys = [_date_key(d) for d in dates]
        values = await r.mget(*keys)
        await r.aclose()
    except Exception as exc:
        logger.warning("cost_fetch_failed", error=str(exc))
        values = [None] * len(dates)

    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "cost_usd": float(v) if v else 0.0,
        }
        for d, v in zip(dates, values, strict=False)
    ]


def build_token_usage_dict(
    input_tokens: int,
    output_tokens: int,
    model: str = "",
) -> dict[str, Any]:
    """Build a token_usage dict suitable for inclusion in API responses."""
    cost = estimate_cost_usd(input_tokens, output_tokens, model)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost, 8),
        "model": model,
    }
