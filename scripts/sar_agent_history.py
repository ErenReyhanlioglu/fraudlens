#!/usr/bin/env python
"""Show past SAR reports from the database.

Queries decisions where sar_report IS NOT NULL (outcome = escalate).
Displays indicators, regulatory triggers, recommended action, and more.

Usage:
    uv run scripts/sar_agent_history.py [--limit N] [--since DURATION] [--verbose]

Examples:
    uv run scripts/sar_agent_history.py --limit 5
    uv run scripts/sar_agent_history.py --since 24h
    uv run scripts/sar_agent_history.py --since 7d --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import wrap

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudlens.db.session import AsyncSessionFactory  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_M = "\033[95m"
_B = "\033[1m"
_D = "\033[2m"
_X = "\033[0m"


def _dim(s: str) -> str:
    return f"{_D}{s}{_X}"


def _magenta(s: str) -> str:
    return f"{_M}{s}{_X}"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_since(s: str) -> datetime:
    m = re.fullmatch(r"(\d+)([hdm])", s.lower())
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid --since '{s}'. Use formats like '24h', '7d', '30m'.")
    n, unit = int(m.group(1)), m.group(2)
    delta = {"h": timedelta(hours=n), "d": timedelta(days=n), "m": timedelta(minutes=n)}[unit]
    return datetime.now(UTC) - delta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show past SAR reports from the database")
    p.add_argument("--limit", type=int, default=10, help="Max records to display (default 10)")
    p.add_argument("--since", type=_parse_since, help="Only show records after this duration (e.g. 24h, 7d, 30m)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show full SAR fields: customer_info, transaction_details, investigation_summary")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_record(idx: int, row: dict, verbose: bool) -> None:
    created: datetime = row["created_at"]
    tx_id = str(row["transaction_id"])
    score = row["fraud_probability"]
    outcome = row.get("outcome", "?")
    hint = row.get("decision_hint") or "?"
    conf = row.get("confidence") or 0.0
    sar: dict = row.get("sar_report") or {}

    indicators: list[str] = sar.get("suspicious_indicators") or []
    triggers: list[str] = sar.get("regulatory_triggers") or []
    action: str = sar.get("recommended_action") or ""
    summary: str = sar.get("investigation_summary") or ""
    model: str = sar.get("agent_model") or "?"
    generated_at: str = sar.get("generated_at") or ""
    customer_info: dict = sar.get("customer_info") or {}
    tx_details: dict = sar.get("transaction_details") or {}

    hint_color = {"likely_legitimate": _G, "suspicious": _R, "inconclusive": _Y}.get(hint, "")
    outcome_color = _R if outcome == "escalate" else _G

    print(
        f"\n{_B}#{idx} [{created.strftime('%Y-%m-%d %H:%M:%S')}]{_X}  "
        f"score={score:.4f}  {hint_color}{hint}{_X}  conf={conf:.2f}  "
        f"outcome={outcome_color}{outcome}{_X}"
    )
    print(f"  {_dim('tx_id   :')} {tx_id}")
    print(f"  {_dim('model   :')} {model}  {_dim('generated:')} {generated_at[:19] if generated_at else '?'}")

    if indicators:
        print(f"  {_dim('indicators:')}")
        for ind in indicators[: 10 if verbose else 4]:
            print(f"    {_R}•{_X} {ind}")
        if not verbose and len(indicators) > 4:
            print(f"    {_dim(f'... +{len(indicators) - 4} more')}")
    else:
        print(f"  {_dim('indicators:')} {_dim('(none)')}")

    if triggers:
        print(f"  {_dim('reg. triggers:')}")
        for t in triggers[: 6 if verbose else 3]:
            print(f"    {_magenta('§')} {t}")
        if not verbose and len(triggers) > 3:
            print(f"    {_dim(f'... +{len(triggers) - 3} more')}")
    else:
        print(f"  {_dim('reg. triggers:')} {_dim('(none)')}")

    if action:
        wrapped = wrap(action, width=85)
        print(f"  {_dim('action  :')} {wrapped[0]}")
        for line in wrapped[1:]:
            print(f"             {line}")

    if verbose:
        if summary:
            print(f"  {_dim('investigation summary:')}")
            for line in wrap(summary, width=85):
                print(f"    {line}")

        if customer_info:
            print(f"  {_dim('customer info:')}")
            for k, v in customer_info.items():
                print(f"    • {k}: {v}")

        if tx_details:
            print(f"  {_dim('transaction details:')}")
            for k, v in tx_details.items():
                print(f"    • {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _async_main() -> int:
    args = _parse_args()

    where = ["sar_report IS NOT NULL"]
    params: dict = {"limit": args.limit}
    if args.since:
        where.append("created_at >= :since")
        params["since"] = args.since

    sql = f"""
        SELECT id, transaction_id, fraud_probability, outcome, decision_hint, confidence,
               sar_report, created_at
        FROM decisions
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC
        LIMIT :limit
    """

    async with AsyncSessionFactory() as db:
        result = await db.execute(text(sql), params)
        rows = result.mappings().all()

    line = "═" * 60
    print(f"\n{_B}{line}")
    print("  FraudLens — SAR Agent History")
    filters = []
    if args.since:
        filters.append(f"since={args.since.strftime('%Y-%m-%d %H:%M')}")
    filter_str = f"  |  {' '.join(filters)}" if filters else ""
    print(f"  {len(rows)} SAR record(s){filter_str}")
    print(f"{line}{_X}")

    if not rows:
        print(f"\n  {_dim('No SAR reports found. Run a critical-tier transaction to generate one.')}\n")
        return 0

    for idx, row in enumerate(rows, 1):
        record = dict(row)
        sar = record.get("sar_report")
        if isinstance(sar, str):
            try:
                record["sar_report"] = json.loads(sar)
            except json.JSONDecodeError:
                record["sar_report"] = {}
        _print_record(idx, record, args.verbose)

    print()
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
