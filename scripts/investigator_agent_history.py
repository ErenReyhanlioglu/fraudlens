#!/usr/bin/env python
"""Show past Investigation Agent decisions from the database.

Usage:
    uv run scripts/investigator_agent_history.py [--limit N] [--hint HINT]
                                                  [--since DURATION] [--verbose]

Examples:
    uv run scripts/investigator_agent_history.py --limit 5
    uv run scripts/investigator_agent_history.py --hint suspicious --limit 10
    uv run scripts/investigator_agent_history.py --since 24h --verbose
    uv run scripts/investigator_agent_history.py --since 7d
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from textwrap import wrap

from sqlalchemy import text

from fraudlens.db.session import AsyncSessionFactory

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_C = "\033[96m"
_B = "\033[1m"
_D = "\033[2m"
_X = "\033[0m"


def _dim(s: str) -> str:
    return f"{_D}{s}{_X}"


def _cyan(s: str) -> str:
    return f"{_C}{s}{_X}"


def _hint_colour(hint: str | None) -> str:
    return {"likely_legitimate": _G, "suspicious": _R, "inconclusive": _Y}.get(hint or "", "")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_since(s: str) -> datetime:
    """Parse strings like '24h', '7d', '30m' into a UTC cutoff datetime."""
    m = re.fullmatch(r"(\d+)([hdm])", s.lower())
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid --since '{s}'. Use formats like '24h', '7d', '30m'.")
    n, unit = int(m.group(1)), m.group(2)
    delta = {"h": timedelta(hours=n), "d": timedelta(days=n), "m": timedelta(minutes=n)}[unit]
    return datetime.now(UTC) - delta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show past Investigation Agent decisions")
    p.add_argument("--limit", type=int, default=10, help="Max records to display (default 10)")
    p.add_argument("--hint", choices=["suspicious", "inconclusive", "likely_legitimate"], help="Filter by decision_hint")
    p.add_argument("--since", type=_parse_since, help="Only show records after this duration (e.g. 24h, 7d)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show full tool trace and reasoning")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_record(idx: int, row: dict, verbose: bool) -> None:
    hint = row.get("decision_hint") or "?"
    hc = _hint_colour(hint)
    score = row["fraud_probability"]
    conf = row.get("confidence") or 0.0
    tools = row.get("tools_called") or []
    evidence = row.get("evidence") or []
    red_flags = row.get("red_flags") or []
    reasoning = row.get("reasoning") or ""
    tool_trace = row.get("tool_trace") or []
    created = row["created_at"]

    print(f"\n{_B}#{idx} [{created.strftime('%Y-%m-%d %H:%M:%S')}]{_X}  "
          f"score={score:.4f}  {hc}{hint}{_X}  conf={conf:.2f}  tools={len(tools)}")
    print(f"  {_dim('tx_id:')} {row['transaction_id']}")
    print(f"  {_dim('tools called:')} {' → '.join(tools) if tools else _dim('(none)')}")

    if evidence:
        print(f"  {_dim('evidence:')}")
        for e in evidence[: 8 if verbose else 3]:
            print(f"    • {e}")
        if not verbose and len(evidence) > 3:
            print(f"    {_dim(f'... +{len(evidence) - 3} more')}")

    if red_flags:
        print(f"  {_dim('red flags:')}")
        for rf in red_flags[: 8 if verbose else 3]:
            print(f"    {_R}•{_X} {rf}")
        if not verbose and len(red_flags) > 3:
            print(f"    {_dim(f'... +{len(red_flags) - 3} more')}")

    if reasoning:
        text_to_print = reasoning if verbose else reasoning[:200] + ("..." if len(reasoning) > 200 else "")
        wrapped = wrap(text_to_print, width=85)
        print(f"  {_dim('reasoning:')}")
        for line in wrapped:
            print(f"    {line}")

    if verbose and tool_trace:
        print(f"  {_dim('tool trace:')}")
        for i, tc in enumerate(tool_trace, 1):
            name = tc.get("tool", "?")
            args = tc.get("args") or {}
            result = str(tc.get("result", ""))[:200]
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if isinstance(args, dict) else str(args)
            print(f"    {i}. {_cyan(name)}({_dim(args_str)})")
            print(f"       {_dim('→')} {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _async_main() -> int:
    args = _parse_args()

    where = ["agent_used = :agent"]
    params: dict = {"agent": "investigation", "limit": args.limit}
    if args.hint:
        where.append("decision_hint = :hint")
        params["hint"] = args.hint
    if args.since:
        where.append("created_at >= :since")
        params["since"] = args.since

    sql = f"""
        SELECT id, transaction_id, fraud_probability, decision_hint, confidence,
               reasoning, evidence, red_flags, tools_called, tool_trace, created_at
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
    print("  FraudLens — Investigation Agent History")
    filters = []
    if args.hint:
        filters.append(f"hint={args.hint}")
    if args.since:
        filters.append(f"since={args.since.strftime('%Y-%m-%d %H:%M')}")
    filter_str = f"  |  {' '.join(filters)}" if filters else ""
    print(f"  {len(rows)} record(s){filter_str}")
    print(f"{line}{_X}")

    if not rows:
        print(f"\n  {_dim('No matching records found.')}\n")
        return 0

    for idx, row in enumerate(rows, 1):
        # JSONB columns come back as already-parsed lists/dicts; normalise just in case
        record = dict(row)
        for fld in ("evidence", "red_flags", "tools_called", "tool_trace"):
            v = record.get(fld)
            if isinstance(v, str):
                try:
                    record[fld] = json.loads(v)
                except json.JSONDecodeError:
                    record[fld] = []
        _print_record(idx, record, args.verbose)

    print()
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
