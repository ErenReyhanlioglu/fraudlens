#!/usr/bin/env python
"""Show past Critical Agent decisions from the database.

Includes critical-tier extras: mandatory tools check, sanctions/PEP highlights,
RAG citation extraction, and tool-trace formatting for network/RAG/adverse_media.

Usage:
    uv run scripts/critical_agent_history.py [--limit N] [--hint HINT]
                                              [--since DURATION] [--verbose]

Examples:
    uv run scripts/critical_agent_history.py --limit 5
    uv run scripts/critical_agent_history.py --hint suspicious --verbose
    uv run scripts/critical_agent_history.py --since 24h
    uv run scripts/critical_agent_history.py --since 7d --verbose
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
_M = "\033[95m"
_B = "\033[1m"
_D = "\033[2m"
_X = "\033[0m"

MANDATORY_TOOLS = {"get_customer_history", "adverse_media_search", "deep_network_analysis"}


def _dim(s: str) -> str:
    return f"{_D}{s}{_X}"


def _cyan(s: str) -> str:
    return f"{_C}{s}{_X}"


def _magenta(s: str) -> str:
    return f"{_M}{s}{_X}"


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
    p = argparse.ArgumentParser(description="Show past Critical Agent decisions")
    p.add_argument("--limit", type=int, default=10, help="Max records to display (default 10)")
    p.add_argument("--hint", choices=["suspicious", "inconclusive", "likely_legitimate"], help="Filter by decision_hint")
    p.add_argument("--since", type=_parse_since, help="Only show records after this duration (e.g. 24h, 7d)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show full tool trace and reasoning")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tool result formatting
# ---------------------------------------------------------------------------


def _fmt_tool_result(tool: str, result_str: str) -> list[str]:
    """Format a tool's raw JSON result for display."""
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, ValueError, TypeError):
        return [_dim(str(result_str)[:140])]

    lines: list[str] = []

    if tool == "deep_network_analysis":
        rl = data.get("risk_level", "?")
        c = _R if rl == "high" else (_Y if rl == "medium" else _G)
        lines.append(
            f"nodes={data.get('node_count')}  edges={data.get('edge_count')}  "
            f"density={data.get('graph_density')}  risk={c}{rl}{_X}"
        )
        sigs = data.get("risk_signals") or []
        if sigs:
            lines.append(f"signals={sigs}")

    elif tool == "regulatory_policy_rag":
        excerpts = data.get("excerpts") or []
        if not excerpts:
            lines.append(_dim(data.get("message", "no results")))
        for ex in excerpts[:2]:
            lines.append(f"{_magenta(ex.get('citation', '?'))}  score={ex.get('relevance_score', 0)}")
            lines.append(f"  {_dim(ex.get('text', '')[:90].replace(chr(10), ' '))}...")

    elif tool == "adverse_media_search":
        rl = data.get("overall_risk_level", "?")
        c = _R if rl in ("critical", "high") else _G
        lines.append(
            f"risk={c}{rl}{_X}  "
            f"sanctions={_R + 'YES' + _X if data.get('sanctions_match') else 'no'}  "
            f"pep={_Y + 'YES' + _X if data.get('pep_flag') else 'no'}  "
            f"adverse={data.get('adverse_media_hit_count', 0)}"
        )
        hits = data.get("sanctions_list_hits") or []
        if hits:
            lines.append(f"{_R}SANCTIONS HIT: {hits}{_X}")

    elif tool == "get_customer_history":
        lines.append(
            f"tx={data.get('transaction_count')}  avg=${data.get('average_transaction_amount_usd')}  "
            f"prior_flags={data.get('prior_suspicious_flags')}  countries={data.get('countries_transacted')}"
        )

    elif tool == "explain_ml_score":
        for feat in (data.get("top_features") or [])[:3]:
            v = feat.get("shap_contribution", 0)
            sign = "+" if v > 0 else ""
            d = "→ fraud" if v > 0 else "→ legit"
            lines.append(f"{feat.get('feature')} = {sign}{v:.4f}  {_dim(d)}")

    elif tool == "check_merchant_reputation":
        lines.append(f"risk_score={data.get('risk_score')}  category={data.get('industry_category')}")

    elif tool == "get_geolocation_context":
        lines.append(f"ip_country={data.get('ip_country')}  vpn={data.get('vpn_detected')}")

    elif tool == "find_similar_patterns":
        lines.append(f"matches={data.get('match_count')}  patterns={data.get('patterns')}")

    return lines or [_dim(str(result_str)[:140])]


def _extract_citations(evidence: list[str]) -> list[str]:
    """Pull '<file>.pdf, p.NN' style citations from evidence strings."""
    pattern = re.compile(r"[\w_]+\.pdf,?\s*p\.?\s*\d+", re.IGNORECASE)
    found: list[str] = []
    for item in evidence:
        for m in pattern.findall(str(item)):
            if m not in found:
                found.append(m)
    return found


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

    tools_set = set(tools)
    missing = MANDATORY_TOOLS - tools_set
    mandatory_tag = (
        f"{_G}✓ all mandatory{_X}"
        if not missing
        else f"{_R}✗ missing: {sorted(missing)}{_X}"
    )

    print(f"\n{_B}#{idx} [{created.strftime('%Y-%m-%d %H:%M:%S')}]{_X}  "
          f"score={score:.4f}  {hc}{hint}{_X}  conf={conf:.2f}  tools={len(tools)}/8")
    print(f"  {_dim('tx_id:')} {row['transaction_id']}")
    print(f"  {_dim('mandatory:')} {mandatory_tag}")
    print(f"  {_dim('tools called:')} {' → '.join(tools) if tools else _dim('(none)')}")

    citations = _extract_citations(evidence)
    if citations:
        print(f"  {_dim('regulatory citations:')}")
        for c in citations[:5]:
            print(f"    {_magenta('§')} {c}")

    if evidence:
        print(f"  {_dim('evidence:')}")
        for e in evidence[: 10 if verbose else 4]:
            print(f"    • {e}")
        if not verbose and len(evidence) > 4:
            print(f"    {_dim(f'... +{len(evidence) - 4} more')}")

    if red_flags:
        print(f"  {_dim('red flags:')}")
        for rf in red_flags[: 10 if verbose else 4]:
            print(f"    {_R}•{_X} {rf}")
        if not verbose and len(red_flags) > 4:
            print(f"    {_dim(f'... +{len(red_flags) - 4} more')}")

    if reasoning:
        text_to_print = reasoning if verbose else reasoning[:240] + ("..." if len(reasoning) > 240 else "")
        wrapped = wrap(text_to_print, width=85)
        print(f"  {_dim('reasoning:')}")
        for line in wrapped:
            print(f"    {line}")

    if verbose and tool_trace:
        print(f"  {_dim('tool trace:')}")
        for i, tc in enumerate(tool_trace, 1):
            name = tc.get("tool", "?")
            args = tc.get("args") or {}
            result = str(tc.get("result", ""))
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if isinstance(args, dict) else str(args)
            print(f"    {i}. {_cyan(name)}({_dim(args_str)})")
            for line in _fmt_tool_result(name, result):
                print(f"       {_dim('→')} {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _async_main() -> int:
    args = _parse_args()

    where = ["agent_used = :agent"]
    params: dict = {"agent": "critical", "limit": args.limit}
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

    line = "═" * 62
    print(f"\n{_B}{line}")
    print("  FraudLens — Critical Agent History")
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
