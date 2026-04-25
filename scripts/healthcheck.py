#!/usr/bin/env python
"""FraudLens end-to-end smoke test / health probe.

Usage:
    uv run scripts/healthcheck.py [--port PORT] [--host HOST] [--no-langsmith]
                                  [--reliability-samples N] [--seed S]

Sections:
  INFRASTRUCTURE        — TCP reachability for PostgreSQL and Redis
  TRIAGE ROUTING        — 1 POST per bucket, validates expected triage_action
  AGENT INVESTIGATION   — detailed tool trace + verdict for investigate/escalate buckets
  STRUCTURED OUTPUT     — validates InvestigationResult fields on 5 investigate transactions
  LANGSMITH             — verifies traces appeared in the fraudlens project

Exit code: 0 = all checks passed, 1 = any failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import sys
import time
import uuid
from pathlib import Path
from textwrap import wrap
from typing import Any

import httpx

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_PATH = REPO_ROOT / "data" / "processed" / "test_scenarios.jsonl"
ENV_PATH = REPO_ROOT / ".env"

BUCKET_ORDER = [
    "investigate_30_40",
    "investigate_40_50",
    "investigate_50_60",
    "investigate_60_70",
]
BUCKET_TRIAGE = {
    "approve_low":       "approve",
    "approve_high":      "approve",
    "investigate_30_40": "investigate",
    "investigate_40_50": "investigate",
    "investigate_50_60": "investigate",
    "investigate_60_70": "investigate",
    "critical_low":      "escalate",
    "critical_high":     "escalate",
}
AGENT_BUCKETS = {"investigate_30_40", "investigate_40_50", "investigate_50_60", "investigate_60_70", "critical_low", "critical_high"}
INVESTIGATE_BUCKETS = {"investigate_30_40", "investigate_40_50", "investigate_50_60", "investigate_60_70"}

_TX_BASE: dict[str, Any] = {
    "timestamp": "2024-01-15T10:00:00Z",
    "amount": 100.0,
    "currency": "TRY",
    "transaction_type": "payment",
    "channel": "api",
    "sender_account_id": "TEST-SENDER-001",
    "sender_bank_code": "TEST01",
    "sender_country": "TR",
    "receiver_account_id": "TEST-RECEIVER-001",
    "receiver_bank_code": "TEST02",
    "receiver_country": "TR",
}

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


def _ok(s: str) -> str:
    return f"{_G}✓{_X} {s}"


def _fail(s: str) -> str:
    return f"{_R}✗{_X} {s}"


def _warn(s: str) -> str:
    return f"{_Y}⚠{_X} {s}"


def _dim(s: str) -> str:
    return f"{_D}{s}{_X}"


def _cyan(s: str) -> str:
    return f"{_C}{s}{_X}"


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------


def load_scenarios() -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {b: [] for b in BUCKET_ORDER}
    with SCENARIOS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            b = row["expected_bucket"]
            if b in buckets:
                buckets[b].append(row)
    return buckets


def make_payload(scenario: dict[str, Any]) -> dict[str, Any]:
    return {**_TX_BASE, "transaction_id": str(uuid.uuid4()), "raw_features": scenario["raw_features"]}


# ---------------------------------------------------------------------------
# Infrastructure checks
# ---------------------------------------------------------------------------


def _check_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_infra_checks(env: dict[str, str]) -> tuple[int, int]:
    print(f"\n{_B}[INFRASTRUCTURE]{_X}")
    passed = 0
    total = 0
    checks = [
        ("PostgreSQL", env.get("POSTGRES_HOST", "localhost"), int(env.get("POSTGRES_PORT", "5432"))),
        ("Redis", env.get("REDIS_HOST", "localhost"), int(env.get("REDIS_PORT", "6379"))),
    ]
    for name, host, port in checks:
        total += 1
        alive = _check_tcp(host, port)
        label = f"{name:<16} {host}:{port}"
        if alive:
            passed += 1
            print(f"  {_ok(f'{label}  healthy')}")
        else:
            print(f"  {_fail(f'{label}  not reachable')}")
    return passed, total


def run_health_check(base_url: str) -> tuple[int, int]:
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        model_tag = "model_loaded=True" if data.get("model_loaded") == "True" else f"{_Y}model_loaded=False{_X}"
        print(f"  {_ok(f'API /health         {resp.status_code} OK  {model_tag}')}")
        return 1, 1
    except Exception as exc:
        print(f"  {_fail(f'API /health         {exc}')}")
        return 0, 1


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _post_one(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    timeout: float = 90.0,  # noqa: ASYNC109
) -> dict[str, Any]:
    try:
        resp = await client.post(
            f"{base_url}/api/v1/transactions",
            params={"raw_mode": "true"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# Triage routing section
# ---------------------------------------------------------------------------


def print_routing_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[int, int]:
    print(f"\n{_B}[TRIAGE ROUTING]{_X}")
    passed = 0
    total = 0
    for bucket, scenario, api_resp in results:
        total += 1
        expected = BUCKET_TRIAGE[bucket]
        if "_error" in api_resp:
            err = api_resp["_error"][:60]
            print(f"  {_fail(f'{bucket:<22} API error: {err}')}")
            continue

        actual = api_resp.get("triage_action", "?")
        prob = api_resp.get("fraud_probability", -1.0)
        inv = api_resp.get("investigation")
        hint_tag = f"  hint={inv['decision_hint']}" if inv and inv.get("decision_hint") else ""
        inv_tag = "  investigation=None" if inv is None and actual == "approve" else ""
        is_ok = actual == expected

        parts = [f"score={prob:.3f}", f"action={actual}", f"is_fraud={scenario['is_fraud']}"]
        line = f"{bucket:<22}  {'  '.join(parts)}{hint_tag}{inv_tag}"
        if is_ok:
            passed += 1
            print(f"  {_ok(line)}")
        else:
            print(f"  {_fail(line)}")
            print(f"       {_dim(f'expected action={expected}  got={actual}')}")
    return passed, total


# ---------------------------------------------------------------------------
# Tool result formatting
# ---------------------------------------------------------------------------


def _fmt_tool_result(tool: str, result_str: str) -> list[str]:
    """Return 1-3 display lines summarising a tool's raw JSON result."""
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, ValueError, TypeError):
        return [_dim(str(result_str)[:100])]

    lines: list[str] = []
    if tool == "explain_ml_score":
        for feat in (data.get("top_features") or [])[:3]:
            val = feat.get("shap_contribution", 0)
            sign = "+" if val > 0 else ""
            direction = "→ fraud" if val > 0 else "→ legit"
            lines.append(f"{feat['feature']} = {sign}{val:.4f}  {_dim(direction)}")
    elif tool == "get_customer_history":
        lines.append(
            f"tx_count={data.get('transaction_count')}  "
            f"avg_amount=${data.get('average_transaction_amount_usd')}  "
            f"prior_flags={data.get('prior_suspicious_flags')}"
        )
        lines.append(f"countries={data.get('countries_transacted')}")
    elif tool == "check_merchant_reputation":
        lines.append(
            f"risk_score={data.get('risk_score')}  "
            f"category={data.get('industry_category')}  "
            f"chargebacks={data.get('chargeback_rate_pct')}%"
        )
        flags = data.get("flags") or []
        if flags:
            lines.append(f"flags={flags}")
    elif tool == "get_geolocation_context":
        lines.append(
            f"ip_country={data.get('ip_country')}  "
            f"vpn={data.get('vpn_detected')}  "
            f"impossible_travel={data.get('impossible_travel')}"
        )
        signals = data.get("risk_signals") or []
        if signals:
            lines.append(f"signals={signals}")
    elif tool == "find_similar_patterns":
        cases = data.get("similar_cases") or []
        if cases:
            top = cases[0]
            lines.append(
                f"{top.get('case_id')}  sim={top.get('similarity_score')}  "
                f"fraud_confirmed={top.get('fraud_confirmed')}  "
                f"modus={top.get('modus_operandi')}"
            )
    return lines or [_dim(str(result_str)[:100])]


# ---------------------------------------------------------------------------
# Agent investigation section
# ---------------------------------------------------------------------------


def _print_single_agent(bucket: str, scenario: dict[str, Any], api_resp: dict[str, Any]) -> None:
    inv = api_resp.get("investigation")
    prob = api_resp.get("fraud_probability", 0.0)

    is_fraud = scenario["is_fraud"]
    print(f"\n  {_B}[AGENT INVESTIGATION — {bucket}]{_X}  {_dim(f'score={prob:.4f}  is_fraud={is_fraud}')}")

    if not inv:
        print(f"  {_warn('investigation=None (agent did not run or failed)')}")
        return

    hint = inv.get("decision_hint", "?")
    conf = inv.get("confidence", 0.0)
    tools = inv.get("tools_called") or []
    evidence = inv.get("evidence") or []
    red_flags = inv.get("red_flags") or []
    summary = inv.get("reasoning_summary", "")
    tool_trace = inv.get("tool_trace") or []

    hint_color = {"likely_legitimate": _G, "suspicious": _R, "inconclusive": _Y}.get(hint, "")

    print(f"  Decision hint  : {hint_color}{hint}{_X}")
    print(f"  Confidence     : {conf:.2f}")
    print(f"  Tools called   : {' → '.join(tools) if tools else _dim('(none)')}")

    if tool_trace:
        print(f"\n  {_D}Tool trace:{_X}")
        for i, tc in enumerate(tool_trace, 1):
            name = tc.get("tool", "?")
            args = tc.get("args") or {}
            result_raw = tc.get("result", "")
            args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
            print(f"    {i}. {_cyan(name)}({_dim(args_str)})")
            for line in _fmt_tool_result(name, result_raw):
                print(f"       {_dim('→')} {line}")

    if evidence:
        print("\n  Evidence       :")
        for e in evidence[:4]:
            print(f"    • {e}")

    if red_flags:
        print("  Red flags      :")
        for rf in red_flags[:4]:
            print(f"    {_R}•{_X} {rf}")

    if summary:
        wrapped = wrap(summary, width=72)
        print(f"  Summary        : {wrapped[0]}")
        for extra in wrapped[1:]:
            print(f"                   {extra}")


def print_agent_section(
    routing_results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> None:
    agent_results = [(b, s, r) for b, s, r in routing_results if b in AGENT_BUCKETS and "_error" not in r]
    if not agent_results:
        print(f"  {_warn('No agent results — all agent buckets errored')}")
        return
    for bucket, scenario, api_resp in agent_results:
        _print_single_agent(bucket, scenario, api_resp)


# ---------------------------------------------------------------------------
# Structured output reliability
# ---------------------------------------------------------------------------

_VALID_HINTS = {"likely_legitimate", "suspicious", "inconclusive"}


def _validate_investigation(inv: dict[str, Any] | None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not inv:
        return False, ["investigation is None"]
    if inv.get("decision_hint") not in _VALID_HINTS:
        issues.append(f"decision_hint={inv.get('decision_hint')!r} not valid")
    conf = inv.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        issues.append(f"confidence={conf!r} not in [0, 1]")
    if not inv.get("evidence"):
        issues.append("evidence list is empty")
    if not inv.get("tools_called"):
        issues.append("tools_called is empty")
    # Detect fallback: inconclusive with no tools = agent failed silently
    if inv.get("decision_hint") == "inconclusive" and not inv.get("tools_called"):
        issues.append("fallback INCONCLUSIVE (no tools called)")
    return len(issues) == 0, issues


def print_reliability_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[int, int]:
    print(f"\n{_B}[STRUCTURED OUTPUT RELIABILITY]{_X}")
    passed = 0
    total = 0
    for idx, (bucket, _scenario, api_resp) in enumerate(results, 1):
        total += 1
        if "_error" in api_resp:
            err = api_resp["_error"][:60]
            print(f"  {_fail(f'#{idx} {bucket:<20}  API error: {err}')}")
            continue

        inv = api_resp.get("investigation")
        ok, issues = _validate_investigation(inv)
        hint = (inv or {}).get("decision_hint", "?")
        conf = (inv or {}).get("confidence", 0.0)
        tools_n = len((inv or {}).get("tools_called") or [])
        prob = api_resp.get("fraud_probability", 0.0)

        label = f"#{idx} {bucket:<20}  hint={hint:<20} conf={conf:.2f}  tools={tools_n}  score={prob:.3f}"
        if ok:
            passed += 1
            print(f"  {_ok(label)}")
        else:
            print(f"  {_fail(label)}")
            for issue in issues:
                print(f"       {_dim('→')} {issue}")
    return passed, total


# ---------------------------------------------------------------------------
# LangSmith check
# ---------------------------------------------------------------------------


def run_langsmith_check(env: dict[str, str]) -> tuple[int, int]:
    print(f"\n{_B}[LANGSMITH]{_X}")
    tracing = env.get("LANGSMITH_TRACING", "false").lower() == "true"
    api_key = env.get("LANGSMITH_API_KEY", "")
    project = env.get("LANGSMITH_PROJECT", "fraudlens")

    if not tracing:
        print(f"  {_warn('LANGSMITH_TRACING=false — tracing disabled, skipping check')}")
        return 0, 0

    if not api_key:
        print(f"  {_warn('LANGSMITH_API_KEY not set — cannot query traces')}")
        return 0, 1

    try:
        from langsmith import Client  # type: ignore[import-untyped]

        client = Client(api_key=api_key)
        runs = list(client.list_runs(project_name=project, limit=10))
        count = len(runs)
        if count > 0:
            print(f"  {_ok(f'{project} project   {count} trace(s) found in last 10')}")
            return 1, 1
        else:
            print(f"  {_warn(f'{project} project   0 traces found (agent may not have run yet)')}")
            return 0, 1
    except Exception as exc:
        print(f"  {_fail(f'LangSmith query failed: {exc}')}")
        return 0, 1


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------


def _header() -> None:
    line = "═" * 51
    print(f"\n{_B}{line}")
    print("  FraudLens Integration Probe")
    print(f"{line}{_X}")


def _footer(passed: int, total: int, errors: int, elapsed: float) -> None:
    line = "═" * 51
    failed = total - passed - errors
    color = _G if passed == total else _R
    print(f"\n{_B}{line}")
    print(f"  {color}Results: {passed}/{total} passed  •  {failed} failed  •  {errors} errors{_X}")
    print(f"  {_dim(f'Elapsed: {elapsed:.1f}s')}")
    print(f"{_B}{line}{_X}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FraudLens smoke test")
    p.add_argument("--port", type=int, default=8001, help="API port (default 8001)")
    p.add_argument("--host", default="127.0.0.1", help="API host (default 127.0.0.1)")
    p.add_argument("--no-langsmith", action="store_true", help="Skip LangSmith check")
    p.add_argument("--reliability-samples", type=int, default=0, help="Extra investigate txns for reliability check (default 0 for free tier)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for scenario selection")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _async_main() -> int:
    args = _parse_args()
    base_url = f"http://{args.host}:{args.port}"
    t_start = time.perf_counter()

    # Load .env without importing fraudlens (standalone script)
    env: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")

    _header()

    if not SCENARIOS_PATH.exists():
        print(f"\n{_R}ERROR: {SCENARIOS_PATH} not found.{_X}")
        print("       Run: uv run scripts/create_test_scenarios.py")
        return 1

    scenarios = load_scenarios()
    rng = random.Random(args.seed)

    # ---- Infrastructure ----
    infra_pass, infra_total = run_infra_checks(env)
    api_pass, api_total = run_health_check(base_url)
    total_passed = infra_pass + api_pass
    total_checks = infra_total + api_total

    # ---- Build payloads ----
    # 1 per bucket for routing
    routing_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (bucket, row := rng.choice(scenarios[bucket]), make_payload(row))
        for bucket in BUCKET_ORDER
    ]

    # Extra investigate scenarios for reliability (pick from full pool)
    invest_pool = (
        scenarios["investigate_30_40"] + scenarios["investigate_40_50"]
        + scenarios["investigate_50_60"] + scenarios["investigate_60_70"]
    )
    extra_scenarios = rng.sample(invest_pool, min(args.reliability_samples, len(invest_pool)))
    reliability_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (row["expected_bucket"], row, make_payload(row)) for row in extra_scenarios
    ]

    total_requests = len(routing_pairs) + len(reliability_pairs)

    # ---- Sequential POST ----
    print(f"\n  {_dim(f'Dispatching {total_requests} requests to {base_url} (sequentially)...')}")

    routing_resps = []
    reliability_resps = []

    async with httpx.AsyncClient() as client:
        for bucket, _scenario, p in routing_pairs:
            resp = await _post_one(client, base_url, p)
            routing_resps.append(resp)

            if "_error" in resp:
                err_msg = resp["_error"][:40]
                print(f"    {_fail(f'{bucket:<20} API error: {err_msg}')}")
            else:
                action = resp.get("triage_action", "?")
                prob = resp.get("fraud_probability", 0.0)
                hint = (resp.get("investigation") or {}).get("decision_hint", "none")
                print(f"    {_ok(f'{bucket:<20} score={prob:.3f} action={action} hint={hint}')}")

        for i, (bucket, _scenario, p) in enumerate(reliability_pairs):
            resp = await _post_one(client, base_url, p)
            reliability_resps.append(resp)

            if "_error" in resp:
                err_msg = resp["_error"][:40]
                print(f"    {_fail(f'[Reliability #{i+1}] {bucket:<15} API error: {err_msg}')}")
            else:
                action = resp.get("triage_action", "?")
                prob = resp.get("fraud_probability", 0.0)
                hint = (resp.get("investigation") or {}).get("decision_hint", "none")
                print(f"    {_ok(f'[Reliability #{i+1}] {bucket:<15} score={prob:.3f} action={action} hint={hint}')}")

    routing_results = [(b, s, r) for (b, s, _), r in zip(routing_pairs, routing_resps, strict=True)]
    reliability_results = [(b, s, r) for (b, s, _), r in zip(reliability_pairs, reliability_resps, strict=True)]

    # ---- Triage routing ----
    r_pass, r_total = print_routing_section(routing_results)
    total_passed += r_pass
    total_checks += r_total

    # ---- Agent investigation (display only, no separate pass/fail) ----
    print(f"\n{_B}[AGENT INVESTIGATION]{_X}")
    print_agent_section(routing_results)

    # ---- Structured output reliability ----
    # Include the investigate results from routing + extra reliability samples
    routing_invest = [(b, s, r) for b, s, r in routing_results if b in INVESTIGATE_BUCKETS]
    all_reliability = routing_invest + reliability_results
    rel_pass, rel_total = print_reliability_section(all_reliability)
    total_passed += rel_pass
    total_checks += rel_total

    # ---- LangSmith ----
    if not args.no_langsmith:
        ls_pass, ls_total = run_langsmith_check(env)
        total_passed += ls_pass
        total_checks += ls_total

    errors = sum(1 for _, _, r in routing_results + reliability_results if "_error" in r)
    elapsed = time.perf_counter() - t_start
    _footer(total_passed, total_checks, errors, elapsed)
    return 0 if total_passed == total_checks else 1


def main() -> None:
    # Ensure UTF-8 output on Windows terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
