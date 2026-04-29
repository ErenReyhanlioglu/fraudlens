#!/usr/bin/env python
"""FraudLens Critical Agent health probe.

Usage:
    uv run scripts/critical_agent_healthcheck.py [--port PORT] [--host HOST]
                                                  [--no-langsmith] [--seed S]

Sections:
  INFRASTRUCTURE          — TCP reachability for PostgreSQL, Redis, Qdrant
  TRIAGE ROUTING          — 1 POST per critical bucket, validates triage_action=escalate
  CRITICAL AGENT DETAIL   — full 8-tool trace with network/RAG/sanctions display
  MANDATORY TOOLS CHECK   — verifies customer_history, adverse_media, network_analysis called
  STRUCTURED OUTPUT       — validates InvestigationResult fields for critical tier
  LANGSMITH               — verifies traces appeared in the fraudlens project

Tests only the Critical Agent (p >= 0.7, 8 tools, claude-haiku-4-5).
For the Investigation Agent use: uv run scripts/investigator_agent_healthcheck.py

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

CRITICAL_BUCKETS = ["critical_low", "critical_high"]
BUCKET_TRIAGE = {
    "critical_low": "escalate",
    "critical_high": "escalate",
}

# Tools that MUST be called by the Critical Agent (compliance requirement).
MANDATORY_TOOLS = {"get_customer_history", "adverse_media_search", "deep_network_analysis", "regulatory_policy_rag"}

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
_M = "\033[95m"
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


def _magenta(s: str) -> str:
    return f"{_M}{s}{_X}"


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------


def load_scenarios() -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {b: [] for b in CRITICAL_BUCKETS}
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
        ("Qdrant", env.get("QDRANT_HOST", "localhost"), int(env.get("QDRANT_PORT", "6333"))),
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
    timeout: float = 120.0,  # noqa: ASYNC109 — critical agent calls 4–6 tools
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
# Tool result formatting — all 8 tools
# ---------------------------------------------------------------------------


def _fmt_tool_result(tool: str, result_str: str) -> list[str]:
    """Return formatted display lines for a tool's raw JSON result."""
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, ValueError, TypeError):
        return [_dim(str(result_str)[:120])]

    lines: list[str] = []

    if tool == "explain_ml_score":
        for feat in (data.get("top_features") or [])[:3]:
            val = feat.get("shap_contribution", 0)
            sign = "+" if val > 0 else ""
            direction = "→ fraud" if val > 0 else "→ legit"
            lines.append(f"{feat['feature']} = {sign}{val:.4f}  {_dim(direction)}")

    elif tool == "get_customer_history":
        lines.append(f"tx_count={data.get('transaction_count')}  avg_amount=${data.get('average_transaction_amount_usd')}  prior_flags={data.get('prior_suspicious_flags')}")
        lines.append(f"countries={data.get('countries_transacted')}  account_age={data.get('account_age_days')}d")

    elif tool == "check_merchant_reputation":
        lines.append(f"risk_score={data.get('risk_score')}  category={data.get('industry_category')}  chargebacks={data.get('chargeback_rate_pct')}%")
        flags = data.get("flags") or []
        if flags:
            lines.append(f"flags={flags}")

    elif tool == "get_geolocation_context":
        lines.append(f"ip_country={data.get('ip_country')}  vpn={data.get('vpn_detected')}  impossible_travel={data.get('impossible_travel')}")
        signals = data.get("risk_signals") or []
        if signals:
            lines.append(f"signals={signals}")

    elif tool == "find_similar_patterns":
        patterns = data.get("patterns") or []
        lines.append(f"match_count={data.get('match_count')}  risk_level={data.get('risk_level')}  patterns={patterns}")

    elif tool == "deep_network_analysis":
        risk_level = data.get("risk_level", "?")
        risk_color = _R if risk_level == "high" else (_Y if risk_level == "medium" else _G)
        lines.append(f"nodes={data.get('node_count')}  edges={data.get('edge_count')}  density={data.get('graph_density')}  risk={risk_color}{risk_level}{_X}")
        signals = data.get("risk_signals") or []
        if signals:
            lines.append(f"signals={signals}")
        lines.append(_dim(data.get("risk_assessment", "")))

    elif tool == "regulatory_policy_rag":
        excerpts = data.get("excerpts") or []
        if excerpts:
            for ex in excerpts[:2]:
                citation = ex.get("citation", "?")
                score = ex.get("relevance_score", 0.0)
                text_preview = ex.get("text", "")[:90].replace("\n", " ")
                lines.append(f"{_magenta(citation)}  score={score}")
                lines.append(f"  {_dim(text_preview)}...")
        else:
            lines.append(_dim(data.get("message", "no results")))

    elif tool == "adverse_media_search":
        risk_level = data.get("overall_risk_level", "?")
        risk_color = _R if risk_level in ("critical", "high") else _G
        sanctions = data.get("sanctions_match", False)
        pep = data.get("pep_flag", False)
        adverse_n = data.get("adverse_media_hit_count", 0)
        lines.append(f"risk={risk_color}{risk_level}{_X}  sanctions={_R + 'YES' + _X if sanctions else 'no'}  pep={_Y + 'YES' + _X if pep else 'no'}  adverse_hits={adverse_n}")
        hits = data.get("sanctions_list_hits") or []
        if hits:
            lines.append(f"{_R}SANCTIONS HIT: {hits}{_X}")
        cats = data.get("adverse_media_categories") or []
        if cats:
            lines.append(f"media_categories={cats}")

    return lines or [_dim(str(result_str)[:120])]


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
            print(f"  {_fail(f'{bucket:<18} API error: {err}')}")
            continue

        actual = api_resp.get("triage_action", "?")
        prob = api_resp.get("fraud_probability", -1.0)
        inv = api_resp.get("investigation")
        hint_tag = f"  hint={inv['decision_hint']}" if inv and inv.get("decision_hint") else ""
        is_ok = actual == expected

        parts = [f"score={prob:.3f}", f"action={actual}", f"is_fraud={scenario['is_fraud']}"]
        line = f"{bucket:<18}  {'  '.join(parts)}{hint_tag}"
        if is_ok:
            passed += 1
            print(f"  {_ok(line)}")
        else:
            print(f"  {_fail(line)}")
            print(f"       {_dim(f'expected action={expected}  got={actual}')}")
    return passed, total


# ---------------------------------------------------------------------------
# Critical agent detail section
# ---------------------------------------------------------------------------


def _print_single_critical(bucket: str, scenario: dict[str, Any], api_resp: dict[str, Any]) -> None:
    inv = api_resp.get("investigation")
    prob = api_resp.get("fraud_probability", 0.0)
    is_fraud = scenario["is_fraud"]

    print(f"\n  {_B}[CRITICAL AGENT — {bucket}]{_X}  {_dim(f'score={prob:.4f}  is_fraud={is_fraud}')}")

    if not inv:
        print(f"  {_warn('investigation=None (critical agent did not run or failed)')}")
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
    print(f"  Tool count     : {len(tools)}/8  {_dim('(critical tier should use 4–6)')}")

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
        for e in evidence[:5]:
            print(f"    • {e}")

    if red_flags:
        print("  Red flags      :")
        for rf in red_flags[:5]:
            print(f"    {_R}•{_X} {rf}")

    if summary:
        wrapped = wrap(summary, width=72)
        print(f"  Summary        : {wrapped[0]}")
        for extra in wrapped[1:]:
            print(f"                   {extra}")


# ---------------------------------------------------------------------------
# Mandatory tools check
# ---------------------------------------------------------------------------


def print_mandatory_tools_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[int, int]:
    print(f"\n{_B}[MANDATORY TOOLS CHECK]{_X}")
    print(f"  Required for every critical transaction: {', '.join(sorted(MANDATORY_TOOLS))}")
    passed = 0
    total = 0
    for bucket, _scenario, api_resp in results:
        total += 1
        if "_error" in api_resp:
            print(f"  {_fail(f'{bucket:<18} API error — cannot check tools')}")
            continue

        inv = api_resp.get("investigation")
        tools_called = set((inv or {}).get("tools_called") or [])
        missing = MANDATORY_TOOLS - tools_called
        prob = api_resp.get("fraud_probability", 0.0)

        if not missing:
            passed += 1
            print(f"  {_ok(f'{bucket:<18} score={prob:.3f}  all mandatory tools called')}")
        else:
            print(f"  {_fail(f'{bucket:<18} score={prob:.3f}  missing: {sorted(missing)}')}")
    return passed, total


# ---------------------------------------------------------------------------
# Structured output validation
# ---------------------------------------------------------------------------

_VALID_HINTS = {"likely_legitimate", "suspicious", "inconclusive"}


def _validate_critical_result(inv: dict[str, Any] | None, prob: float) -> tuple[bool, list[str]]:
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
    tools = inv.get("tools_called") or []
    if len(tools) < 3:
        issues.append(f"only {len(tools)} tools called — critical tier expects 4+")
    if inv.get("decision_hint") == "likely_legitimate" and prob >= 0.80:
        issues.append(f"likely_legitimate verdict for score={prob:.3f} — suspicious expected")
    if inv.get("decision_hint") == "inconclusive" and not tools:
        issues.append("fallback SUSPICIOUS expected on error, got inconclusive with no tools")
    return len(issues) == 0, issues


def print_structured_output_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[int, int]:
    print(f"\n{_B}[STRUCTURED OUTPUT VALIDATION]{_X}")
    passed = 0
    total = 0
    for idx, (bucket, _scenario, api_resp) in enumerate(results, 1):
        total += 1
        if "_error" in api_resp:
            err = api_resp["_error"][:60]
            print(f"  {_fail(f'#{idx} {bucket:<18}  API error: {err}')}")
            continue

        inv = api_resp.get("investigation")
        prob = api_resp.get("fraud_probability", 0.0)
        ok, issues = _validate_critical_result(inv, prob)
        hint = (inv or {}).get("decision_hint", "?")
        conf = (inv or {}).get("confidence", 0.0)
        tools_n = len((inv or {}).get("tools_called") or [])

        label = f"#{idx} {bucket:<18}  hint={hint:<20} conf={conf:.2f}  tools={tools_n}  score={prob:.3f}"
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
    line = "═" * 58
    print(f"\n{_B}{line}")
    print("  FraudLens — Critical Agent Health Check")
    print("  Model: claude-haiku-4-5  |  Tools: 8  |  Tier: p >= 0.7")
    print("  Extra tools: network_analysis, regulatory_rag, adverse_media")
    print(f"{line}{_X}")


def _footer(passed: int, total: int, errors: int, elapsed: float) -> None:
    line = "═" * 58
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
    p = argparse.ArgumentParser(description="FraudLens Critical Agent smoke test")
    p.add_argument("--port", type=int, default=8001, help="API port (default 8001)")
    p.add_argument("--host", default="127.0.0.1", help="API host (default 127.0.0.1)")
    p.add_argument("--no-langsmith", action="store_true", help="Skip LangSmith check")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for scenario selection")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _async_main() -> int:
    args = _parse_args()
    base_url = f"http://{args.host}:{args.port}"
    t_start = time.perf_counter()

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

    # Check that critical buckets have scenarios
    for b in CRITICAL_BUCKETS:
        if not scenarios.get(b):
            print(f"\n{_R}ERROR: No scenarios found for bucket '{b}'.{_X}")
            print("       Run: uv run scripts/create_test_scenarios.py")
            return 1

    infra_pass, infra_total = run_infra_checks(env)
    api_pass, api_total = run_health_check(base_url)
    total_passed = infra_pass + api_pass
    total_checks = infra_total + api_total

    # Pick 1 scenario per critical bucket
    test_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = [(bucket, row := rng.choice(scenarios[bucket]), make_payload(row)) for bucket in CRITICAL_BUCKETS]

    total_requests = len(test_pairs)
    print(f"\n  {_dim(f'Dispatching {total_requests} critical-tier requests to {base_url}...')}")
    print(f"  {_dim('Critical agent calls 4–6 tools per transaction. Timeout: 120s each.')}")

    test_resps = []
    async with httpx.AsyncClient() as client:
        for bucket, _scenario, p in test_pairs:
            print(f"\n  {_dim(f'→ Sending {bucket}...')}", flush=True)
            resp = await _post_one(client, base_url, p)
            test_resps.append(resp)
            if "_error" in resp:
                err = resp["_error"][:50]
                print(f"    {_fail(f'{bucket:<18} API error: {err}')}")
            else:
                action = resp.get("triage_action", "?")
                prob = resp.get("fraud_probability", 0.0)
                hint = (resp.get("investigation") or {}).get("decision_hint", "none")
                tools_n = len((resp.get("investigation") or {}).get("tools_called") or [])
                print(f"    {_ok(f'{bucket:<18} score={prob:.3f} action={action} hint={hint} tools_used={tools_n}')}")

    results = [(b, s, r) for (b, s, _), r in zip(test_pairs, test_resps, strict=True)]

    # Triage routing
    r_pass, r_total = print_routing_section(results)
    total_passed += r_pass
    total_checks += r_total

    # Detailed agent output
    print(f"\n{_B}[CRITICAL AGENT DETAIL]{_X}")
    for bucket, scenario, api_resp in results:
        if "_error" not in api_resp:
            _print_single_critical(bucket, scenario, api_resp)

    # Mandatory tools check
    mt_pass, mt_total = print_mandatory_tools_section(results)
    total_passed += mt_pass
    total_checks += mt_total

    # Structured output validation
    so_pass, so_total = print_structured_output_section(results)
    total_passed += so_pass
    total_checks += so_total

    if not args.no_langsmith:
        ls_pass, ls_total = run_langsmith_check(env)
        total_passed += ls_pass
        total_checks += ls_total

    errors = sum(1 for _, _, r in results if "_error" in r)
    elapsed = time.perf_counter() - t_start
    _footer(total_passed, total_checks, errors, elapsed)
    return 0 if total_passed == total_checks else 1


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
