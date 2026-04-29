#!/usr/bin/env python
"""FraudLens SAR Generator end-to-end health probe.

Usage:
    uv run scripts/sar_agent_healthcheck.py [--port PORT] [--host HOST]
                                            [--no-langsmith] [--seed S]

Sections:
  INFRASTRUCTURE        — TCP reachability for PostgreSQL, Redis, Qdrant
  TRIAGE ROUTING        — validates triage_action=escalate for critical buckets
  MANDATORY TOOLS       — verifies 4 mandatory critical agent tools called
  SYNTHESIZER OUTPUT    — validates fraud_decision (outcome, confidence, citations)
  SAR GENERATION        — validates SARReport structure via Pydantic
  SAR REPORT DETAIL     — displays a sample SAR for manual inspection
  LANGSMITH             — verifies traces in the fraudlens project
  DB PERSISTENCE        — verifies SAR saved in decisions table; logs run to healthcheck_runs

Tests the full end-to-end escalation pipeline:
  Critical Agent (p >= 0.7) → Decision Synthesizer → SAR Generator → DB

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
from datetime import UTC, datetime
from pathlib import Path
from textwrap import wrap
from typing import Any

import httpx
from pydantic import ValidationError
from sqlalchemy import text

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraudlens.db.session import AsyncSessionFactory  # noqa: E402
from fraudlens.schemas.sar import SARReport  # noqa: E402

SCENARIOS_PATH = REPO_ROOT / "data" / "processed" / "test_scenarios.jsonl"
ENV_PATH = REPO_ROOT / ".env"

CRITICAL_BUCKETS = ["critical_low", "critical_high"]
BUCKET_TRIAGE = {"critical_low": "escalate", "critical_high": "escalate"}
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


def _magenta(s: str) -> str:
    return f"{_M}{s}{_X}"


# ---------------------------------------------------------------------------
# Scenario loading & HTTP helpers
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


def _check_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _post_one(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    timeout: float = 120.0,  # noqa: ASYNC109 — critical agent calls 4–6 tools
) -> dict[str, Any]:
    try:
        resp = await client.post(
            f"{base_url}/api/v1/transactions", params={"raw_mode": "true"}, json=payload, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# Section: INFRASTRUCTURE
# ---------------------------------------------------------------------------


def run_infra_checks(env: dict[str, str]) -> tuple[int, int]:
    print(f"\n{_B}[INFRASTRUCTURE]{_X}")
    passed, total = 0, 0
    checks = [
        ("PostgreSQL", env.get("POSTGRES_HOST", "localhost"), int(env.get("POSTGRES_PORT", "5432"))),
        ("Redis", env.get("REDIS_HOST", "localhost"), int(env.get("REDIS_PORT", "6379"))),
        ("Qdrant", env.get("QDRANT_HOST", "localhost"), int(env.get("QDRANT_PORT", "6333"))),
    ]
    for name, host, port in checks:
        total += 1
        label = f"{name:<16} {host}:{port}"
        if _check_tcp(host, port):
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
        print(f"  {_ok(f'API /health      {resp.status_code} OK  {model_tag}')}")
        return 1, 1
    except Exception as exc:
        print(f"  {_fail(f'API /health      {exc}')}")
        return 0, 1


# ---------------------------------------------------------------------------
# Section: TRIAGE ROUTING
# ---------------------------------------------------------------------------


def print_routing_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[int, int]:
    print(f"\n{_B}[TRIAGE ROUTING]{_X}")
    passed, total = 0, 0
    for bucket, scenario, api_resp in results:
        total += 1
        expected = BUCKET_TRIAGE[bucket]
        if "_error" in api_resp:
            err = api_resp["_error"][:60]
            print(f"  {_fail(f'{bucket:<18} API error: {err}')}")
            continue
        actual = api_resp.get("triage_action", "?")
        prob = api_resp.get("fraud_probability", 0.0)
        label = f"{bucket:<18}  score={prob:.3f}  action={actual}  is_fraud={scenario['is_fraud']}"
        if actual == expected:
            passed += 1
            print(f"  {_ok(label)}")
        else:
            print(f"  {_fail(label)}")
            print(f"       {_dim(f'expected action={expected}')}")
    return passed, total


# ---------------------------------------------------------------------------
# Section: MANDATORY TOOLS CHECK
# ---------------------------------------------------------------------------


def print_mandatory_tools_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[int, int]:
    print(f"\n{_B}[MANDATORY TOOLS CHECK]{_X}")
    print(f"  Required: {', '.join(sorted(MANDATORY_TOOLS))}")
    passed, total = 0, 0
    for bucket, _, api_resp in results:
        total += 1
        if "_error" in api_resp:
            print(f"  {_fail(f'{bucket:<18} API error — cannot check tools')}")
            continue
        inv = api_resp.get("investigation")
        tools_called = set((inv or {}).get("tools_called") or [])
        missing = MANDATORY_TOOLS - tools_called
        prob = api_resp.get("fraud_probability", 0.0)
        tools_n = len(tools_called)
        if not missing:
            passed += 1
            print(f"  {_ok(f'{bucket:<18} score={prob:.3f}  tools={tools_n}/8  all mandatory called')}")
        else:
            print(f"  {_fail(f'{bucket:<18} score={prob:.3f}  tools={tools_n}/8  missing: {sorted(missing)}')}")
    return passed, total


# ---------------------------------------------------------------------------
# Section: SYNTHESIZER OUTPUT
# ---------------------------------------------------------------------------


def print_synthesizer_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[int, int]:
    print(f"\n{_B}[SYNTHESIZER OUTPUT]{_X}")
    passed, total = 0, 0
    for bucket, _, api_resp in results:
        total += 1
        if "_error" in api_resp:
            print(f"  {_fail(f'{bucket:<18} API error — no fraud_decision')}")
            continue
        fd = api_resp.get("fraud_decision")
        prob = api_resp.get("fraud_probability", 0.0)
        if fd is None:
            print(f"  {_fail(f'{bucket:<18} score={prob:.3f}  fraud_decision=None (synthesizer did not run)')}")
            continue
        issues: list[str] = []
        outcome = fd.get("outcome", "?")
        if outcome != "escalate":
            issues.append(f"outcome={outcome!r} expected 'escalate'")
        conf = fd.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            issues.append(f"confidence={conf!r} not in [0, 1]")
        cit_count = len(fd.get("regulatory_citations") or [])
        hint = fd.get("decision_hint", "?")
        conf_val = float(conf) if isinstance(conf, (int, float)) else 0.0
        label = f"{bucket:<18}  score={prob:.3f}  outcome={outcome}  hint={hint}  conf={conf_val:.2f}  citations={cit_count}"
        if not issues:
            passed += 1
            print(f"  {_ok(label)}")
            if cit_count == 0:
                print(f"       {_warn('→ citations=0  (regulatory_rag called but RAG index may be empty)')}")
        else:
            print(f"  {_fail(label)}")
            for issue in issues:
                print(f"       {_dim('→')} {issue}")
    return passed, total


# ---------------------------------------------------------------------------
# Section: SAR GENERATION & VALIDATION
# ---------------------------------------------------------------------------


def _validate_sar_report(report: dict[str, Any] | None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not report:
        return False, ["sar_report is None or missing"]
    try:
        sar = SARReport.model_validate(report)
    except ValidationError as exc:
        return False, [f"Pydantic validation failed: {exc}"]
    if not sar.investigation_summary:
        issues.append("investigation_summary is empty")
    if not sar.recommended_action:
        issues.append("recommended_action is empty")
    if not sar.suspicious_indicators:
        issues.append("suspicious_indicators is empty")
    if not sar.regulatory_triggers:
        issues.append("regulatory_triggers is empty (fallback should always set a default)")
    return len(issues) == 0, issues


def print_sar_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[int, int]:
    print(f"\n{_B}[SAR GENERATION & VALIDATION]{_X}")
    passed, total = 0, 0
    for bucket, _, api_resp in results:
        total += 1
        if "_error" in api_resp:
            print(f"  {_fail(f'{bucket:<18} API error — cannot validate SAR')}")
            continue
        prob = api_resp.get("fraud_probability", 0.0)
        sar = api_resp.get("sar_report")
        is_ok, issues = _validate_sar_report(sar)
        indicator_n = len((sar or {}).get("suspicious_indicators") or [])
        trigger_n = len((sar or {}).get("regulatory_triggers") or [])
        model = (sar or {}).get("agent_model", "?")
        label = f"{bucket:<18}  score={prob:.3f}  indicators={indicator_n}  triggers={trigger_n}  model={model}"
        if is_ok:
            passed += 1
            print(f"  {_ok(f'{label}  SAR valid')}")
        else:
            print(f"  {_fail(f'{label}  SAR invalid')}")
            for issue in issues:
                print(f"       {_dim('→')} {issue}")
    return passed, total


# ---------------------------------------------------------------------------
# SAR report detail display (no pass/fail — informational only)
# ---------------------------------------------------------------------------


def print_sar_detail(api_resp: dict[str, Any]) -> None:
    report = api_resp.get("sar_report")
    if not report:
        print(f"  {_warn('No SAR report in this response.')}")
        return
    print(_dim("  " + "─" * 70))
    for key, value in report.items():
        key_label = f"{key.replace('_', ' ').title():<28}:"
        if isinstance(value, list):
            print(f"  {key_label}")
            for item in value:
                print(f"    • {item}")
        elif isinstance(value, dict):
            print(f"  {key_label}")
            for k, v in value.items():
                print(f"    • {k}: {v}")
        elif isinstance(value, str) and len(value) > 60:
            lines = wrap(value, width=65)
            print(f"  {key_label} {lines[0]}")
            for line in lines[1:]:
                print(f"                             {line}")
        else:
            print(f"  {key_label} {value}")
    print(_dim("  " + "─" * 70))


# ---------------------------------------------------------------------------
# Section: LANGSMITH
# ---------------------------------------------------------------------------


def run_langsmith_check(env: dict[str, str]) -> tuple[int, int]:
    print(f"\n{_B}[LANGSMITH]{_X}")
    tracing = env.get("LANGSMITH_TRACING", "false").lower() == "true"
    api_key = env.get("LANGSMITH_API_KEY", "")
    project = env.get("LANGSMITH_PROJECT", "fraudlens")
    if not tracing:
        print(f"  {_warn('LANGSMITH_TRACING=false — tracing disabled')}")
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
            print(f"  {_ok(f'{project}   {count} trace(s) found in last 10')}")
            return 1, 1
        print(f"  {_warn(f'{project}   0 traces found')}")
        return 0, 1
    except Exception as exc:
        print(f"  {_fail(f'LangSmith query failed: {exc}')}")
        return 0, 1


# ---------------------------------------------------------------------------
# Section: DB PERSISTENCE
# ---------------------------------------------------------------------------


async def run_db_persistence_section(
    results: list[tuple[str, dict[str, Any], dict[str, Any]]],
    started_at: datetime,
    check_summary: list[dict[str, Any]],
    grand_passed: int,
    grand_total: int,
) -> tuple[int, int]:
    """Verify SAR saved in decisions; persist the healthcheck run record.

    Returns the SAR-in-DB verification counts (pass, total).
    The run record insert is done at the end and covers all sections.
    """
    print(f"\n{_B}[DB PERSISTENCE]{_X}")
    passed, total = 0, 0
    sar_checks: list[dict[str, Any]] = []

    try:
        async with AsyncSessionFactory() as session:
            # Verify each escalated decision has sar_report persisted
            for bucket, _, api_resp in results:
                total += 1
                if "_error" in api_resp:
                    print(f"  {_fail(f'{bucket:<18} API error — skipping DB check')}")
                    sar_checks.append({"bucket": bucket, "status": "api_error"})
                    continue

                decision_id_raw = api_resp.get("decision_id")
                if not decision_id_raw:
                    print(f"  {_fail(f'{bucket:<18} decision_id missing from response')}")
                    sar_checks.append({"bucket": bucket, "status": "missing_decision_id"})
                    continue

                try:
                    decision_uuid = uuid.UUID(str(decision_id_raw))
                except ValueError:
                    bad_id = str(decision_id_raw)[:20]
                    print(f"  {_fail(f'{bucket:<18} invalid decision_id: {bad_id!r}')}")
                    sar_checks.append({"bucket": bucket, "status": "invalid_uuid"})
                    continue

                row_result = await session.execute(
                    text("SELECT (sar_report IS NOT NULL) AS has_sar, outcome FROM decisions WHERE id = :id"),
                    {"id": decision_uuid},
                )
                rec = row_result.fetchone()
                short = str(decision_uuid)[:8]
                if rec is None:
                    print(f"  {_fail(f'{bucket:<18} id={short}…  not found in decisions table')}")
                    sar_checks.append({"bucket": bucket, "decision_id": str(decision_uuid), "status": "not_found"})
                    continue

                has_sar, outcome = rec[0], rec[1]
                sar_checks.append({
                    "bucket": bucket,
                    "decision_id": str(decision_uuid),
                    "outcome": outcome,
                    "sar_saved": bool(has_sar),
                })
                if has_sar:
                    passed += 1
                    print(f"  {_ok(f'{bucket:<18} id={short}…  sar_report=saved  outcome={outcome}')}")
                else:
                    print(f"  {_fail(f'{bucket:<18} id={short}…  sar_report=NULL  outcome={outcome}')}")

            # Persist the healthcheck run record
            finished_at = datetime.now(UTC)
            elapsed_ms = (finished_at - started_at).total_seconds() * 1000
            final_summary = check_summary + [
                {"section": "db_sar_verification", "passed": passed, "total": total}
            ]
            final_passed = grand_passed + passed
            final_total = grand_total + total
            error_count = sum(1 for _, _, r in results if "_error" in r)
            transaction_ids = [
                api_resp.get("transaction_id")
                for _, _, api_resp in results
                if "_error" not in api_resp
            ]
            run_id = uuid.uuid4()

            await session.execute(
                text("""
                    INSERT INTO healthcheck_runs
                      (id, script_name, started_at, finished_at, elapsed_ms,
                       total_checks, passed_checks, failed_checks, error_count,
                       all_passed, check_details, transaction_ids)
                    VALUES
                      (:id, :script_name, :started_at, :finished_at, :elapsed_ms,
                       :total_checks, :passed_checks, :failed_checks, :error_count,
                       :all_passed, CAST(:check_details AS jsonb), CAST(:transaction_ids AS jsonb))
                """),
                {
                    "id": run_id,
                    "script_name": "sar_agent_healthcheck",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "total_checks": final_total,
                    "passed_checks": final_passed,
                    "failed_checks": final_total - final_passed,
                    "error_count": error_count,
                    "all_passed": final_passed == final_total,
                    "check_details": json.dumps(final_summary),
                    "transaction_ids": json.dumps([str(t) for t in transaction_ids if t]),
                },
            )
            await session.commit()
            short_run = str(run_id)[:8]
            print(f"  {_ok(f'Run record saved   id={short_run}…  passed={final_passed}/{final_total}  elapsed={elapsed_ms:.0f}ms')}")

    except Exception as exc:
        print(f"  {_fail(f'DB persistence failed: {exc}')}")

    return passed, total


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------


def _header() -> None:
    line = "═" * 62
    print(f"\n{_B}{line}")
    print("  FraudLens — SAR Generator End-to-End Health Check")
    print("  Pipeline: Critical Agent → Synthesizer → SAR Generator → DB")
    print(f"  Model: claude-haiku-4-5  |  Buckets: {', '.join(CRITICAL_BUCKETS)}")
    print(f"{line}{_X}")


def _footer(passed: int, total: int, errors: int, elapsed: float) -> None:
    line = "═" * 62
    failed = total - passed
    color = _G if passed == total else _R
    print(f"\n{_B}{line}")
    print(f"  {color}Results: {passed}/{total} passed  •  {failed} failed  •  {errors} errors{_X}")
    print(f"  {_dim(f'Elapsed: {elapsed:.1f}s')}")
    print(f"{_B}{line}{_X}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FraudLens SAR Generator end-to-end health check")
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
    started_at = datetime.now(UTC)
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
        print(f"\n{_R}ERROR: {SCENARIOS_PATH} not found.")
        print(f"       Run: uv run scripts/create_test_scenarios.py{_X}")
        return 1

    scenarios = load_scenarios()
    rng = random.Random(args.seed)

    for b in CRITICAL_BUCKETS:
        if not scenarios.get(b):
            print(f"\n{_R}ERROR: No scenarios found for bucket '{b}'.{_X}")
            print("       Run: uv run scripts/create_test_scenarios.py")
            return 1

    infra_pass, infra_total = run_infra_checks(env)
    api_pass, api_total = run_health_check(base_url)
    total_passed = infra_pass + api_pass
    total_checks = infra_total + api_total

    test_pairs = [(b, row := rng.choice(scenarios[b]), make_payload(row)) for b in CRITICAL_BUCKETS]
    print(f"\n  {_dim(f'Dispatching {len(test_pairs)} critical-tier requests to {base_url}...')}")
    print(f"  {_dim('Critical agent calls 4–6 tools per transaction. Timeout: 120s each.')}")

    test_resps: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for bucket, _, payload in test_pairs:
            print(f"\n  {_dim(f'→ Sending {bucket}...')}", flush=True)
            resp = await _post_one(client, base_url, payload)
            test_resps.append(resp)
            if "_error" in resp:
                err = resp["_error"][:50]
                print(f"    {_fail(f'{bucket:<18} error: {err}')}")
            else:
                prob = resp.get("fraud_probability", 0.0)
                action = resp.get("triage_action", "?")
                fd = resp.get("fraud_decision") or {}
                outcome = fd.get("outcome", "none")
                has_sar = resp.get("sar_report") is not None
                print(f"    {_ok(f'{bucket:<18} score={prob:.3f}  action={action}  outcome={outcome}  sar={has_sar}')}")

    results: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (b, s, r) for (b, s, _), r in zip(test_pairs, test_resps, strict=True)
    ]
    check_summary: list[dict[str, Any]] = []

    r_pass, r_total = print_routing_section(results)
    total_passed += r_pass
    total_checks += r_total
    check_summary.append({"section": "triage_routing", "passed": r_pass, "total": r_total})

    mt_pass, mt_total = print_mandatory_tools_section(results)
    total_passed += mt_pass
    total_checks += mt_total
    check_summary.append({"section": "mandatory_tools", "passed": mt_pass, "total": mt_total})

    sy_pass, sy_total = print_synthesizer_section(results)
    total_passed += sy_pass
    total_checks += sy_total
    check_summary.append({"section": "synthesizer_output", "passed": sy_pass, "total": sy_total})

    sar_pass, sar_total = print_sar_section(results)
    total_passed += sar_pass
    total_checks += sar_total
    check_summary.append({"section": "sar_generation", "passed": sar_pass, "total": sar_total})

    # SAR report detail — display only, first successful response
    for _, _, api_resp in results:
        if "_error" not in api_resp and api_resp.get("sar_report"):
            print(f"\n{_B}[SAR REPORT DETAIL]{_X}")
            print_sar_detail(api_resp)
            break

    if not args.no_langsmith:
        ls_pass, ls_total = run_langsmith_check(env)
        total_passed += ls_pass
        total_checks += ls_total
        check_summary.append({"section": "langsmith", "passed": ls_pass, "total": ls_total})

    db_pass, db_total = await run_db_persistence_section(
        results, started_at, check_summary, total_passed, total_checks
    )
    total_passed += db_pass
    total_checks += db_total

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
