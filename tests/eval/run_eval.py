"""Evaluation runner for the FraudLens gold standard test set.

Reads gold_set.jsonl, POSTs each case to the API, compares actual vs expected
outcomes, and produces metrics + a timestamped JSON result file.

Exit code 0 = all thresholds passed. Exit code 1 = one or more failed (CI use).

Usage:
    uv run tests/eval/run_eval.py [--base-url http://localhost:8001] [--concurrency 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GOLD_SET = Path(__file__).parent / "gold_set.jsonl"
_RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Thresholds (CI pass/fail gates)
# ---------------------------------------------------------------------------

_THRESHOLD_OUTCOME_ACCURACY = 0.80
_THRESHOLD_SAR_PRECISION = 0.80
_THRESHOLD_CITATION_COVERAGE = 0.60

# Haiku pricing (per million tokens) — kept in sync with cost_tracker.py
_HAIKU_INPUT_PRICE_PER_M = 0.80
_HAIKU_OUTPUT_PRICE_PER_M = 4.00


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gold_set() -> list[dict[str, Any]]:
    cases = []
    with open(_GOLD_SET) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _actual_outcome(response: dict[str, Any]) -> str:
    """Extract the final outcome from an API response dict."""
    if response.get("fraud_decision"):
        return response["fraud_decision"].get("outcome", "unknown")
    return response.get("triage_action", "unknown")


def _has_sar(response: dict[str, Any]) -> bool:
    return bool(response.get("sar_report"))


def _has_citations(response: dict[str, Any]) -> bool:
    fd = response.get("fraud_decision") or {}
    return bool(fd.get("regulatory_citations"))


def _hallucination_count(response: dict[str, Any], case: dict[str, Any]) -> tuple[int, int]:
    """Return (hallucinated_claims, total_claims) in suspicious_indicators.

    A claim is hallucinated if none of its keywords appear in:
      - the raw transaction JSON
      - the case description
      - the agent's own red_flags and evidence (derived findings are not hallucinations)
    """
    sar = response.get("sar_report")
    if not sar:
        return 0, 0

    context_parts = [
        json.dumps(case["transaction"]).lower(),
        case.get("description", "").lower(),
    ]
    investigation = response.get("investigation") or {}
    for item in (investigation.get("red_flags") or []) + (investigation.get("evidence") or []):
        context_parts.append(str(item).lower())

    context_tokens = set(re.findall(r"[a-z]{4,}", " ".join(context_parts)))
    indicators: list[str] = sar.get("suspicious_indicators", [])
    hallucinated = 0
    for indicator in indicators:
        indicator_words = set(re.findall(r"[a-z]{4,}", indicator.lower()))
        if indicator_words and not indicator_words.intersection(context_tokens):
            hallucinated += 1

    return hallucinated, len(indicators)


def _estimate_cost(response: dict[str, Any]) -> float:
    """Estimate USD cost from token_usage if present in response."""
    usage = response.get("token_usage") or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return (input_tokens * _HAIKU_INPUT_PRICE_PER_M + output_tokens * _HAIKU_OUTPUT_PRICE_PER_M) / 1_000_000


# ---------------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------------


async def _run_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    base_url: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    case_id = case["case_id"]
    t0 = time.perf_counter()

    url = f"{base_url}/api/v1/transactions"
    if case.get("raw_mode"):
        url += "?raw_mode=true"

    async with semaphore:
        try:
            resp = await client.post(
                url,
                json=case["transaction"],
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000

            actual_triage = data.get("triage_action", "unknown")
            actual_outcome = _actual_outcome(data)
            sar_produced = _has_sar(data)
            has_citations = _has_citations(data)
            hall_count, total_indicators = _hallucination_count(data, case)
            cost_usd = _estimate_cost(data)

            triage_match = actual_triage == case["expected_triage"]
            outcome_match = actual_outcome == case["expected_outcome"]
            sar_match = sar_produced == case["expected_sar"]

            inv = data.get("investigation") or {}
            fd = data.get("fraud_decision") or {}
            agent_confidence = inv.get("confidence")
            agent_hint = inv.get("decision_hint")
            agent_tools = inv.get("tools_called") or []
            agent_flags = inv.get("red_flags") or []
            citations_count = len(fd.get("regulatory_citations") or [])

            result = {
                "case_id": case_id,
                "description": case["description"],
                "tier": case["tier"],
                "tags": case["tags"],
                "expected_triage": case["expected_triage"],
                "actual_triage": actual_triage,
                "triage_match": triage_match,
                "expected_outcome": case["expected_outcome"],
                "actual_outcome": actual_outcome,
                "outcome_match": outcome_match,
                "expected_sar": case["expected_sar"],
                "sar_produced": sar_produced,
                "sar_match": sar_match,
                "has_citations": has_citations,
                "citations_count": citations_count,
                "hallucinated_indicators": hall_count,
                "total_indicators": total_indicators,
                "agent_confidence": agent_confidence,
                "agent_hint": agent_hint,
                "agent_tools_called": agent_tools,
                "agent_tools_count": len(agent_tools),
                "agent_red_flags": agent_flags,
                "latency_ms": round(latency_ms, 1),
                "estimated_cost_usd": round(cost_usd, 6),
                "status": "ok",
                "error": None,
            }

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            result = {
                "case_id": case_id,
                "description": case["description"],
                "tier": case["tier"],
                "tags": case["tags"],
                "expected_triage": case["expected_triage"],
                "actual_triage": "error",
                "triage_match": False,
                "expected_outcome": case["expected_outcome"],
                "actual_outcome": "error",
                "outcome_match": False,
                "expected_sar": case["expected_sar"],
                "sar_produced": False,
                "sar_match": False,
                "has_citations": False,
                "citations_count": 0,
                "hallucinated_indicators": 0,
                "total_indicators": 0,
                "agent_confidence": None,
                "agent_hint": None,
                "agent_tools_called": [],
                "agent_tools_count": 0,
                "agent_red_flags": [],
                "latency_ms": round(latency_ms, 1),
                "estimated_cost_usd": 0.0,
                "status": "error",
                "error": str(exc),
            }

    status_char = "." if result["outcome_match"] else ("E" if result["status"] == "error" else "F")
    err_suffix = f"  !! {result['error'][:60]}" if result["status"] == "error" else ""
    print(f"  [{status_char}] {case_id}: {case['description'][:55]:<55} "
          f"triage={result['actual_triage']:<11} outcome={result['actual_outcome']:<13} "
          f"{result['latency_ms']:.0f}ms{err_suffix}", flush=True)

    if result["actual_triage"] != "approve" and result["status"] == "ok":
        conf = result["agent_confidence"]
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        hint = result["agent_hint"] or "n/a"
        tools_n = result["agent_tools_count"]
        flags = result["agent_red_flags"][:3]
        flags_str = ", ".join(f[:22] for f in flags) if flags else "none"
        cit = result["citations_count"]
        print(f"        hint={hint:<20} conf={conf_str}  tools={tools_n}  "
              f"flags=[{flags_str}]  cit={cit}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def _compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    ok_results = [r for r in results if r["status"] == "ok"]

    outcome_matches = sum(1 for r in results if r["outcome_match"])
    outcome_accuracy = outcome_matches / total if total else 0.0

    # SAR precision: escalate cases where SAR was expected
    sar_expected = [r for r in results if r["expected_sar"]]
    sar_correct = sum(1 for r in sar_expected if r["sar_produced"])
    sar_precision = sar_correct / len(sar_expected) if sar_expected else 1.0

    # Citation coverage: escalate cases (expected_outcome=escalate)
    escalate_cases = [r for r in results if r["expected_outcome"] == "escalate"]
    citation_cases = sum(1 for r in escalate_cases if r["has_citations"])
    citation_coverage = citation_cases / len(escalate_cases) if escalate_cases else 1.0

    # Hallucination rate across all results with SAR
    total_indicators = sum(r["total_indicators"] for r in results)
    total_hallucinated = sum(r["hallucinated_indicators"] for r in results)
    hallucination_rate = total_hallucinated / total_indicators if total_indicators else 0.0

    # Latency
    latencies = [r["latency_ms"] for r in ok_results if r["latency_ms"] > 0]
    latencies_by_tier: dict[str, list[float]] = {}
    for r in ok_results:
        latencies_by_tier.setdefault(r["tier"], []).append(r["latency_ms"])

    def _pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        sorted_v = sorted(vals)
        idx = int(len(sorted_v) * p)
        return sorted_v[min(idx, len(sorted_v) - 1)]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _pct(latencies, 0.95)

    tier_latency = {
        tier: {
            "avg_ms": round(sum(v) / len(v), 1) if v else 0,
            "p95_ms": round(_pct(v, 0.95), 1),
            "count": len(v),
        }
        for tier, v in latencies_by_tier.items()
    }

    total_cost = sum(r["estimated_cost_usd"] for r in results)

    # Agent confidence stats — only for cases where an agent ran
    agent_results = [r for r in ok_results if r.get("agent_confidence") is not None]
    hint_counts: dict[str, int] = {}
    for r in agent_results:
        hint = r.get("agent_hint") or "unknown"
        hint_counts[hint] = hint_counts.get(hint, 0) + 1
    conf_values = [r["agent_confidence"] for r in agent_results]
    avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0.0

    thresholds_passed = (
        outcome_accuracy >= _THRESHOLD_OUTCOME_ACCURACY
        and sar_precision >= _THRESHOLD_SAR_PRECISION
        and citation_coverage >= _THRESHOLD_CITATION_COVERAGE
    )

    return {
        "total_cases": total,
        "ok_cases": len(ok_results),
        "error_cases": total - len(ok_results),
        "outcome_accuracy": round(outcome_accuracy, 4),
        "sar_precision": round(sar_precision, 4),
        "citation_coverage": round(citation_coverage, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "latency_by_tier": tier_latency,
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_per_case_usd": round(total_cost / total, 6) if total else 0.0,
        "thresholds": {
            "outcome_accuracy": {"value": round(outcome_accuracy, 4), "threshold": _THRESHOLD_OUTCOME_ACCURACY, "passed": outcome_accuracy >= _THRESHOLD_OUTCOME_ACCURACY},
            "sar_precision": {"value": round(sar_precision, 4), "threshold": _THRESHOLD_SAR_PRECISION, "passed": sar_precision >= _THRESHOLD_SAR_PRECISION},
            "citation_coverage": {"value": round(citation_coverage, 4), "threshold": _THRESHOLD_CITATION_COVERAGE, "passed": citation_coverage >= _THRESHOLD_CITATION_COVERAGE},
        },
        "all_thresholds_passed": thresholds_passed,
        "agent_avg_confidence": round(avg_confidence, 4),
        "agent_hint_counts": hint_counts,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)

    th = metrics["thresholds"]
    rows = [
        ("outcome_accuracy", th["outcome_accuracy"]),
        ("sar_precision", th["sar_precision"]),
        ("citation_coverage", th["citation_coverage"]),
    ]
    for name, t in rows:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  {name:<25} {t['value']:.4f}  (threshold {t['threshold']:.2f})  [{status}]")

    print(f"\n  hallucination_rate        {metrics['hallucination_rate']:.4f}")
    print(f"  avg_latency_ms            {metrics['avg_latency_ms']:.1f}")
    print(f"  p95_latency_ms            {metrics['p95_latency_ms']:.1f}")
    print(f"  total_cost_usd            ${metrics['total_cost_usd']:.4f}")
    print(f"  avg_cost_per_case_usd     ${metrics['avg_cost_per_case_usd']:.6f}")

    hints = metrics.get("agent_hint_counts", {})
    hints_str = "  ".join(f"{k}={v}" for k, v in sorted(hints.items()))
    print(f"  agent_avg_confidence      {metrics['agent_avg_confidence']:.4f}")
    print(f"  agent_hint_dist           {hints_str or 'n/a'}")

    print("\n  Latency by tier:")
    for tier, v in metrics["latency_by_tier"].items():
        print(f"    {tier:<8}  avg={v['avg_ms']:.1f}ms  p95={v['p95_ms']:.1f}ms  n={v['count']}")

    verdict = "ALL THRESHOLDS PASSED" if metrics["all_thresholds_passed"] else "THRESHOLD(S) FAILED"
    print(f"\n  {'=' * 30}")
    print(f"  {verdict}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _check_api_health(base_url: str) -> bool:
    """Return True if the API server is reachable and healthy."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            return resp.status_code == 200
    except Exception:
        return False


async def main(base_url: str, concurrency: int) -> bool:
    print(f"Checking API at {base_url}/health ...")
    if not await _check_api_health(base_url):
        print(
            f"\nERROR: Cannot reach API at {base_url}\n"
            "Start the stack first:\n"
            "  docker compose up -d\n"
            "  uv run alembic upgrade head\n"
            "  uv run uvicorn fraudlens.api.main:app --port 8001\n"
        )
        return False

    print("API is healthy.\n")
    cases = _load_gold_set()
    print(f"Loaded {len(cases)} cases from {_GOLD_SET}")
    print(f"Target: {base_url}  |  concurrency: {concurrency}\n")

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        tasks = [_run_case(client, case, base_url, semaphore) for case in cases]
        results = await asyncio.gather(*tasks)

    results = list(results)
    metrics = _compute_metrics(results)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    out_path = _RESULTS_DIR / f"{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "cases": results}, f, indent=2, default=str)

    print(f"\nResults saved → {out_path}")
    _print_summary(metrics)

    return metrics["all_thresholds_passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudLens eval runner")
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent requests")
    args = parser.parse_args()

    passed = asyncio.run(main(args.base_url, args.concurrency))
    sys.exit(0 if passed else 1)
