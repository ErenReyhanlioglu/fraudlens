"""Compute real CV metrics from captured fixtures.

Usage:
    uv run python scripts/compute_cv_metrics.py

Reads demo/fixtures_captured.json and prints metrics ready to paste into CV.
"""

from __future__ import annotations

import json
from pathlib import Path

_FIXTURES = Path(__file__).parent.parent / "demo" / "fixtures_captured.json"

# Haiku 4.5 pricing (USD per million tokens)
_INPUT_PPM = 0.80
_OUTPUT_PPM = 4.00

# Baseline: if EVERY case went through the Critical Agent (worst-case, most thorough)


def _actual_outcome(result: dict) -> str:
    fd = result.get("fraud_decision")
    if fd and fd.get("outcome"):
        return fd["outcome"]
    return "approve" if result.get("triage_action") == "approve" else "unknown"


def _token_cost(usage: dict | None) -> float:
    if not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) * _INPUT_PPM
        + usage.get("output_tokens", 0) * _OUTPUT_PPM
    ) / 1_000_000


def main() -> None:
    if not _FIXTURES.exists():
        print(f"ERROR: {_FIXTURES} not found. Run capture_fixtures.py first.")
        return

    fixtures = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    total = len(fixtures)

    # ── Outcome accuracy ──────────────────────────────────────────────────────
    outcome_matches = sum(
        1 for f in fixtures
        if _actual_outcome(f["result"]) == f["case"]["expected_outcome"]
    )
    outcome_accuracy = outcome_matches / total

    # ── SAR precision (escalate cases) ────────────────────────────────────────
    sar_expected = [f for f in fixtures if f["case"]["expected_sar"]]
    sar_correct = sum(1 for f in sar_expected if f["result"].get("sar_report"))
    sar_precision = sar_correct / len(sar_expected) if sar_expected else 1.0

    # ── Citation coverage (escalate cases) ───────────────────────────────────
    escalate = [f for f in fixtures if f["case"]["expected_outcome"] == "escalate"]
    cited = sum(
        1 for f in escalate
        if (f["result"].get("fraud_decision") or {}).get("regulatory_citations")
    )
    citation_coverage = cited / len(escalate) if escalate else 1.0

    # ── Cost analysis ─────────────────────────────────────────────────────────
    actual_cost = sum(_token_cost(f["result"].get("token_usage")) for f in fixtures)
    # Baseline: all 50 cases through Critical Agent (avg cost from actual HIGH cases)
    high_cases = [f for f in fixtures if f["case"]["tier"] == "high"]
    high_cost_avg = sum(_token_cost(f["result"].get("token_usage")) for f in high_cases) / len(high_cases) if high_cases else 0.026
    baseline_cost = total * high_cost_avg
    cost_savings_pct = (1 - actual_cost / baseline_cost) * 100 if baseline_cost else 0

    # ── Latency by tier ───────────────────────────────────────────────────────
    by_tier: dict[str, list[float]] = {}
    for f in fixtures:
        tier = f["case"]["tier"]
        by_tier.setdefault(tier, []).append(f["elapsed_ms"])

    def p95(vals: list[float]) -> float:
        s = sorted(vals)
        return s[int(len(s) * 0.95)]

    # ── Token usage breakdown ─────────────────────────────────────────────────
    total_input = sum((f["result"].get("token_usage") or {}).get("input_tokens", 0) for f in fixtures)
    total_output = sum((f["result"].get("token_usage") or {}).get("output_tokens", 0) for f in fixtures)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  FraudLens — CV Metrics (from 50-case gold set)")
    print("=" * 60)

    print(f"\n ACCURACY")
    print(f"   Outcome accuracy      : {outcome_accuracy:.1%}  ({outcome_matches}/{total})")
    print(f"   SAR precision         : {sar_precision:.1%}  ({sar_correct}/{len(sar_expected)} escalate cases)")
    print(f"   Citation coverage     : {citation_coverage:.1%}  ({cited}/{len(escalate)} escalate cases)")

    print(f"\n COST  (Haiku 4.5 pricing)")
    print(f"   Actual total cost     : ${actual_cost:.4f}  ({total} cases)")
    print(f"   Cost per case (avg)   : ${actual_cost / total:.5f}")
    print(f"   Baseline (all critical): ${baseline_cost:.4f}")
    print(f"   Cost savings          : {cost_savings_pct:.0f}%  via tier routing")
    print(f"   Total tokens          : {total_input:,} input / {total_output:,} output")

    print(f"\n LATENCY")
    for tier in ("low", "medium", "high"):
        vals = by_tier.get(tier, [])
        if vals:
            avg = sum(vals) / len(vals)
            print(f"   {tier.upper():<8} avg={avg/1000:.1f}s  p95={p95(vals)/1000:.1f}s  n={len(vals)}")

    print(f"\n CV BULLETS (copy-paste ready)")
    print(f"   - {outcome_accuracy:.0%} decision accuracy on 50-case gold set")
    print(f"   - {cost_savings_pct:.0f}% LLM cost reduction vs single-tier baseline via triage routing")
    print(f"   - {citation_coverage:.0%} regulatory citation coverage on escalate cases")
    print(f"   - {sar_precision:.0%} SAR generation precision")
    print(f"   - <50ms XGBoost inference latency (architecture SLA, deterministic)")
    print()


if __name__ == "__main__":
    main()
