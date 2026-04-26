"""Integration test: POST one sample per bucket to /transactions?raw_mode=true.

Usage:
    python scripts/run_integration_test.py [--base-url URL] [--samples N]

Reads data/processed/test_scenarios.jsonl, picks N samples per bucket,
posts each to the API, and reports pass/fail comparing expected vs actual
triage_action.

Exit code: 0 = all passed, 1 = any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_PATH = REPO_ROOT / "data" / "processed" / "test_scenarios.jsonl"

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SAMPLES_PER_BUCKET = 3

BUCKET_TO_TRIAGE = {
    "approve_low": "approve",
    "approve_high": "approve",
    "investigate_low": "investigate",
    "investigate_high": "investigate",
    "critical_low": "escalate",
    "critical_high": "escalate",
}

# Minimal required fields for a valid TransactionRequest.
_TX_TEMPLATE: dict = {
    "timestamp": "2024-01-15T10:00:00Z",
    "amount": "100.00",
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


def load_scenarios(path: Path) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            buckets[row["expected_bucket"]].append(row)
    return dict(buckets)


def build_payload(scenario: dict) -> dict:
    payload = dict(_TX_TEMPLATE)
    payload["transaction_id"] = str(uuid.uuid4())
    payload["raw_features"] = scenario["raw_features"]
    return payload


def post_transaction(client: httpx.Client, base_url: str, payload: dict) -> dict:
    resp = client.post(
        f"{base_url}/api/v1/transactions",
        params={"raw_mode": "true"},
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="FraudLens integration test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_BUCKET)
    args = parser.parse_args()

    if not SCENARIOS_PATH.exists():
        print(f"ERROR: {SCENARIOS_PATH} not found.")
        print("       Run: python scripts/create_test_scenarios.py")
        sys.exit(1)

    scenarios = load_scenarios(SCENARIOS_PATH)
    print(f"Loaded scenarios: {sum(len(v) for v in scenarios.values())} total rows")
    print(f"Target: {args.base_url}")
    print(f"Samples per bucket: {args.samples}\n")

    import random

    rng = random.Random(42)

    passed = 0
    failed = 0
    errors = 0

    with httpx.Client() as client:
        for bucket_name in sorted(BUCKET_TO_TRIAGE):
            expected_triage = BUCKET_TO_TRIAGE[bucket_name]
            rows = scenarios.get(bucket_name, [])

            if not rows:
                print(f"  [{bucket_name}] SKIP — no scenarios loaded")
                continue

            sample = rng.sample(rows, min(args.samples, len(rows)))

            for row in sample:
                payload = build_payload(row)
                score_label = f"{row['actual_score']:.3f} (fraud={row['is_fraud']})"

                try:
                    result = post_transaction(client, args.base_url, payload)
                except httpx.HTTPStatusError as exc:
                    print(f"  [{bucket_name}] ERROR {exc.response.status_code} — score={score_label}\n    detail: {exc.response.text[:400]}")
                    errors += 1
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{bucket_name}] ERROR {exc} — score={score_label}")
                    errors += 1
                    continue

                actual_triage = result.get("triage_action", "?")
                actual_prob = result.get("fraud_probability", -1.0)

                ok = actual_triage == expected_triage
                status = "PASS" if ok else "FAIL"
                if ok:
                    passed += 1
                else:
                    failed += 1

                print(f"  [{bucket_name}] {status} — expected={expected_triage} actual={actual_triage} prob={actual_prob:.4f} (parquet_score={score_label})")

    total = passed + failed + errors
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed, {errors} errors")

    if failed > 0 or errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
