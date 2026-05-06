"""Manual polling test: POST /transactions → poll GET /jobs/{job_id} until done.

Runs two scenarios:
  1. Low-risk  (auto-approve, no agent)
  2. High-risk (critical agent + SAR, loaded from test_scenarios.jsonl)

Usage:
    uv run python scripts/test_polling.py [--port PORT]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_PATH = REPO_ROOT / "data" / "processed" / "test_scenarios.jsonl"

_TX_BASE = {
    "timestamp": "2026-04-30T12:00:00Z",
    "currency": "USD",
    "transaction_type": "transfer",
    "channel": "api",
    "sender_account_id": "POLL-TEST-001",
    "sender_bank_code": "TEST0001",
    "sender_country": "US",
    "receiver_account_id": "POLL-TEST-002",
    "receiver_bank_code": "TEST0002",
    "receiver_country": "DE",
    "amount": 100.0,
}

LOW_RISK_PAYLOAD = {
    **_TX_BASE,
    "channel": "online",
    "ip_address": "10.0.0.1",
    "amount": 12.50,
    "transaction_type": "payment",
    "sender_country": "TR",
    "receiver_country": "TR",
}


def _load_critical_payload() -> dict | None:
    """Return a raw_mode payload from the first critical_high or critical_low scenario."""
    if not SCENARIOS_PATH.exists():
        return None
    with SCENARIOS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("expected_bucket") in ("critical_high", "critical_low"):
                return {
                    **_TX_BASE,
                    "transaction_id": str(uuid.uuid4()),
                    "raw_features": row["raw_features"],
                }
    return None


async def _poll(client: httpx.AsyncClient, base_url: str, job_id: str, label: str) -> dict:
    t0 = time.perf_counter()
    seen: set[str] = set()
    while True:
        await asyncio.sleep(0.5)
        resp = await client.get(f"{base_url}/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()

        status = data["status"]
        key = f"{status}:{data.get('current_tool')}"
        elapsed = time.perf_counter() - t0

        if key not in seen:
            seen.add(key)
            tool_tag = f"  tool={data['current_tool']}" if data.get("current_tool") else ""
            print(f"  [{label}] [{elapsed:6.1f}s] {status:<14} {data['stage_label']}{tool_tag}")

        if status in ("done", "error"):
            return data


async def _run_scenario(
    client: httpx.AsyncClient,
    base_url: str,
    label: str,
    payload: dict,
    raw_mode: bool = False,
) -> None:
    params = {"raw_mode": "true"} if raw_mode else {}
    resp = await client.post(f"{base_url}/transactions", json=payload, params=params)
    resp.raise_for_status()
    submit = resp.json()
    job_id = submit["job_id"]
    print(f"\n[{label}] submitted  job_id={job_id}")

    t0 = time.perf_counter()
    data = await _poll(client, base_url, job_id, label)
    elapsed = time.perf_counter() - t0

    if data["status"] == "error":
        print(f"  [{label}] ERROR: {data['error']}")
        return

    result = data.get("result") or {}
    print(f"  [{label}] done in {elapsed:.2f}s")
    print(f"    fraud_probability : {result.get('fraud_probability'):.4f}")
    print(f"    risk_tier         : {result.get('risk_tier')}")
    print(f"    triage_action     : {result.get('triage_action')}")
    print(f"    processing_ms     : {result.get('processing_time_ms', 0):.0f}")
    if result.get("fraud_decision"):
        print(f"    outcome           : {result['fraud_decision'].get('outcome')}")
    if result.get("sar_report"):
        print(f"    sar_report        : present  subject={result['sar_report'].get('subject_name')}")


async def main(port: int) -> None:
    base_url = f"http://localhost:{port}/api/v1"

    async with httpx.AsyncClient(timeout=180.0) as client:
        # --- Scenario 1: low-risk, auto-approve ---
        await _run_scenario(client, base_url, "LOW ", LOW_RISK_PAYLOAD)

        # --- Scenario 2: high-risk, critical agent + SAR ---
        critical_payload = _load_critical_payload()
        if critical_payload is None:
            print(
                "\n[HIGH] SKIP — test_scenarios.jsonl not found.\n"
                "       Run: uv run python scripts/create_test_scenarios.py"
            )
            return
        await _run_scenario(client, base_url, "HIGH", critical_payload, raw_mode=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    asyncio.run(main(args.port))
