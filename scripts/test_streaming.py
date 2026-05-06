"""Streaming thought visibility test.

Sends a high-risk transaction and polls every 300ms to verify that:
- status transitions are visible in real time
- current_tool updates as each tool fires
- thought updates as the LLM reasons (visible DURING the agent run)

Usage:
    uv run python scripts/test_streaming.py [--port PORT]
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
    "sender_account_id": "STREAM-TEST-001",
    "sender_bank_code": "TEST0001",
    "sender_country": "US",
    "receiver_account_id": "STREAM-TEST-002",
    "receiver_bank_code": "TEST0002",
    "receiver_country": "DE",
    "amount": 100.0,
}


def _load_critical_payload() -> dict | None:
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


async def main(port: int) -> None:
    base_url = f"http://localhost:{port}/api/v1"
    payload = _load_critical_payload()

    if payload is None:
        print(
            "SKIP — test_scenarios.jsonl not found.\n"
            "Run: uv run python scripts/create_test_scenarios.py"
        )
        return

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{base_url}/transactions",
            json=payload,
            params={"raw_mode": "true"},
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        print(f"job_id  : {job_id}")
        print(f"{'─' * 72}")

        t0 = time.perf_counter()
        prev_status = ""
        prev_tool = ""
        prev_thought_tail = ""

        while True:
            await asyncio.sleep(0.3)
            poll = await client.get(f"{base_url}/jobs/{job_id}")
            poll.raise_for_status()
            data = poll.json()

            elapsed = time.perf_counter() - t0
            status = data["status"]
            tool = data.get("current_tool") or ""
            thought: str = data.get("thought") or ""
            completed = data.get("completed_tools") or []

            # Use the last 120 chars as fingerprint — catches both growing and
            # replaced thought buffers (SAR resets between phases)
            thought_tail = thought[-120:] if thought else ""

            status_changed = status != prev_status
            tool_changed = tool != prev_tool
            thought_changed = thought_tail != prev_thought_tail

            if status_changed or tool_changed or thought_changed:
                print(
                    f"[{elapsed:6.1f}s] {status:<14} "
                    + (f"tool={tool:<28}" if tool else " " * 33)
                    + f" done={completed}"
                )
                if thought and (status_changed or thought_changed):
                    preview = thought.replace("\n", " ")[-120:]
                    print(f"          thought: ...{preview}")

                prev_status = status
                prev_tool = tool
                prev_thought_tail = thought_tail

            if status in ("done", "error"):
                break

        elapsed = time.perf_counter() - t0
        print(f"{'─' * 72}")

        if data["status"] == "error":
            print(f"ERROR: {data['error']}")
            return

        result = data.get("result") or {}
        print(f"total time        : {elapsed:.2f}s")
        print(f"fraud_probability : {result.get('fraud_probability'):.4f}")
        print(f"risk_tier         : {result.get('risk_tier')}")
        print(f"triage_action     : {result.get('triage_action')}")
        print(f"processing_ms     : {result.get('processing_time_ms', 0):.0f}")
        if result.get("fraud_decision"):
            print(f"outcome           : {result['fraud_decision'].get('outcome')}")
        if result.get("sar_report"):
            sar = result["sar_report"]
            cinfo = sar.get("customer_info") or {}
            print(f"sar_report        : present")
            print(f"  sender_account  : {cinfo.get('sender_account_id', '(n/a)')}")
            print(f"  sender_country  : {cinfo.get('sender_country', '(n/a)')}")
            indicators = sar.get("suspicious_indicators") or []
            print(f"  indicators      : {indicators[:3]}")
            print(f"  rec_action      : {sar.get('recommended_action', '')[:80]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    asyncio.run(main(args.port))
