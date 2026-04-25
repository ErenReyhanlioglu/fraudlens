"""One-time script: enrich test_scenarios.jsonl with banking_context.

Reads  data/processed/test_scenarios.jsonl
Writes data/processed/test_scenarios_enriched.jsonl

Each output line = original line + {"banking_context": {...}} key.
Run once from the project root:

    uv run scripts/enrich_scenarios.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is on the path when running from project root without install.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fraudlens.ml.feature_extractor import enrich_with_context  # noqa: E402

INPUT = Path("data/processed/test_scenarios.jsonl")
OUTPUT = Path("data/processed/test_scenarios_enriched.jsonl")


def main() -> None:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found — run from the project root.")
        sys.exit(1)

    lines = INPUT.read_text(encoding="utf-8").splitlines()
    enriched: list[str] = []

    for _i, line in enumerate(lines):
        if not line.strip():
            continue
        row = json.loads(line)
        raw = row.get("raw_features") or {}
        row["banking_context"] = enrich_with_context(raw)
        enriched.append(json.dumps(row, ensure_ascii=False))

    OUTPUT.write_text("\n".join(enriched) + "\n", encoding="utf-8")
    print(f"Wrote {len(enriched)} enriched scenarios -> {OUTPUT}")


if __name__ == "__main__":
    main()
