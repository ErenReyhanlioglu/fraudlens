"""Score test.parquet and build stratified scenario samples for integration tests.

Outputs data/processed/test_scenarios.jsonl — one JSON object per line:
  expected_bucket  : str  (approve_low | approve_high | investigate_low |
                           investigate_high | critical_low | critical_high)
  actual_score     : float
  is_fraud         : int  (0 or 1)
  triage_action    : str  (approve | investigate | escalate)
  raw_features     : dict[str, float|None]  — all IEEE-CIS model features
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "test.parquet"
MODEL_PATH = REPO_ROOT / "data" / "processed" / "xgb_tuned_v1.joblib"
OUT_PATH = REPO_ROOT / "data" / "processed" / "test_scenarios.jsonl"

# ---------------------------------------------------------------------------
# Triage thresholds (must match api/routers/transactions.py)
# ---------------------------------------------------------------------------

_APPROVE_THRESHOLD = 0.3
_ESCALATE_THRESHOLD = 0.7

BUCKETS: list[tuple[str, float, float]] = [
    ("approve_low", 0.00, 0.10),
    ("approve_high", 0.10, 0.30),
    ("investigate_30_40", 0.30, 0.40),
    ("investigate_40_50", 0.40, 0.50),
    ("investigate_50_60", 0.50, 0.60),
    ("investigate_60_70", 0.60, 0.70),
    ("critical_low", 0.70, 0.85),
    ("critical_high", 0.85, 1.00),
]

SAMPLES_PER_BUCKET = 50


def triage(prob: float) -> str:
    if prob < _APPROVE_THRESHOLD:
        return "approve"
    if prob < _ESCALATE_THRESHOLD:
        return "investigate"
    return "escalate"


def bucket_label(prob: float) -> str:
    if prob < 0.10:
        return "approve_low"
    if prob < 0.30:
        return "approve_high"
    if prob < 0.40:
        return "investigate_30_40"
    if prob < 0.50:
        return "investigate_40_50"
    if prob < 0.60:
        return "investigate_50_60"
    if prob < 0.70:
        return "investigate_60_70"
    if prob < 0.85:
        return "critical_low"
    return "critical_high"


def load_model(path: Path) -> tuple[xgb.XGBClassifier | xgb.Booster, list[str]]:
    raw = joblib.load(path)
    if isinstance(raw, xgb.XGBClassifier):
        feature_names: list[str] = raw.get_booster().feature_names or []
        return raw, feature_names
    if isinstance(raw, xgb.Booster):
        feature_names = raw.feature_names or []
        return raw, feature_names
    raise TypeError(f"Unexpected artifact type: {type(raw).__name__}")


def predict_proba(
    model: xgb.XGBClassifier | xgb.Booster,
    feature_names: list[str],
    df: pd.DataFrame,
) -> np.ndarray:
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=np.nan)
    if isinstance(model, xgb.XGBClassifier):
        return model.predict_proba(df)[:, 1]
    dmatrix = xgb.DMatrix(df)
    return model.predict(dmatrix)


def main() -> None:
    if not PARQUET_PATH.exists():
        print(f"ERROR: {PARQUET_PATH} not found. Run scripts/download_data.py first.")
        sys.exit(1)

    print(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  shape: {df.shape}")

    model, feature_names = load_model(MODEL_PATH)
    print(f"  model features: {len(feature_names)}")

    label_col = "isFraud"
    if label_col not in df.columns:
        print(f"ERROR: '{label_col}' column not found in parquet.")
        sys.exit(1)

    labels = df[label_col].values
    feature_df = df.drop(columns=["TransactionID", label_col], errors="ignore")

    print("Scoring all rows ...")
    scores = predict_proba(model, feature_names, feature_df)
    print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Align feature columns once (same reindex as predict_proba above)
    aligned_features = feature_df.reindex(columns=feature_names, fill_value=np.nan)

    scenarios: list[dict] = []

    for bucket_name, lo, hi in BUCKETS:
        mask = (scores >= lo) & (scores < hi)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            print(f"  WARN: no samples in bucket '{bucket_name}' [{lo:.2f}, {hi:.2f})")
            continue

        # Attempt balanced fraud/legit sampling; fall back to random if needed.
        fraud_idx = idx[labels[idx] == 1]
        legit_idx = idx[labels[idx] == 0]

        half = SAMPLES_PER_BUCKET // 2
        rng = np.random.default_rng(42)

        selected: list[int] = []
        if len(fraud_idx) >= half and len(legit_idx) >= half:
            selected = list(rng.choice(fraud_idx, half, replace=False)) + list(rng.choice(legit_idx, half, replace=False))
        else:
            n = min(SAMPLES_PER_BUCKET, len(idx))
            selected = list(rng.choice(idx, n, replace=False))

        for i in selected:
            raw: dict = {k: (None if np.isnan(v) else float(v)) for k, v in aligned_features.iloc[i].items()}
            scenarios.append(
                {
                    "expected_bucket": bucket_name,
                    "actual_score": float(scores[i]),
                    "is_fraud": int(labels[i]),
                    "triage_action": triage(float(scores[i])),
                    "raw_features": raw,
                }
            )

        print(f"  bucket '{bucket_name}': {len(selected)} samples (fraud={sum(labels[s] == 1 for s in selected)}, legit={sum(labels[s] == 0 for s in selected)})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for row in scenarios:
            fh.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(scenarios)} scenarios -> {OUT_PATH}")


if __name__ == "__main__":
    main()
