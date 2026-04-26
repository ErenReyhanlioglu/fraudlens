"""Maps a TransactionRequest to the feature vector expected by FraudScorer.

Design contract
---------------
* All ``compute_*`` logic ran at training time and was serialised to JSON.
  This module only calls ``apply_*`` functions — never ``compute_*``.
* IEEE-CIS fields with no real-world equivalent are left as NaN.
  XGBoost handles missing values natively via its learned split directions,
  so NaN is strictly correct, not a hack.
* ``card1`` and ``addr1`` are hashed to integer keys so aggregation lookups
  in ``aggregation_mappings.json`` can fire when the same account/country
  appears in multiple requests. Unseen keys fall back to ``global_mean``
  (the Bayesian-smoothing ``count=0`` limit).
* ``ProductCD`` uses a rule-based mapping (see ``_PRODUCT_CD_MAP``).
  Callers can override via ``metadata['product_cd']``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.core.exceptions import FraudLensError
from fraudlens.core.logging import get_logger
from fraudlens.ml.preprocessor import (
    apply_aggregation_mappings,
    apply_domain_mappings,
    apply_drop_columns,
    apply_encoding_mappings,
    apply_missing_strategy,
    apply_time_features,
    create_uid_feature,
    normalize_d_features,
)
from fraudlens.schemas.transaction import Channel, TransactionRequest, TransactionType

logger = get_logger(__name__)


class ExtractorNotLoadedError(FraudLensError):
    """Raised when ``transform()`` is called before ``load()``."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# reference point.  All timestamps are converted relative to this epoch.
_REFERENCE_EPOCH = datetime(2017, 11, 1, tzinfo=UTC)

# card1 range observed in IEEE-CIS training data.
_CARD1_MAX = 18_396

# addr1 range observed in IEEE-CIS training data.
_ADDR1_MIN = 100
_ADDR1_RANGE = 400

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# Best-effort mapping based on community reverse-engineering of Vesta codes:
#   W = Web / digital transfer
#   C = Cash / card-not-present payment
#   H = Home / physical retail purchase
#   S = Service / subscription / account operation
#   R = Refund / return  (not used below — we have no refund type yet)
_PRODUCT_CD_MAP: dict[TransactionType, str] = {
    TransactionType.TRANSFER: "W",
    TransactionType.PAYMENT: "C",
    TransactionType.WITHDRAWAL: "C",
    TransactionType.DEPOSIT: "S",
    TransactionType.PURCHASE: "H",
}

# Channel → IEEE-CIS DeviceType.
_DEVICE_TYPE_MAP: dict[Channel, str] = {
    Channel.ONLINE: "desktop",
    Channel.MOBILE: "mobile",
    Channel.ATM: "desktop",
    Channel.BRANCH: "desktop",
    Channel.API: "desktop",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_to_int(value: str, max_val: int) -> int:
    """Stable, deterministic hash of *value* to the integer range [1, max_val].

    SHA-256 is used so the distribution is uniform and the mapping never
    changes between process restarts (unlike Python's built-in ``hash``).
    """
    digest = int(hashlib.sha256(value.encode()).hexdigest(), 16)
    return (digest % max_val) + 1


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InferenceExtractor:
    """Maps a single ``TransactionRequest`` to a model-ready DataFrame.

    The extractor mirrors the ``apply_*`` steps of the training pipeline
    (``01_eda_ieee_cis.ipynb``) in the same order, using the JSON rule files
    written by the corresponding ``compute_*`` calls.

    Usage::

        extractor = InferenceExtractor()
        extractor.load(Path("data/processed"))
        features_df = extractor.transform(tx_request)
        prob, shap = scorer.score(features_df)
    """

    def __init__(self) -> None:
        self._loaded = False
        self._missing_strategy: dict[str, Any] = {}
        self._agg_mappings: dict[str, Any] = {}
        self._domain_mappings: dict[str, Any] = {}
        self._enc_mappings: dict[str, Any] = {}
        self._time_rules: dict[str, Any] = {}
        # Columns to drop before aggregation (low-variance + V-feature pruning)
        self._drop_pre: list[str] = []
        # Columns to drop after encoding (correlation + adversarial + PSI)
        self._drop_post: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, rules_dir: Path) -> None:
        """Load all fitted rule files from *rules_dir*.

        Args:
            rules_dir: Directory that contains the 10 JSON artefacts written
                by the training pipeline (e.g. ``data/processed/``).

        Raises:
            FileNotFoundError: If any required rule file is absent.
        """

        def _read(name: str) -> Any:
            path = rules_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Rule file not found: {path}")
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)

        self._missing_strategy = _read("missing_strategy.json")
        self._agg_mappings = _read("aggregation_mappings.json")
        self._domain_mappings = _read("domain_mappings.json")
        self._enc_mappings = _read("encoding_mappings.json")
        self._time_rules = _read("time_rules.json")

        low_var: list[str] = _read("drop_low_variance.json")
        v_feats_raw = _read("drop_v_features.json")
        v_feats: list[str] = v_feats_raw.get("dropped_v_cols", []) if isinstance(v_feats_raw, dict) else v_feats_raw
        self._drop_pre = list(low_var) + list(v_feats)

        corr: list[str] = _read("drop_corr.json")
        adv: list[str] = _read("drop_adversarial.json")
        psi_raw = _read("drop_psi.json")
        psi: list[str] = psi_raw.get("psi_drift_columns", []) if isinstance(psi_raw, dict) else psi_raw
        self._drop_post = list(corr) + list(adv) + list(psi)

        self._loaded = True
        logger.info(
            "inference_extractor_loaded",
            rules_dir=str(rules_dir),
            drop_pre=len(self._drop_pre),
            drop_post=len(self._drop_post),
        )

    def transform(self, tx: TransactionRequest) -> pd.DataFrame:
        """Transform a single transaction into a model-ready DataFrame.

        Mirrors the ``apply_*`` pipeline from ``01_eda_ieee_cis.ipynb``
        in the same order.  Fields that cannot be derived from ``tx`` are
        left as NaN — XGBoost handles them via learned split directions.

        Args:
            tx: Validated transaction payload from the API layer.

        Returns:
            Single-row DataFrame.  ``FraudScorer.score`` will align it to
            the model's exact feature list before prediction.

        Raises:
            ExtractorNotLoadedError: If ``load()`` has not been called.
        """
        if not self._loaded:
            raise ExtractorNotLoadedError("InferenceExtractor has not been loaded.")

        df = self._build_raw_row(tx)

        # Steps mirror the training pipeline (apply_* only, never compute_*).

        df = normalize_d_features(df, verbose=False)

        df = create_uid_feature(df, verbose=False)

        flag_cols: list[str] = self._missing_strategy.get("flag_columns", [])
        missing_flag_cols = {col: np.nan for col in flag_cols if col not in df.columns}
        if missing_flag_cols:
            df = pd.concat([df, pd.DataFrame([missing_flag_cols], index=df.index)], axis=1)

        df = apply_missing_strategy(df, self._missing_strategy, verbose=False)

        df = apply_drop_columns(df, self._drop_pre, verbose=False)

        df = apply_aggregation_mappings(df, self._agg_mappings, verbose=False)

        df = apply_domain_mappings(df, self._domain_mappings, verbose=False)

        df = apply_time_features(df, self._time_rules, verbose=False)

        df = apply_encoding_mappings(df, self._enc_mappings, verbose=False)

        df = apply_drop_columns(df, self._drop_post, verbose=False)

        if "uid" in df.columns:
            df = df.drop(columns=["uid"])

        return df

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Raw row builder
    # ------------------------------------------------------------------

    def _build_raw_row(self, tx: TransactionRequest) -> pd.DataFrame:
        """Map TransactionRequest fields to the IEEE-CIS column space.

        Fields with no reliable equivalent are excluded (→ NaN after
        DataFrame construction).  See module docstring for rationale.

        Metadata pass-through keys
        --------------------------
        ``metadata['product_cd']``    — override ProductCD (W/H/C/S/R)
        ``metadata['card6']``         — "debit" or "credit"
        ``metadata['p_emaildomain']`` — purchaser email domain
        ``metadata['r_emaildomain']`` — recipient email domain
        ``metadata['id_30']``         — OS version string (e.g. "Windows 10")
        """
        meta: dict[str, Any] = tx.metadata or {}

        # --- TransactionDT ---
        ts = tx.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        txn_dt = (ts - _REFERENCE_EPOCH).total_seconds()

        # --- ProductCD: rule-based map + metadata override ---
        product_cd: str | None = meta.get("product_cd") or _PRODUCT_CD_MAP.get(tx.transaction_type)

        # --- card1: stable hash of sender_account_id into IEEE-CIS range ---
        card1 = _hash_to_int(tx.sender_account_id, _CARD1_MAX)

        # --- addr1: hash of sender_country into addr1 numeric range ---
        addr1 = _hash_to_int(tx.sender_country, _ADDR1_RANGE) + _ADDR1_MIN

        row: dict[str, Any] = {
            "TransactionAmt": float(tx.amount),
            "TransactionDT": txn_dt,
            "ProductCD": product_cd,
            "card1": card1,
            "card6": meta.get("card6"),  # "debit" / "credit" if known
            "addr1": addr1,
            "P_emaildomain": meta.get("p_emaildomain"),
            "R_emaildomain": meta.get("r_emaildomain"),
            "DeviceType": _DEVICE_TYPE_MAP.get(tx.channel),
            "DeviceInfo": tx.device_fingerprint,
            "id_30": meta.get("id_30"),  # OS version, e.g. "Windows 10"
            # D1 must exist as float NaN (not Python None) so the column dtype is
            "D1": np.nan,
        }

        return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Banking context enrichment (IEEE-CIS → human-readable agent payload)
# ---------------------------------------------------------------------------

# ProductCD label classes in the order stored by encoding_mappings.json.
# Used to reverse the label-encoded integer back to the code letter.
_PRODUCTCD_CLASSES: list[str] = []

_PRODUCTCD_TO_TYPE: dict[str, str] = {
    "W": "transfer",
    "C": "payment",
    "H": "purchase",
    "S": "deposit",
    "R": "refund",
}


def _load_productcd_classes() -> list[str]:
    global _PRODUCTCD_CLASSES
    if _PRODUCTCD_CLASSES:
        return _PRODUCTCD_CLASSES
    try:
        path = Path("data/processed/encoding_mappings.json")
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        _PRODUCTCD_CLASSES = data.get("label_classes", {}).get("ProductCD", [])
    except Exception:
        pass
    if not _PRODUCTCD_CLASSES:
        _PRODUCTCD_CLASSES = ["C", "H", "R", "S", "W"]
    return _PRODUCTCD_CLASSES


def _reconstruct_hour(raw: dict[str, Any]) -> int | None:
    """Estimate hour-of-day from dt_hour_sin / dt_hour_cos encoding.

    Returns None when either component is missing.
    """
    import math

    sin_val = raw.get("dt_hour_sin")
    cos_val = raw.get("dt_hour_cos")
    if sin_val is None or cos_val is None:
        return None
    try:
        angle = math.atan2(float(sin_val), float(cos_val))
        return round(angle * 24 / (2 * math.pi)) % 24
    except (TypeError, ValueError):
        return None


def enrich_with_context(raw_features: dict[str, Any]) -> dict[str, Any]:
    """Derive a human-readable banking context dict from IEEE-CIS raw features.

    Produces a deterministic, ~13-field dict suitable for passing to the
    investigation agent instead of the full 79-feature vector.  NaN / None
    values are replaced with sensible defaults; no exceptions are raised.

    Args:
        raw_features: Dict of IEEE-CIS feature name → value (as stored in
            test_scenarios.jsonl or returned by score_raw).

    Returns:
        Banking context dict with keys: customer_avg_amount, amount_note,
        timestamp, transaction_type, sender_account_id, merchant_id,
        ip_address, device_fingerprint, sender_country, receiver_country,
        currency, channel, is_weekend, is_night, hour_of_day.
    """

    def _safe_int(val: Any, default: int = 0) -> int:
        try:
            if val is None:
                return default
            f = float(val)
            return default if f != f else int(f)  # NaN check
        except (TypeError, ValueError):
            return default

    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None:
                return default
            f = float(val)
            return default if f != f else f
        except (TypeError, ValueError):
            return default

    customer_avg_amount = round(_safe_float(raw_features.get("uid_TransactionAmt_mean"), 0.0), 2)

    dt_val = raw_features.get("TransactionDT")
    if dt_val is not None:
        try:
            from datetime import timedelta

            ts = _REFERENCE_EPOCH + timedelta(seconds=float(dt_val))
            timestamp = ts.isoformat()
        except Exception:
            timestamp = _REFERENCE_EPOCH.isoformat()
    else:
        timestamp = _REFERENCE_EPOCH.isoformat()

    classes = _load_productcd_classes()
    product_cd_idx = _safe_int(raw_features.get("ProductCD"), -1)
    product_cd = classes[product_cd_idx] if 0 <= product_cd_idx < len(classes) else "W"
    transaction_type = _PRODUCTCD_TO_TYPE.get(product_cd, "transfer")

    card1 = _safe_int(raw_features.get("card1"), 0)
    addr1 = _safe_int(raw_features.get("addr1"), 0)
    device_info = _safe_int(raw_features.get("DeviceInfo"), 0)

    return {
        "customer_avg_amount": customer_avg_amount,
        "amount_note": "exact transaction amount unavailable; customer average shown",
        "timestamp": timestamp,
        "transaction_type": transaction_type,
        "sender_account_id": f"ACC-{card1}",
        "merchant_id": f"MERCH-{card1 % 1000:04d}",
        "ip_address": f"192.168.{card1 % 255}.{addr1 % 255}",
        "device_fingerprint": f"DEV-{device_info}",
        "sender_country": "US",
        "receiver_country": "US",
        "currency": "USD",
        "channel": "api",
        "is_weekend": bool(_safe_int(raw_features.get("is_weekend"), 0)),
        "is_night": bool(_safe_int(raw_features.get("is_night"), 0)),
        "hour_of_day": _reconstruct_hour(raw_features),
    }
