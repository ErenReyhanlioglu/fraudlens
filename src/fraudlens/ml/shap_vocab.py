"""Converts SHAP feature contributions into verbal descriptions for the agent.

Three tiers:
  - Definitely interpretable: produce specific, actionable descriptions.
  - Partially interpretable: produce general signal warnings.
  - Blackbox (V*, most card/addr/M aggregations, encoded categoricals): silently dropped.

An empty list is a valid return — the agent uses ml_score and tool outputs.

Note: ``v`` throughout is the SHAP contribution (float), not the raw feature value.
Positive v = pushes toward fraud. Thresholds are calibrated against observed
SHAP magnitudes in the IEEE-CIS model (typical range 0.1–2.0 for top features).
"""

from __future__ import annotations

from typing import Any

_THRESHOLD = 0.1  # minimum |SHAP| to report anything


def annotate_shap(shap_top10: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return interpretable SHAP features merged with their plain-language meaning.

    Non-interpretable features (V*, card*, M*, addr*, encoded categoricals) are
    dropped entirely.  If nothing is interpretable the list is empty — the agent
    relies on ml_score and tool outputs instead.

    Args:
        shap_top10: List of dicts with keys ``feature`` (str) and ``shap``
            (float, positive = pushes toward fraud).

    Returns:
        List of ``{"feature": str, "shap": float, "meaning": str}`` dicts,
        one per interpretable feature, in descending |shap| order.
    """
    results: list[dict[str, Any]] = []
    for entry in shap_top10:
        feature = entry.get("feature", "")
        v = float(entry.get("shap", 0.0))
        meaning = _describe(feature, v)
        if meaning:
            results.append({"feature": feature, "shap": round(v, 4), "meaning": meaning})
    return results


def _describe(feature: str, v: float) -> str | None:  # noqa: PLR0911, PLR0912, PLR0915
    # -------------------------------------------------------------------------
    # DEFINITELY INTERPRETABLE
    # -------------------------------------------------------------------------

    # --- Amount vs customer baseline ---
    if feature == "amt_zscore_uid":
        if v > _THRESHOLD:
            return f"Amount {abs(v):.1f}σ above customer average"
        if v < -_THRESHOLD:
            return f"Amount {abs(v):.1f}σ below customer average"
        return None

    if feature == "amt_to_median_uid":
        if v > _THRESHOLD:
            return "Amount significantly above customer median"
        return None

    if feature == "uid_TransactionAmt_mean":
        if v > _THRESHOLD:
            return "Customer transaction history shows elevated amounts"
        return None

    if feature == "uid_TransactionAmt_std":
        if v > _THRESHOLD:
            return "High variance in customer transaction history"
        return None

    if feature == "uid_TransactionAmt_min":
        if v > _THRESHOLD:
            return "Amount above customer's minimum transaction threshold"
        return None

    if feature == "card1_TransactionAmt_min":
        if v > _THRESHOLD:
            return "Amount above minimum seen for this card"
        return None

    if feature == "addr1_TransactionAmt_min":
        if v > _THRESHOLD:
            return "Amount above minimum seen for this billing area"
        return None

    # --- Customer behaviour: address & velocity counts ---
    if feature == "C1":
        if v > 0.3:
            return "Elevated address count linked to this card"
        if v < -0.3:
            return "Normal address count for this card"
        return None

    if feature == "C5":
        if v > _THRESHOLD:
            return "High transaction count linked to this card"
        return None

    if feature == "uid_C1_mean":
        if v > _THRESHOLD:
            return "Customer typically has many linked addresses"
        return None

    if feature == "card1_C1_max":
        if v > _THRESHOLD:
            return "High address count seen for this card historically"
        return None

    # --- Customer behaviour: timing / recency (D-features) ---
    if feature == "D1":
        if v > _THRESHOLD:
            return "Very recent previous transaction — velocity risk"
        if v < -_THRESHOLD:
            return "Long gap since last transaction (unusual pattern)"
        return None

    if feature == "D3":
        if v > _THRESHOLD:
            return "Recent transaction at a different merchant"
        return None

    if feature == "D10":
        if v > _THRESHOLD:
            return "Recent transaction from a different device"
        return None

    if feature == "D15":
        if v > _THRESHOLD:
            return "Unusual gap since last large transaction"
        return None

    if feature == "uid_D1_mean":
        if v > _THRESHOLD:
            return "Customer's typical transaction gap is large — velocity anomaly possible"
        return None

    # --- Time signals ---
    if feature in ("dt_hour_sin", "dt_hour_cos"):
        if abs(v) > 0.2:
            return "Unusual transaction hour"
        return None

    if feature == "is_weekend":
        if v > _THRESHOLD:
            return "Weekend transaction"
        return None

    # --- Email & identity ---
    if feature == "P_emaildomain_freq":
        if v > _THRESHOLD:
            return "Uncommon sender email domain"
        return None

    if feature == "email_match":
        if v > _THRESHOLD:
            return "Sender and receiver email domains differ"
        return None

    if feature == "uid_dist1_mean":
        if v > _THRESHOLD:
            return "Customer's typical transaction distance suggests geographic spread"
        return None

    # -------------------------------------------------------------------------
    # PARTIALLY INTERPRETABLE — general signal only
    # -------------------------------------------------------------------------

    if feature == "id_01":
        if v > _THRESHOLD:
            return "Elevated velocity risk score detected"
        return None

    if feature == "id_02":
        if v > _THRESHOLD:
            return "Device age signal flagged"
        if v < -_THRESHOLD:
            return "New or unrecognized device"
        return None

    if feature == "id_12":
        if v > _THRESHOLD:
            return "Identity verification signal flagged"
        return None

    if feature == "id_30":
        if v > _THRESHOLD:
            return "Browser or OS anomaly signal"
        return None

    if feature == "D2_is_null":
        if v > _THRESHOLD:
            return "Transaction timing data missing"
        return None

    if feature == "dist1":
        if v > _THRESHOLD:
            return "Elevated billing-to-transaction location distance"
        return None

    # -------------------------------------------------------------------------
    # BLACKBOX — silently drop
    # V*, card1-6, addr1-2, M*, ProductCD, R_emaildomain, P_emaildomain,
    # DeviceInfo, uid_dist1_mean aggregations, card3_*/card5_*/addr1_* aggs,
    # uid_D1_nunique, uid_C1_max, id_17 and other id_* not listed above
    # -------------------------------------------------------------------------
    return None
