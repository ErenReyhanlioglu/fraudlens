"""Pre-computed pipeline results for the HF Spaces demo — all 49 eval cases.

Each entry matches the view_entry structure produced by the live FraudLens API.
Generated deterministically from gold_set metadata; no API calls required.
"""

from __future__ import annotations

import hashlib


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _det(case_id: str, salt: str, lo: float, hi: float) -> float:
    h = int(hashlib.sha1(f"{case_id}:{salt}".encode()).hexdigest()[:8], 16)
    return lo + (hi - lo) * h / 0xFFFFFFFF


def _shap_low(case_id: str) -> list[dict]:
    feats = ["C1", "card1", "TransactionAmt", "D1", "V45"]
    return [{"feature": f, "contribution": round(-_det(case_id, f"shap_{f}", 0.05, 0.65), 3)} for f in feats]


def _shap_med_review(case_id: str) -> list[dict]:
    pos = ["C1", "TransactionAmt", "is_night"]
    neg = ["D1", "card1"]
    out = [{"feature": f, "contribution": round(_det(case_id, f"shap_{f}", 0.18, 0.65), 3)} for f in pos]
    out += [{"feature": f, "contribution": round(-_det(case_id, f"shap_{f}", 0.10, 0.35), 3)} for f in neg]
    return out


def _shap_med_decline(case_id: str) -> list[dict]:
    feats = ["C1", "V45", "TransactionAmt", "card1", "D1"]
    return [{"feature": f, "contribution": round(_det(case_id, f"shap_{f}", 0.28, 0.95), 3)} for f in feats]


def _shap_high(case_id: str) -> list[dict]:
    feats = ["C1", "V45", "V307", "TransactionAmt", "card1", "C5", "D15"]
    vals = [_det(case_id, f"shap_{f}", 1.2, 2.8) for f in feats]
    vals[-2] *= 0.25
    vals[-1] *= 0.15
    return [{"feature": f, "contribution": round(v, 4)} for f, v in zip(feats, vals)]


# ---------------------------------------------------------------------------
# Evidence / flag templates
# ---------------------------------------------------------------------------

_REDFLAGS: dict[str, str] = {
    "geo_anomaly": "cross_border_destination",
    "velocity": "high_velocity",
    "aml": "aml_indicators",
    "structuring": "structuring_pattern",
    "sanctions": "sanctions_jurisdiction",
    "crypto_merchant": "high_risk_merchant",
    "fraud": "suspicious_pattern",
}

_EV_DECLINE: dict[str, str] = {
    "geo_anomaly": "Cross-border destination atypical for account; first-ever international transfer detected.",
    "velocity": "Transaction velocity anomaly: multiple large transactions in a short timeframe.",
    "aml": "AML typology match: pattern consistent with layering or placement stage.",
    "structuring": "Amount positioned just below MASAK mandatory reporting threshold — threshold-avoidance structuring.",
    "sanctions": "Destination jurisdiction or counterparty subject to international sanctions (OFAC/EU).",
    "crypto_merchant": "High-risk merchant category (cryptocurrency/money services); elevated fraud/AML typology.",
    "fraud": "XGBoost and agent converge on fraud classification with strong signal agreement.",
}

_EV_REVIEW: dict[str, str] = {
    "retail": "Merchant reputation clean; dispute rate within normal range for retail category.",
    "travel": "Travel-category merchant reputable; timing consistent with business hours.",
    "transfer": "Same-jurisdiction domestic routing reduces layering risk.",
    "atm": "ATM withdrawal within daily limits; domestic location confirmed.",
    "grocery": "Recurring grocery merchant; amount within typical basket range.",
    "subscription": "API channel consistent with recurring subscription payment patterns.",
    "legitimate": "Transaction pattern broadly consistent with account history; no confirmed fraud signals.",
}

_RAG_CITATIONS: list[dict] = [
    {
        "source": "MASAK_5549_Regulation.pdf",
        "article": None,
        "page": 12,
        "excerpt": (
            "Financial institutions shall report to MASAK any transaction that appears designed "
            "to evade reporting thresholds. Threshold structuring is a predicate indicator "
            "requiring immediate STR filing."
        ),
        "relevance_score": 0.81,
    },
    {
        "source": "FATF_40_RECOMMENDATIONS_2012.pdf",
        "article": None,
        "page": 13,
        "excerpt": (
            "Countries should require financial institutions to report suspicious transactions "
            "to the FIU when they suspect that funds are the proceeds of a criminal activity "
            "or terrorism financing."
        ),
        "relevance_score": 0.74,
    },
    {
        "source": "FATF_40_RECOMMENDATIONS_2012.pdf",
        "article": None,
        "page": 19,
        "excerpt": (
            "Financial institutions should apply enhanced due diligence measures for higher "
            "risk countries identified by the FATF as having strategic AML/CFT deficiencies."
        ),
        "relevance_score": 0.71,
    },
    {
        "source": "BDDK_AML_Rehberi_2021.pdf",
        "article": None,
        "page": 7,
        "excerpt": (
            "Bankaların şüpheli işlemleri MASAK'a bildirme yükümlülüğü, 5549 sayılı Kanun'un "
            "4. maddesi kapsamında, işlemin gerçekleşmesinden itibaren en geç iş günü sonuna "
            "kadar yerine getirilmelidir."
        ),
        "relevance_score": 0.68,
    },
]


# ---------------------------------------------------------------------------
# Tool-trace generators
# ---------------------------------------------------------------------------


def _trace_medium(m: dict, tools: list[str]) -> list[dict]:
    acct = m.get("sender_acct", "TR330006200119800006672335")
    ip = m.get("ip", "78.181.10.22")
    cid = m["id"]
    trace = []
    if "get_customer_history" in tools:
        trace.append({
            "tool": "get_customer_history",
            "args": {"account_id": acct},
            "result": (
                f'{{"transactions_30d": {int(_det(cid,"tx30",8,45))}, '
                f'"avg_amount": {int(_det(cid,"avg_amt",400,3200))}, '
                f'"account_age_days": {int(_det(cid,"age",60,800))}, '
                f'"prior_flags": {int(_det(cid,"flags",0,3))}}}'
            ),
        })
    if "check_merchant_reputation" in tools and m.get("merchant_id"):
        trace.append({
            "tool": "check_merchant_reputation",
            "args": {"merchant_id": m["merchant_id"]},
            "result": (
                f'{{"risk_score": {round(_det(cid,"mrisk",0.08,0.55),2)}, '
                f'"disputes_90d": {int(_det(cid,"disp",0,4))}, "category": "retail"}}'
            ),
        })
    if "get_geolocation_context" in tools and ip:
        is_proxy = "true" if any(t in m.get("tags", []) for t in ["geo_anomaly", "aml", "sanctions"]) else "false"
        trace.append({
            "tool": "get_geolocation_context",
            "args": {"ip_address": ip},
            "result": (
                f'{{"city": "Istanbul", "country": "{m.get("src_country","TR")}", '
                f'"is_proxy": {is_proxy}, "isp": "Turk Telekom"}}'
            ),
        })
    return trace


def _trace_high(m: dict) -> list[dict]:
    cid = m["id"]
    tags = m.get("tags", [])
    acct = m.get("sender_acct", "TR330006200119800006672355")
    ip = m.get("ip", "103.28.54.200")
    has_sanctions = "sanctions" in tags
    has_struct = "structuring" in tags
    return [
        {
            "tool": "explain_ml_score",
            "args": {},
            "result": '{"top_features": [{"feature": "C1", "shap": 2.11, "meaning": "address count anomaly"}, {"feature": "V45", "shap": 1.08, "meaning": "velocity spike"}]}',
        },
        {
            "tool": "get_customer_history",
            "args": {"account_id": acct},
            "result": (
                f'{{"transactions_30d": {int(_det(cid,"tx30",12,55))}, '
                f'"avg_amount": {int(_det(cid,"avg_amt",1200,8500))}, '
                f'"account_age_days": {int(_det(cid,"age",180,900))}, '
                f'"prior_flags": {int(_det(cid,"flags",0,3))}}}'
            ),
        },
        {
            "tool": "adverse_media_search",
            "args": {"account_id": acct},
            "result": (
                f'{{"sanctions_match": {"true" if has_sanctions else "false"}, '
                f'"pep_flag": false, "sdn_adjacent": {"true" if has_sanctions else "false"}}}'
            ),
        },
        {
            "tool": "deep_network_analysis",
            "args": {"transaction_id": m["txn_id"]},
            "result": (
                f'{{"nodes": {int(_det(cid,"nodes",3,8))}, '
                f'"circular_flow": {"true" if has_struct else "false"}, '
                f'"cross_border_risk": "{"critical" if has_sanctions else "high"}"}}'
            ),
        },
        {
            "tool": "regulatory_policy_rag",
            "args": {"query": "sanctions jurisdiction wire transfer reporting" if has_sanctions else "structuring threshold avoidance STR"},
            "result": (
                '{"excerpts": [{"text": "Financial institutions shall report to MASAK any '
                'transaction appearing designed to evade reporting thresholds.", '
                '"citation": "MASAK_5549_Regulation.pdf, p.12", "relevance_score": 0.81}]}'
            ),
        },
        {
            "tool": "find_similar_patterns",
            "args": {"transaction_id": m["txn_id"]},
            "result": (
                f'{{"similar_cases": {int(_det(cid,"similar",1,4))}, '
                f'"top_match_score": {round(_det(cid,"match",0.62,0.85),2)}, '
                f'"typologies": {str(tags[:2]).replace(chr(39), chr(34))}}}'
            ),
        },
        {
            "tool": "get_geolocation_context",
            "args": {"ip_address": ip},
            "result": (
                f'{{"city": "Unknown", "country": "{m.get("src_country","TR")}", '
                '"is_proxy": true, "isp": "Anonymous VPN"}}'
            ),
        },
    ]


# ---------------------------------------------------------------------------
# SAR builder
# ---------------------------------------------------------------------------


def _build_sar(m: dict, prob: float, conf: float, red_flags: list, tags: list) -> dict:
    acct = m.get("sender_acct", "TR330006200119800006672355")
    bank = m.get("sender_bank", "AKBNKTR")
    recv_country = m.get("dst_country", "VG")
    ts = m.get("timestamp", "2025-03-15T03:22:00+03:00")
    ip = m.get("ip", "103.28.54.200")
    device = m.get("device", "ANON-DEVICE-001")
    recv_bank = m.get("recv_bank", "UNKNOWN")
    has_sanctions = "sanctions" in tags
    has_struct = "structuring" in tags

    indicators = [
        f"ML fraud probability {prob:.4f} — critical risk threshold exceeded",
        f"Amount {m['amount']:,.1f} {m['currency']} significantly above MASAK reporting thresholds",
    ]
    if has_struct:
        indicators.append("Amount near MASAK 30,000 TRY threshold — deliberate structuring pattern")
    if has_sanctions:
        indicators.append(f"Destination {recv_country} — sanctions-listed jurisdiction; receiver bank sanctioned")
    if device:
        indicators.append(f"Device fingerprint '{device}' indicates anonymization or automation")
    indicators.append(f"Prior suspicious flags: {int(_det(m['id'],'flags',0,3))}")

    reg_triggers = [
        "MASAK Article 4, Law No. 5549 — STR filing obligation",
        "FATF Recommendation 13 — Suspicious Transaction Reporting",
    ]
    if has_sanctions:
        reg_triggers += [
            "FATF Recommendation 19 — Enhanced measures for high-risk jurisdictions",
            "OFAC / EU sanctions compliance requirements",
        ]
    if has_struct:
        reg_triggers.append("FATF Recommendation 11 — Unusual Transaction Patterns")

    action = (
        f"1. File Şüpheli İşlem Bildirimi (ŞİB) with MASAK immediately under Law 5549 Art. 4. "
        f"2. Place hold and freeze outbound transfers on account {acct}. "
        f"3. {'Block all transactions to/from ' + recv_country + ' and notify correspondent banks.' if has_sanctions else 'Expand network graph analysis to map all connected accounts.'} "
        f"4. Initiate Enhanced Due Diligence: contact account holder, verify transaction purpose. "
        f"5. Escalate to senior compliance officer within 24 hours per BDDK AML procedure."
    )

    return {
        "transaction_id": m["txn_id"],
        "customer_info": {
            "account_id": acct,
            "bank_code": bank,
            "country": m.get("src_country", "TR"),
            "account_age_days": int(_det(m["id"], "age", 180, 900)),
            "prior_suspicious_flags": int(_det(m["id"], "flags", 0, 3)),
            "kyc_risk_tier": "high",
        },
        "transaction_details": {
            "amount": f"{m['amount']:,.1f} {m['currency']}",
            "channel": m["channel"],
            "timestamp": ts,
            "device_fingerprint": device,
            "ip_address": ip,
            "receiver_bank": recv_bank,
            "receiver_country": recv_country,
        },
        "suspicious_indicators": indicators,
        "investigation_summary": (
            f"Investigation of transaction {m['txn_id']} reveals convergence of AML signals: "
            f"{', '.join(tags[:3])}. XGBoost probability {prob:.4f}, agent confidence {conf:.2f}. "
            f"Red flags: {', '.join(red_flags[:4])}. "
            f"Regulatory obligations triggered under MASAK Law 5549 and FATF Recommendations 13/19."
        ),
        "regulatory_triggers": reg_triggers,
        "recommended_action": action,
        "generated_at": "2025-03-15T05:00:00+00:00",
        "agent_model": "claude-haiku-4-5-20251001",
    }


# ---------------------------------------------------------------------------
# Entry builders
# ---------------------------------------------------------------------------


def _txn(m: dict) -> dict:
    d = {
        "transaction_id": m["txn_id"],
        "amount": m["amount"],
        "currency": m["currency"],
        "channel": m["channel"],
        "sender_country": m["src_country"],
        "receiver_country": m["dst_country"],
    }
    if m.get("merchant"):
        d["merchant_name"] = m["merchant"]
    return d


def _low(m: dict) -> dict:
    prob = round(_det(m["id"], "prob", 0.02, 0.24), 4)
    ms = round(_det(m["id"], "ms", 800, 2200), 1)
    return {
        "case": {"case_id": m["id"], "description": m["desc"], "tier": "low", "tags": m["tags"], "expected_outcome": "approve", "transaction": _txn(m)},
        "result": {
            "transaction_id": m["txn_id"],
            "fraud_probability": prob,
            "risk_tier": "low",
            "triage_action": "approve",
            "shap_top_features": _shap_low(m["id"]),
            "processing_time_ms": ms,
            "investigation": None,
            "fraud_decision": None,
            "sar_report": None,
            "token_usage": None,
        },
        "elapsed_ms": round(ms + _det(m["id"], "extra", 100, 400), 1),
    }


def _med_decline(m: dict) -> dict:
    cid = m["id"]
    prob = round(_det(cid, "prob", 0.47, 0.68), 4)
    conf = round(_det(cid, "conf", 0.72, 0.89), 2)
    tags = m["tags"]
    red_flags = list(dict.fromkeys([_REDFLAGS[t] for t in tags if t in _REDFLAGS] or ["suspicious_pattern"]))
    if m.get("device"):
        red_flags.append("anomalous_device")
    red_flags = list(dict.fromkeys(red_flags))
    evidence = [
        f"XGBoost fraud probability: {prob:.4f} — elevated risk score",
        f"Amount {m['amount']:,.1f} {m['currency']} significantly above account baseline",
    ] + [_EV_DECLINE[t] for t in tags if t in _EV_DECLINE]
    if m.get("device"):
        evidence.append(f"Device fingerprint '{m['device']}' flagged as high-risk.")
    tools = ["get_customer_history", "check_merchant_reputation", "get_geolocation_context"]
    reasoning = (
        f"Transaction exhibits {', '.join(tags[:2])} signals. ML score {prob:.4f} with agent "
        f"confidence {conf:.2f} supports decline. Red flags: {', '.join(red_flags[:3])}."
    )
    ms = round(_det(cid, "ms", 18000, 38000), 1)
    inv = {"decision_hint": "suspicious", "confidence": conf, "evidence": evidence, "red_flags": red_flags, "tools_called": tools, "reasoning_summary": reasoning, "tool_trace": _trace_medium(m, tools), "cited_sources": []}
    fd = {"transaction_id": m["txn_id"], "outcome": "decline", "confidence": conf, "ml_score": prob, "agent_used": "investigation", "decision_hint": "suspicious", "evidence": evidence, "red_flags": red_flags, "regulatory_citations": [], "reasoning": reasoning, "tools_called": tools}
    return {
        "case": {"case_id": cid, "description": m["desc"], "tier": "medium", "tags": tags, "expected_outcome": "decline", "transaction": _txn(m)},
        "result": {
            "transaction_id": m["txn_id"], "fraud_probability": prob, "risk_tier": "medium", "triage_action": "investigate",
            "shap_top_features": _shap_med_decline(cid), "processing_time_ms": ms,
            "investigation": inv, "fraud_decision": fd, "sar_report": None,
            "token_usage": {"input_tokens": int(_det(cid, "itok", 4000, 8000)), "output_tokens": int(_det(cid, "otok", 500, 900))},
        },
        "elapsed_ms": round(ms + _det(cid, "extra", 800, 2000), 1),
    }


def _med_review(m: dict) -> dict:
    cid = m["id"]
    prob = round(_det(cid, "prob", 0.33, 0.63), 4)
    conf = round(_det(cid, "conf", 0.45, 0.62), 2)
    tags = m["tags"]
    red_flags = []
    if m.get("device") and "NEW" in m.get("device", "").upper():
        red_flags.append("new_device")
    red_flags.append("elevated_amount")
    red_flags = list(dict.fromkeys(red_flags))
    evidence = [
        "Customer history: normal velocity for account age; transaction within 1.5× average",
        f"Amount {m['amount']:,.1f} {m['currency']} mildly elevated but not anomalous",
    ] + [_EV_REVIEW[t] for t in tags if t in _EV_REVIEW]
    tools = ["get_customer_history", "get_geolocation_context"]
    reasoning = (
        f"Transaction risk {prob:.4f}: evidence is conflicting. Account history and merchant "
        "reputation are clean; timing adds minor suspicion. Inconclusive — routed to manual review."
    )
    ms = round(_det(cid, "ms", 14000, 28000), 1)
    inv = {"decision_hint": "inconclusive", "confidence": conf, "evidence": evidence, "red_flags": red_flags, "tools_called": tools, "reasoning_summary": reasoning, "tool_trace": _trace_medium(m, tools), "cited_sources": []}
    fd = {"transaction_id": m["txn_id"], "outcome": "manual_review", "confidence": conf, "ml_score": prob, "agent_used": "investigation", "decision_hint": "inconclusive", "evidence": evidence, "red_flags": red_flags, "regulatory_citations": [], "reasoning": reasoning, "tools_called": tools}
    return {
        "case": {"case_id": cid, "description": m["desc"], "tier": "medium", "tags": tags, "expected_outcome": "manual_review", "transaction": _txn(m)},
        "result": {
            "transaction_id": m["txn_id"], "fraud_probability": prob, "risk_tier": "medium", "triage_action": "investigate",
            "shap_top_features": _shap_med_review(cid), "processing_time_ms": ms,
            "investigation": inv, "fraud_decision": fd, "sar_report": None,
            "token_usage": {"input_tokens": int(_det(cid, "itok", 3500, 6000)), "output_tokens": int(_det(cid, "otok", 450, 750))},
        },
        "elapsed_ms": round(ms + _det(cid, "extra", 600, 1500), 1),
    }


def _high(m: dict) -> dict:
    cid = m["id"]
    prob = round(_det(cid, "prob", 0.72, 0.97), 4)
    conf = round(_det(cid, "conf", 0.88, 0.97), 2)
    tags = m["tags"]
    red_flags = list(dict.fromkeys([_REDFLAGS[t] for t in tags if t in _REDFLAGS] + ["night_transaction"]))
    evidence = [
        f"XGBoost fraud probability: {prob:.4f} — critical risk score",
        f"Amount {m['amount']:,.1f} {m['currency']} significantly above reporting thresholds",
    ] + [_EV_DECLINE[t] for t in tags if t in _EV_DECLINE]
    if m.get("device"):
        evidence.append(f"Device fingerprint '{m['device']}' indicates anonymization or automation.")
    tools = ["explain_ml_score", "get_customer_history", "adverse_media_search", "deep_network_analysis", "regulatory_policy_rag", "find_similar_patterns", "get_geolocation_context"]
    cited = ["MASAK_5549_Regulation.pdf, p.12", "FATF_40_RECOMMENDATIONS_2012.pdf, p.13"]
    reasoning = (
        f"Convergence of AML signals: {', '.join(tags[:3])}. ML score {prob:.4f}, agent "
        f"confidence {conf:.2f}. MASAK Law 5549 Art. 4 and FATF Rec. 13 mandate STR filing. "
        f"Red flags: {', '.join(red_flags[:4])}."
    )
    citations_fd = _RAG_CITATIONS[:2] if "structuring" in tags else _RAG_CITATIONS[1:3]
    ms = round(_det(cid, "ms", 85000, 135000), 1)
    inv = {"decision_hint": "suspicious", "confidence": conf, "evidence": evidence, "red_flags": red_flags, "tools_called": tools, "reasoning_summary": reasoning, "tool_trace": _trace_high(m), "cited_sources": cited}
    fd = {"transaction_id": m["txn_id"], "outcome": "escalate", "confidence": conf, "ml_score": prob, "agent_used": "critical", "decision_hint": "suspicious", "evidence": evidence, "red_flags": red_flags, "regulatory_citations": citations_fd, "reasoning": reasoning, "tools_called": tools}
    return {
        "case": {"case_id": cid, "description": m["desc"], "tier": "high", "tags": tags, "expected_outcome": "escalate", "transaction": _txn(m)},
        "result": {
            "transaction_id": m["txn_id"], "fraud_probability": prob, "risk_tier": "high", "triage_action": "escalate",
            "shap_top_features": _shap_high(cid), "processing_time_ms": ms,
            "investigation": inv, "fraud_decision": fd,
            "sar_report": _build_sar(m, prob, conf, red_flags, tags),
            "token_usage": {"input_tokens": int(_det(cid, "itok", 15000, 22000)), "output_tokens": int(_det(cid, "otok", 2400, 3500))},
        },
        "elapsed_ms": round(ms + _det(cid, "extra", 1500, 4000), 1),
    }


# ---------------------------------------------------------------------------
# Case metadata (from tests/eval/gold_set.jsonl)
# ---------------------------------------------------------------------------

_META: list[dict] = [
    # LOW — approve (case_001–020, case_047, case_048)
    {"id": "case_001", "desc": "Small grocery purchase, domestic, daytime", "tier": "low", "outcome": "approve", "tags": ["legitimate", "grocery"], "amount": 47.5, "currency": "TRY", "channel": "mobile", "merchant": "Migros Supermarket", "merchant_id": "MIGROS-IST-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000001", "ip": "78.181.10.22", "sender_acct": "TR330006200119800006672315", "sender_bank": "AKBNKTR"},
    {"id": "case_002", "desc": "Utility bill payment, domestic, morning", "tier": "low", "outcome": "approve", "tags": ["legitimate", "utility"], "amount": 285.0, "currency": "TRY", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000002", "ip": "78.181.10.23", "sender_acct": "TR330006200119800006672316", "sender_bank": "AKBNKTR"},
    {"id": "case_003", "desc": "Coffee shop mobile payment, daytime", "tier": "low", "outcome": "approve", "tags": ["legitimate", "restaurant"], "amount": 32.0, "currency": "TRY", "channel": "mobile", "merchant": "Starbucks Turkiye", "merchant_id": "STARBUCKS-TR-042", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000003", "ip": "88.226.33.11", "sender_acct": "TR330006200119800006672317", "sender_bank": "ISBKTR"},
    {"id": "case_004", "desc": "Pharmacy purchase, branch channel", "tier": "low", "outcome": "approve", "tags": ["legitimate", "pharmacy"], "amount": 68.75, "currency": "TRY", "channel": "branch", "merchant": "Seckin Eczane", "merchant_id": "ECZANE-TR-0891", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000004", "ip": None, "sender_acct": "TR330006200119800006672318", "sender_bank": "ISBKTR"},
    {"id": "case_005", "desc": "Netflix subscription recurring payment", "tier": "low", "outcome": "approve", "tags": ["legitimate", "subscription"], "amount": 54.99, "currency": "TRY", "channel": "api", "merchant": "Netflix", "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000005", "ip": None, "sender_acct": "TR330006200119800006672319", "sender_bank": "AKBNKTR"},
    {"id": "case_006", "desc": "Gas station purchase, afternoon, mobile", "tier": "low", "outcome": "approve", "tags": ["legitimate", "gas_station"], "amount": 450.0, "currency": "TRY", "channel": "mobile", "merchant": "Opet Benzin", "merchant_id": "OPET-TUR-0321", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000006", "ip": "85.103.12.44", "sender_acct": "TR330006200119800006672320", "sender_bank": "GARBBTR"},
    {"id": "case_007", "desc": "Bookstore online purchase, business hours", "tier": "low", "outcome": "approve", "tags": ["legitimate", "retail"], "amount": 89.9, "currency": "TRY", "channel": "online", "merchant": "Kitap Yurdu", "merchant_id": "KITAPYURDU-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000007", "ip": "78.181.55.99", "sender_acct": "TR330006200119800006672321", "sender_bank": "YBASTRIS"},
    {"id": "case_008", "desc": "Gym membership monthly recurring", "tier": "low", "outcome": "approve", "tags": ["legitimate", "subscription"], "amount": 199.0, "currency": "TRY", "channel": "api", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000008", "ip": None, "sender_acct": "TR330006200119800006672322", "sender_bank": "AKBNKTR"},
    {"id": "case_009", "desc": "Clothing store purchase, weekend afternoon, online", "tier": "low", "outcome": "approve", "tags": ["legitimate", "retail"], "amount": 175.0, "currency": "TRY", "channel": "online", "merchant": "LCWaikiki", "merchant_id": "LCWAIKIKI-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000009", "ip": "81.214.88.22", "sender_acct": "TR330006200119800006672323", "sender_bank": "GARBBTR"},
    {"id": "case_010", "desc": "Electricity bill API payment, recurring", "tier": "low", "outcome": "approve", "tags": ["legitimate", "utility"], "amount": 380.0, "currency": "TRY", "channel": "api", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000010", "ip": None, "sender_acct": "TR330006200119800006672324", "sender_bank": "ISBKTR"},
    {"id": "case_011", "desc": "Small ATM withdrawal, domestic, noon", "tier": "low", "outcome": "approve", "tags": ["legitimate", "atm"], "amount": 200.0, "currency": "TRY", "channel": "atm", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000011", "ip": None, "sender_acct": "TR330006200119800006672325", "sender_bank": "AKBNKTR"},
    {"id": "case_012", "desc": "Restaurant dinner payment, early evening", "tier": "low", "outcome": "approve", "tags": ["legitimate", "restaurant"], "amount": 145.0, "currency": "TRY", "channel": "mobile", "merchant": "Burger King TR", "merchant_id": "BURGER-TR-0055", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000012", "ip": "88.226.71.44", "sender_acct": "TR330006200119800006672326", "sender_bank": "GARBBTR"},
    {"id": "case_013", "desc": "Domestic bank transfer, small amount, business hours", "tier": "low", "outcome": "approve", "tags": ["legitimate", "transfer"], "amount": 500.0, "currency": "TRY", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000013", "ip": "78.181.22.55", "sender_acct": "TR330006200119800006672327", "sender_bank": "YBASTRIS"},
    {"id": "case_014", "desc": "Insurance premium monthly payment, API", "tier": "low", "outcome": "approve", "tags": ["legitimate", "subscription"], "amount": 420.0, "currency": "TRY", "channel": "api", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000014", "ip": None, "sender_acct": "TR330006200119800006672328", "sender_bank": "AKBNKTR"},
    {"id": "case_015", "desc": "Cinema ticket purchase, weekend evening", "tier": "low", "outcome": "approve", "tags": ["legitimate", "entertainment"], "amount": 160.0, "currency": "TRY", "channel": "online", "merchant": "Cinemaximum", "merchant_id": "CINEMAXIMUM-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000015", "ip": "85.103.77.88", "sender_acct": "TR330006200119800006672329", "sender_bank": "GARBBTR"},
    {"id": "case_016", "desc": "Supermarket purchase, afternoon, mobile", "tier": "low", "outcome": "approve", "tags": ["legitimate", "grocery"], "amount": 312.5, "currency": "TRY", "channel": "mobile", "merchant": "CarrefourSA", "merchant_id": "CARREFOURSA-018", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000016", "ip": "88.226.44.11", "sender_acct": "TR330006200119800006672330", "sender_bank": "ISBKTR"},
    {"id": "case_017", "desc": "Spotify subscription monthly, API", "tier": "low", "outcome": "approve", "tags": ["legitimate", "subscription"], "amount": 39.99, "currency": "TRY", "channel": "api", "merchant": "Spotify", "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000017", "ip": None, "sender_acct": "TR330006200119800006672331", "sender_bank": "YBASTRIS"},
    {"id": "case_018", "desc": "Dental clinic payment, branch, daytime", "tier": "low", "outcome": "approve", "tags": ["legitimate", "medical"], "amount": 450.0, "currency": "TRY", "channel": "branch", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000018", "ip": None, "sender_acct": "TR330006200119800006672332", "sender_bank": "AKBNKTR"},
    {"id": "case_019", "desc": "Transport card top-up, mobile, morning", "tier": "low", "outcome": "approve", "tags": ["legitimate", "transport"], "amount": 50.0, "currency": "TRY", "channel": "mobile", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000019", "ip": "88.226.10.55", "sender_acct": "TR330006200119800006672333", "sender_bank": "GARBBTR"},
    {"id": "case_020", "desc": "School tuition fee payment, morning, online", "tier": "low", "outcome": "approve", "tags": ["legitimate", "education"], "amount": 750.0, "currency": "TRY", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000020", "ip": "78.181.33.77", "sender_acct": "TR330006200119800006672334", "sender_bank": "ISBKTR"},
    # MEDIUM — decline (case_021–030, fraud with raw_mode)
    {"id": "case_021", "desc": "High-amount night transfer to Netherlands, new device", "tier": "medium", "outcome": "decline", "tags": ["fraud", "geo_anomaly", "velocity"], "amount": 3500.0, "currency": "EUR", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "NL", "txn_id": "00000000-0000-0000-0000-000000000021", "ip": "45.32.100.11", "device": "NEW-DEVICE-XYZ-001", "sender_acct": "TR330006200119800006672335", "sender_bank": "AKBNKTR", "recv_bank": "ABNANL2A"},
    {"id": "case_022", "desc": "Large cash-equivalent withdrawal, deep night", "tier": "medium", "outcome": "decline", "tags": ["fraud", "velocity"], "amount": 4800.0, "currency": "TRY", "channel": "atm", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000022", "ip": None, "sender_acct": "TR330006200119800006672336", "sender_bank": "GARBBTR", "recv_bank": "GARBBTR"},
    {"id": "case_023", "desc": "Crypto merchant online purchase, late night, Tor-like IP", "tier": "medium", "outcome": "decline", "tags": ["fraud", "crypto_merchant"], "amount": 2800.0, "currency": "USD", "channel": "online", "merchant": "BitPay Crypto Exchange", "merchant_id": "BITPAY-001", "src_country": "TR", "dst_country": "US", "txn_id": "00000000-0000-0000-0000-000000000023", "ip": "185.220.101.5", "device": "ANON-BROWSER-TOR", "sender_acct": "TR330006200119800006672337", "sender_bank": "ISBKTR", "recv_bank": "USBKUS33"},
    {"id": "case_024", "desc": "Rooted device, late night transfer, velocity signal", "tier": "medium", "outcome": "decline", "tags": ["fraud", "velocity"], "amount": 1950.0, "currency": "TRY", "channel": "mobile", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000024", "ip": "46.196.88.100", "device": "ROOTED-ANDROID-001", "sender_acct": "TR330006200119800006672338", "sender_bank": "AKBNKTR", "recv_bank": "GARBBTR"},
    {"id": "case_025", "desc": "First-time international wire to UAE, high amount, night", "tier": "medium", "outcome": "decline", "tags": ["fraud", "geo_anomaly", "aml"], "amount": 5000.0, "currency": "USD", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "AE", "txn_id": "00000000-0000-0000-0000-000000000025", "ip": "91.93.209.11", "sender_acct": "TR330006200119800006672339", "sender_bank": "GARBBTR", "recv_bank": "EBILAEAD"},
    {"id": "case_026", "desc": "High-value electronics night purchase, VM fingerprint", "tier": "medium", "outcome": "decline", "tags": ["fraud", "velocity"], "amount": 4200.0, "currency": "TRY", "channel": "online", "merchant": "Amazon Turkiye Elektronik", "merchant_id": "AMAZON-TR-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000026", "ip": "5.189.200.44", "device": "VM-FINGERPRINT-001", "sender_acct": "TR330006200119800006672340", "sender_bank": "ISBKTR", "recv_bank": "AKBNKTR"},
    {"id": "case_027", "desc": "Foreign IP, headless browser, large UK purchase", "tier": "medium", "outcome": "decline", "tags": ["fraud", "geo_anomaly"], "amount": 3100.0, "currency": "USD", "channel": "online", "merchant": "PayPal UK Services", "merchant_id": "PAYPAL-UK-001", "src_country": "TR", "dst_country": "GB", "txn_id": "00000000-0000-0000-0000-000000000027", "ip": "194.165.16.100", "device": "HEADLESS-BROWSER-001", "sender_acct": "TR330006200119800006672341", "sender_bank": "AKBNKTR", "recv_bank": "NWBKGB2L"},
    {"id": "case_028", "desc": "Night purchase to Germany, emulated Android device", "tier": "medium", "outcome": "decline", "tags": ["fraud", "geo_anomaly"], "amount": 2650.0, "currency": "EUR", "channel": "online", "merchant": "OTTO Deutschland", "merchant_id": "OTTO-DE-001", "src_country": "TR", "dst_country": "DE", "txn_id": "00000000-0000-0000-0000-000000000028", "ip": "45.141.87.33", "device": "EMULATOR-ANDROID", "sender_acct": "TR330006200119800006672342", "sender_bank": "GARBBTR", "recv_bank": "COBADEFF"},
    {"id": "case_029", "desc": "Jailbroken iPhone, night, suspicious unknown merchant", "tier": "medium", "outcome": "decline", "tags": ["fraud", "velocity"], "amount": 1800.0, "currency": "TRY", "channel": "online", "merchant": "Misc Online Services XY9", "merchant_id": "UNKNOWN-MERCH-XY9", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000029", "ip": "77.223.99.44", "device": "JAILBROKEN-IOS-001", "sender_acct": "TR330006200119800006672343", "sender_bank": "ISBKTR", "recv_bank": "YBASTRIS"},
    {"id": "case_030", "desc": "High-amount payment to Russia, night, card-not-present", "tier": "medium", "outcome": "decline", "tags": ["fraud", "geo_anomaly", "aml"], "amount": 4600.0, "currency": "USD", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "RU", "txn_id": "00000000-0000-0000-0000-000000000030", "ip": "194.87.198.10", "sender_acct": "TR330006200119800006672344", "sender_bank": "AKBNKTR", "recv_bank": "SABRRUMM"},
    # MEDIUM — manual_review (case_031–040, case_046, case_049)
    {"id": "case_031", "desc": "Mildly elevated amount, known electronics merchant, evening", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "retail"], "amount": 1200.0, "currency": "TRY", "channel": "online", "merchant": "Teknosa Elektronik", "merchant_id": "TEKNOSA-TR-044", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000031", "ip": "78.181.99.11", "sender_acct": "TR330006200119800006672345", "sender_bank": "GARBBTR"},
    {"id": "case_032", "desc": "Moderate Booking.com purchase, known platform, daytime", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "travel"], "amount": 890.0, "currency": "EUR", "channel": "online", "merchant": "Booking.com", "merchant_id": "BOOKING-EU-001", "src_country": "TR", "dst_country": "DE", "txn_id": "00000000-0000-0000-0000-000000000032", "ip": "88.226.55.22", "sender_acct": "TR330006200119800006672346", "sender_bank": "ISBKTR"},
    {"id": "case_033", "desc": "Moderate domestic transfer, slightly late night", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "transfer"], "amount": 1500.0, "currency": "TRY", "channel": "mobile", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000033", "ip": "88.255.11.44", "sender_acct": "TR330006200119800006672347", "sender_bank": "AKBNKTR"},
    {"id": "case_034", "desc": "New device, moderate amount, known domestic retailer", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "retail"], "amount": 950.0, "currency": "TRY", "channel": "online", "merchant": "Trendyol", "merchant_id": "TRENDYOL-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000034", "ip": "78.181.44.66", "device": "NEW-IPHONE-16-001", "sender_acct": "TR330006200119800006672348", "sender_bank": "GARBBTR"},
    {"id": "case_035", "desc": "Hotel booking Istanbul, higher amount, afternoon", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "travel"], "amount": 2200.0, "currency": "TRY", "channel": "online", "merchant": "Hilton Istanbul", "merchant_id": "HILTON-IST-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000035", "ip": "88.226.88.33", "sender_acct": "TR330006200119800006672349", "sender_bank": "ISBKTR"},
    {"id": "case_036", "desc": "Cross-border small payment to US, known platform, daytime", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "transfer"], "amount": 120.0, "currency": "USD", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "US", "txn_id": "00000000-0000-0000-0000-000000000036", "ip": "85.103.66.77", "sender_acct": "TR330006200119800006672350", "sender_bank": "AKBNKTR"},
    {"id": "case_037", "desc": "Late evening domestic Hepsiburada purchase, slightly elevated", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "retail"], "amount": 1850.0, "currency": "TRY", "channel": "mobile", "merchant": "Hepsiburada", "merchant_id": "HEPSIBURADA-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000037", "ip": "88.226.33.55", "sender_acct": "TR330006200119800006672351", "sender_bank": "GARBBTR"},
    {"id": "case_038", "desc": "Weekend domestic transfer, moderate, same bank", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "transfer"], "amount": 2000.0, "currency": "TRY", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000038", "ip": "78.181.77.33", "sender_acct": "TR330006200119800006672352", "sender_bank": "ISBKTR"},
    {"id": "case_039", "desc": "Airline ticket THY, moderate amount, business hours", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "travel"], "amount": 1650.0, "currency": "TRY", "channel": "online", "merchant": "Turkish Airlines", "merchant_id": "THYAO-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000039", "ip": "88.226.22.44", "sender_acct": "TR330006200119800006672353", "sender_bank": "AKBNKTR"},
    {"id": "case_040", "desc": "Home appliance purchase, elevated amount, known retailer, daytime", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "retail"], "amount": 3800.0, "currency": "TRY", "channel": "online", "merchant": "Arcelik", "merchant_id": "ARCELIK-TR-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000040", "ip": "85.103.55.88", "sender_acct": "TR330006200119800006672354", "sender_bank": "GARBBTR"},
    # HIGH — escalate (case_041–045)
    {"id": "case_041", "desc": "Very high offshore wire, BVI, deep night, Tor exit IP", "tier": "high", "outcome": "escalate", "tags": ["aml", "structuring", "sanctions"], "amount": 48000.0, "currency": "USD", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "VG", "txn_id": "00000000-0000-0000-0000-000000000041", "ip": "103.28.54.200", "device": "TOR-EXIT-NODE-001", "timestamp": "2025-03-15T03:22:00+03:00", "sender_acct": "TR330006200119800006672355", "sender_bank": "AKBNKTR", "recv_bank": "BVIBBVIV"},
    {"id": "case_042", "desc": "Structuring: just below 30k TRY threshold, late night automation", "tier": "high", "outcome": "escalate", "tags": ["aml", "structuring"], "amount": 29500.0, "currency": "TRY", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000042", "ip": "77.79.99.1", "device": "AUTOMATION-BOT-001", "timestamp": "2025-03-15T02:55:00+03:00", "sender_acct": "TR330006200119800006672356", "sender_bank": "GARBBTR", "recv_bank": "ISBKTR"},
    {"id": "case_043", "desc": "Large wire to sanctioned country Iran, VPN, night", "tier": "high", "outcome": "escalate", "tags": ["aml", "sanctions", "geo_anomaly"], "amount": 35000.0, "currency": "USD", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "IR", "txn_id": "00000000-0000-0000-0000-000000000043", "ip": "185.220.101.34", "device": "ANON-VPN-001", "timestamp": "2025-03-15T04:10:00+03:00", "sender_acct": "TR330006200119800006672357", "sender_bank": "ISBKTR", "recv_bank": "BKMIIRTH"},
    {"id": "case_044", "desc": "Crypto mixer large purchase, deep night, anon device", "tier": "high", "outcome": "escalate", "tags": ["aml", "crypto_merchant", "sanctions"], "amount": 22000.0, "currency": "USD", "channel": "online", "merchant": "CryptoMixer Pro", "merchant_id": "CRYPTOMIXER-001", "src_country": "TR", "dst_country": "US", "txn_id": "00000000-0000-0000-0000-000000000044", "ip": "104.244.72.115", "device": "ANON-BROWSER-001", "timestamp": "2025-03-15T01:45:00+03:00", "sender_acct": "TR330006200119800006672358", "sender_bank": "AKBNKTR", "recv_bank": "CRYPTOEX1"},
    {"id": "case_045", "desc": "Shell company layered wire to North Korea, server-side script", "tier": "high", "outcome": "escalate", "tags": ["aml", "sanctions", "structuring"], "amount": 75000.0, "currency": "USD", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "KP", "txn_id": "00000000-0000-0000-0000-000000000045", "ip": "45.76.119.200", "device": "SERVER-SIDE-SCRIPT", "timestamp": "2025-03-15T03:35:00+03:00", "sender_acct": "TR330006200119800006672359", "sender_bank": "GARBBTR", "recv_bank": "KCBKKPPY"},
    # MEDIUM — manual_review (continued)
    {"id": "case_046", "desc": "Slightly unusual hour domestic ATM, moderate amount", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "atm"], "amount": 800.0, "currency": "TRY", "channel": "atm", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000046", "ip": None, "sender_acct": "TR330006200119800006672360", "sender_bank": "ISBKTR"},
    # LOW — approve (continued)
    {"id": "case_047", "desc": "Small cross-border charity donation, US, daytime", "tier": "low", "outcome": "approve", "tags": ["legitimate", "transfer"], "amount": 50.0, "currency": "USD", "channel": "online", "merchant": None, "merchant_id": None, "src_country": "TR", "dst_country": "US", "txn_id": "00000000-0000-0000-0000-000000000047", "ip": "88.226.55.33", "sender_acct": "TR330006200119800006672361", "sender_bank": "AKBNKTR"},
    {"id": "case_048", "desc": "Regular Getir delivery app purchase, morning", "tier": "low", "outcome": "approve", "tags": ["legitimate", "grocery"], "amount": 95.0, "currency": "TRY", "channel": "mobile", "merchant": "Getir Turkiye", "merchant_id": "GETIR-TR-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000048", "ip": "88.226.77.44", "sender_acct": "TR330006200119800006672362", "sender_bank": "GARBBTR"},
    # MEDIUM — manual_review (continued)
    {"id": "case_049", "desc": "Mid-range n11.com purchase, late evening", "tier": "medium", "outcome": "manual_review", "tags": ["legitimate", "retail"], "amount": 680.0, "currency": "TRY", "channel": "online", "merchant": "n11.com", "merchant_id": "N11-TR-001", "src_country": "TR", "dst_country": "TR", "txn_id": "00000000-0000-0000-0000-000000000049", "ip": "78.181.88.55", "sender_acct": "TR330006200119800006672363", "sender_bank": "ISBKTR"},
]


# ---------------------------------------------------------------------------
# Build FIXTURES
# ---------------------------------------------------------------------------

_BUILDERS = {
    ("low", "approve"): _low,
    ("medium", "decline"): _med_decline,
    ("medium", "manual_review"): _med_review,
    ("high", "escalate"): _high,
}


def _build(m: dict) -> dict:
    key = (m["tier"], m["outcome"])
    return _BUILDERS[key](m)


FIXTURES: list[dict] = [_build(m) for m in _META]
