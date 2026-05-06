"""Pre-computed pipeline results for the HF Spaces demo.

Each entry is a view_entry dict matching the structure produced by the live
FraudLens API, so all existing render_* functions in app.py work without
modification.

Five cases cover every pipeline path:
  case_001 — LOW  / auto-approve   (no agents)
  case_038 — MED  / manual_review  (Investigation Agent)
  case_040 — MED  / decline        (Investigation Agent, suspicious)
  case_042 — HIGH / escalate       (Critical Agent, structuring + SAR)
  case_043 — HIGH / escalate       (Critical Agent, sanctions + SAR)
"""

from __future__ import annotations

FIXTURES: list[dict] = [
    # ------------------------------------------------------------------
    # case_001 — LOW / APPROVE
    # ------------------------------------------------------------------
    {
        "case": {
            "case_id": "case_001",
            "description": "Small grocery purchase, domestic, daytime",
            "tier": "low",
            "tags": ["legitimate", "grocery"],
            "expected_outcome": "approve",
            "transaction": {
                "transaction_id": "00000000-0000-0000-0000-000000000001",
                "amount": 47.5,
                "currency": "TRY",
                "channel": "mobile",
                "merchant_name": "Migros Supermarket",
                "sender_country": "TR",
                "receiver_country": "TR",
            },
        },
        "result": {
            "transaction_id": "00000000-0000-0000-0000-000000000001",
            "fraud_probability": 0.0412,
            "risk_tier": "low",
            "triage_action": "approve",
            "shap_top_features": [
                {"feature": "C1", "contribution": -0.612},
                {"feature": "card1", "contribution": -0.389},
                {"feature": "TransactionAmt", "contribution": -0.281},
                {"feature": "D1", "contribution": -0.174},
                {"feature": "V45", "contribution": -0.098},
            ],
            "processing_time_ms": 1243.5,
            "investigation": None,
            "fraud_decision": None,
            "sar_report": None,
            "token_usage": None,
        },
        "elapsed_ms": 1380.0,
    },

    # ------------------------------------------------------------------
    # case_038 — MEDIUM / MANUAL REVIEW
    # ------------------------------------------------------------------
    {
        "case": {
            "case_id": "case_038",
            "description": "Weekend domestic transfer, moderate, same bank",
            "tier": "medium",
            "tags": ["legitimate", "transfer"],
            "expected_outcome": "manual_review",
            "transaction": {
                "transaction_id": "00000000-0000-0000-0000-000000000038",
                "amount": 2000.0,
                "currency": "TRY",
                "channel": "online",
                "sender_country": "TR",
                "receiver_country": "TR",
            },
        },
        "result": {
            "transaction_id": "00000000-0000-0000-0000-000000000038",
            "fraud_probability": 0.4812,
            "risk_tier": "medium",
            "triage_action": "investigate",
            "shap_top_features": [
                {"feature": "C1", "contribution": 0.612},
                {"feature": "is_night", "contribution": 0.445},
                {"feature": "TransactionAmt", "contribution": 0.318},
                {"feature": "D1", "contribution": -0.221},
                {"feature": "card1", "contribution": -0.187},
            ],
            "processing_time_ms": 18432.5,
            "investigation": {
                "decision_hint": "inconclusive",
                "confidence": 0.52,
                "evidence": [
                    "Customer history: 18 transactions in 30 days — normal velocity for this account",
                    "Transfer amount 2,000 TRY is 1.3× the customer average (1,540 TRY)",
                    "Same-bank transfer (ISBKTR → ISBKTR) reduces layering risk significantly",
                    "Late-night submission (22:00) adds minor suspicion but within observed patterns",
                    "IP geolocation: Istanbul domestic, no proxy or VPN detected",
                ],
                "red_flags": ["night_transaction", "elevated_amount"],
                "tools_called": ["get_customer_history", "get_geolocation_context"],
                "reasoning_summary": (
                    "Medium-risk domestic transfer shows mildly elevated amount and late-night timing "
                    "but same-bank routing and normal customer velocity suggest legitimate activity. "
                    "Conflicting signals yield an inconclusive verdict; routed to manual review."
                ),
                "tool_trace": [
                    {
                        "tool": "get_customer_history",
                        "args": {"account_id": "TR330006200119800006672352"},
                        "result": '{"transactions_30d": 18, "avg_amount": 1540, "account_age_days": 412, "prior_flags": 0}',
                    },
                    {
                        "tool": "get_geolocation_context",
                        "args": {"ip_address": "78.181.77.33"},
                        "result": '{"city": "Istanbul", "country": "TR", "is_proxy": false, "isp": "Turk Telekom"}',
                    },
                ],
                "cited_sources": [],
            },
            "fraud_decision": {
                "transaction_id": "00000000-0000-0000-0000-000000000038",
                "outcome": "manual_review",
                "confidence": 0.52,
                "ml_score": 0.4812,
                "agent_used": "investigation",
                "decision_hint": "inconclusive",
                "evidence": [
                    "Customer history: 18 transactions in 30 days — normal velocity for this account",
                    "Transfer amount 2,000 TRY is 1.3× the customer average (1,540 TRY)",
                    "Same-bank transfer (ISBKTR → ISBKTR) reduces layering risk significantly",
                    "Late-night submission (22:00) adds minor suspicion but within observed patterns",
                    "IP geolocation: Istanbul domestic, no proxy or VPN detected",
                ],
                "red_flags": ["night_transaction", "elevated_amount"],
                "regulatory_citations": [],
                "reasoning": (
                    "Medium-risk domestic transfer shows mildly elevated amount and late-night timing "
                    "but same-bank routing and normal customer velocity suggest legitimate activity. "
                    "Conflicting signals yield an inconclusive verdict; routed to manual review."
                ),
                "tools_called": ["get_customer_history", "get_geolocation_context"],
            },
            "sar_report": None,
            "token_usage": {"input_tokens": 4820, "output_tokens": 612},
        },
        "elapsed_ms": 19850.0,
    },

    # ------------------------------------------------------------------
    # case_040 — MEDIUM / DECLINE
    # ------------------------------------------------------------------
    {
        "case": {
            "case_id": "case_040",
            "description": "Home appliance purchase, elevated amount, known retailer",
            "tier": "medium",
            "tags": ["legitimate", "retail"],
            "expected_outcome": "decline",
            "transaction": {
                "transaction_id": "00000000-0000-0000-0000-000000000040",
                "amount": 3800.0,
                "currency": "TRY",
                "channel": "online",
                "merchant_name": "Arcelik",
                "sender_country": "TR",
                "receiver_country": "TR",
            },
        },
        "result": {
            "transaction_id": "00000000-0000-0000-0000-000000000040",
            "fraud_probability": 0.6218,
            "risk_tier": "medium",
            "triage_action": "investigate",
            "shap_top_features": [
                {"feature": "C1", "contribution": 0.894},
                {"feature": "V45", "contribution": 0.621},
                {"feature": "TransactionAmt", "contribution": 0.512},
                {"feature": "card1", "contribution": 0.341},
                {"feature": "D1", "contribution": -0.198},
                {"feature": "C5", "contribution": 0.187},
            ],
            "processing_time_ms": 24180.0,
            "investigation": {
                "decision_hint": "suspicious",
                "confidence": 0.81,
                "evidence": [
                    "Customer history: 44 transactions in 30 days — elevated velocity (2× baseline)",
                    "Account age only 38 days — new account with high transaction volume is a red flag",
                    "3,800 TRY is 4.2× customer average transaction amount (905 TRY)",
                    "Merchant ARCELIK-TR-001 reputation: clean (risk score 0.12, 0 disputes in 90d)",
                    "IP geolocation: Ankara domestic, clean ISP — no anonymisation detected",
                    "2 prior suspicious flags on this account within 15 days",
                ],
                "red_flags": ["high_velocity", "new_account_high_spend", "prior_suspicious_flags", "elevated_amount"],
                "tools_called": ["get_customer_history", "check_merchant_reputation", "get_geolocation_context"],
                "reasoning_summary": (
                    "New account (38 days) with 2× velocity baseline and 4× amount anomaly "
                    "and 2 prior suspicious flags presents a clear card-testing or account-takeover profile. "
                    "Merchant is clean so the risk is account-side. High confidence suspicious verdict."
                ),
                "tool_trace": [
                    {
                        "tool": "get_customer_history",
                        "args": {"account_id": "TR330006200119800006672354"},
                        "result": '{"transactions_30d": 44, "avg_amount": 905, "account_age_days": 38, "prior_flags": 2}',
                    },
                    {
                        "tool": "check_merchant_reputation",
                        "args": {"merchant_id": "ARCELIK-TR-001"},
                        "result": '{"risk_score": 0.12, "disputes_90d": 0, "category": "retail_electronics"}',
                    },
                    {
                        "tool": "get_geolocation_context",
                        "args": {"ip_address": "85.103.55.88"},
                        "result": '{"city": "Ankara", "country": "TR", "is_proxy": false, "isp": "Superonline"}',
                    },
                ],
                "cited_sources": [],
            },
            "fraud_decision": {
                "transaction_id": "00000000-0000-0000-0000-000000000040",
                "outcome": "decline",
                "confidence": 0.81,
                "ml_score": 0.6218,
                "agent_used": "investigation",
                "decision_hint": "suspicious",
                "evidence": [
                    "Customer history: 44 transactions in 30 days — elevated velocity (2× baseline)",
                    "Account age only 38 days — new account with high transaction volume is a red flag",
                    "3,800 TRY is 4.2× customer average transaction amount (905 TRY)",
                    "Merchant ARCELIK-TR-001 reputation: clean (risk score 0.12, 0 disputes in 90d)",
                    "IP geolocation: Ankara domestic, clean ISP — no anonymisation detected",
                    "2 prior suspicious flags on this account within 15 days",
                ],
                "red_flags": ["high_velocity", "new_account_high_spend", "prior_suspicious_flags", "elevated_amount"],
                "regulatory_citations": [],
                "reasoning": (
                    "New account (38 days) with 2× velocity baseline and 4× amount anomaly "
                    "and 2 prior suspicious flags presents a clear card-testing or account-takeover profile. "
                    "Merchant is clean so the risk is account-side. High confidence suspicious verdict."
                ),
                "tools_called": ["get_customer_history", "check_merchant_reputation", "get_geolocation_context"],
            },
            "sar_report": None,
            "token_usage": {"input_tokens": 6210, "output_tokens": 748},
        },
        "elapsed_ms": 25600.0,
    },

    # ------------------------------------------------------------------
    # case_042 — HIGH / ESCALATE — structuring
    # ------------------------------------------------------------------
    {
        "case": {
            "case_id": "case_042",
            "description": "Structuring: just below 30k TRY threshold, late night automation",
            "tier": "high",
            "tags": ["aml", "structuring"],
            "expected_outcome": "escalate",
            "transaction": {
                "transaction_id": "00000000-0000-0000-0000-000000000042",
                "amount": 29500.0,
                "currency": "TRY",
                "channel": "online",
                "sender_country": "TR",
                "receiver_country": "TR",
            },
        },
        "result": {
            "transaction_id": "00000000-0000-0000-0000-000000000042",
            "fraud_probability": 0.7329,
            "risk_tier": "high",
            "triage_action": "escalate",
            "shap_top_features": [
                {"feature": "C1", "contribution": 2.1077},
                {"feature": "V45", "contribution": 1.0826},
                {"feature": "V307", "contribution": 0.4214},
                {"feature": "C5", "contribution": 0.3637},
                {"feature": "V314", "contribution": -0.3468},
                {"feature": "card6", "contribution": -0.2634},
                {"feature": "D15", "contribution": 0.2256},
            ],
            "processing_time_ms": 87430.0,
            "investigation": {
                "decision_hint": "suspicious",
                "confidence": 0.88,
                "evidence": [
                    "XGBoost fraud probability: 0.7329 (73.29%) with agent confidence 0.88",
                    "Amount 29,500 TRY is 500 TRY below the MASAK 30,000 TRY mandatory reporting threshold — classic structuring pattern",
                    "Transaction submitted at 02:55 via automation bot device fingerprint (AUTOMATION-BOT-001)",
                    "Customer history: 44 transactions in 30 days (high velocity) on a 254-day account",
                    "2 prior suspicious flags on account TR330006200119800006672356",
                    "Circular fund flow detected: 6-node network, density 0.2 — hallmark of layering technique",
                    "Adverse media: no direct sanctions match, but associated network node flagged in FATF mutual evaluation database",
                    "Historical pattern matching: 2 prior structuring cases with identical typology (top match 0.72)",
                ],
                "red_flags": [
                    "structuring_pattern",
                    "threshold_avoidance",
                    "automation_bot_device",
                    "circular_fund_flow",
                    "prior_suspicious_flags",
                    "high_velocity",
                    "night_transaction",
                ],
                "tools_called": [
                    "explain_ml_score",
                    "get_customer_history",
                    "adverse_media_search",
                    "deep_network_analysis",
                    "regulatory_policy_rag",
                    "check_merchant_reputation",
                    "find_similar_patterns",
                ],
                "reasoning_summary": (
                    "Transaction exhibits threshold-avoidance structuring: 29,500 TRY positioned 500 TRY "
                    "below the MASAK mandatory STR threshold, submitted at 02:55 via automation bot. "
                    "Deep network analysis confirms circular fund flow across a 6-node graph (density 0.2), "
                    "consistent with layering. MASAK Article 4 (Law 5549) and FATF Recommendation 13 "
                    "mandate immediate STR filing. Account shows 2 prior flags and high velocity. "
                    "Convergence of structuring signal, automation, circular flow, and prior history "
                    "yields high-confidence suspicious verdict."
                ),
                "tool_trace": [
                    {
                        "tool": "explain_ml_score",
                        "args": {},
                        "result": '{"top_features": [{"feature": "C1", "shap": 2.1077, "meaning": "elevated address count"}, {"feature": "V45", "shap": 1.0826, "meaning": "velocity anomaly"}]}',
                    },
                    {
                        "tool": "get_customer_history",
                        "args": {"account_id": "TR330006200119800006672356"},
                        "result": '{"transactions_30d": 44, "avg_amount": 8200, "account_age_days": 254, "prior_flags": 2}',
                    },
                    {
                        "tool": "adverse_media_search",
                        "args": {"account_id": "TR330006200119800006672356"},
                        "result": '{"sanctions_match": false, "pep_flag": false, "media_hits": 0, "network_flag": true}',
                    },
                    {
                        "tool": "deep_network_analysis",
                        "args": {"transaction_id": "00000000-0000-0000-0000-000000000042"},
                        "result": '{"nodes": 6, "edges": 6, "density": 0.2, "circular_flow": true, "pattern": "layering"}',
                    },
                    {
                        "tool": "regulatory_policy_rag",
                        "args": {"query": "structuring threshold avoidance STR reporting obligation"},
                        "result": '{"excerpts": [{"text": "Financial institutions shall report to MASAK any transaction that appears designed to evade reporting thresholds. Threshold structuring is a predicate indicator requiring immediate STR filing.", "citation": "MASAK_5549_Regulation.pdf, p.12", "source": "MASAK_5549_Regulation.pdf", "page": 12, "relevance_score": 0.81}, {"text": "Countries should require financial institutions to report suspicious transactions to the FIU when they suspect that funds are the proceeds of a criminal activity.", "citation": "FATF_40_RECOMMENDATIONS_2012.pdf, p.13", "source": "FATF_40_RECOMMENDATIONS_2012.pdf", "page": 13, "relevance_score": 0.74}]}',
                    },
                    {
                        "tool": "find_similar_patterns",
                        "args": {"transaction_id": "00000000-0000-0000-0000-000000000042"},
                        "result": '{"similar_cases": 2, "top_match_score": 0.72, "typologies": ["structuring", "automation_bot"]}',
                    },
                ],
                "cited_sources": [
                    "MASAK_5549_Regulation.pdf, p.12",
                    "FATF_40_RECOMMENDATIONS_2012.pdf, p.13",
                ],
            },
            "fraud_decision": {
                "transaction_id": "00000000-0000-0000-0000-000000000042",
                "outcome": "escalate",
                "confidence": 0.88,
                "ml_score": 0.7329,
                "agent_used": "critical",
                "decision_hint": "suspicious",
                "evidence": [
                    "XGBoost fraud probability: 0.7329 (73.29%) with agent confidence 0.88",
                    "Amount 29,500 TRY is 500 TRY below the MASAK 30,000 TRY mandatory reporting threshold — classic structuring pattern",
                    "Transaction submitted at 02:55 via automation bot device fingerprint (AUTOMATION-BOT-001)",
                    "Customer history: 44 transactions in 30 days (high velocity) on a 254-day account",
                    "2 prior suspicious flags on account TR330006200119800006672356",
                    "Circular fund flow detected: 6-node network, density 0.2 — hallmark of layering technique",
                    "Adverse media: no direct sanctions match, but associated network node flagged in FATF mutual evaluation database",
                    "Historical pattern matching: 2 prior structuring cases with identical typology (top match 0.72)",
                ],
                "red_flags": [
                    "structuring_pattern",
                    "threshold_avoidance",
                    "automation_bot_device",
                    "circular_fund_flow",
                    "prior_suspicious_flags",
                    "high_velocity",
                    "night_transaction",
                ],
                "regulatory_citations": [
                    {
                        "source": "MASAK_5549_Regulation.pdf",
                        "article": None,
                        "page": 12,
                        "excerpt": "Financial institutions shall report to MASAK any transaction that appears designed to evade reporting thresholds. Threshold structuring is a predicate indicator requiring immediate STR filing.",
                        "relevance_score": 0.81,
                    },
                    {
                        "source": "FATF_40_RECOMMENDATIONS_2012.pdf",
                        "article": None,
                        "page": 13,
                        "excerpt": "Countries should require financial institutions to report suspicious transactions to the FIU when they suspect that funds are the proceeds of a criminal activity or terrorism financing.",
                        "relevance_score": 0.74,
                    },
                ],
                "reasoning": (
                    "Transaction exhibits threshold-avoidance structuring: 29,500 TRY positioned 500 TRY "
                    "below the MASAK mandatory STR threshold, submitted at 02:55 via automation bot. "
                    "Deep network analysis confirms circular fund flow across a 6-node graph (density 0.2), "
                    "consistent with layering. MASAK Article 4 (Law 5549) and FATF Recommendation 13 "
                    "mandate immediate STR filing. Account shows 2 prior flags and high velocity. "
                    "Convergence of structuring signal, automation, circular flow, and prior history "
                    "yields high-confidence suspicious verdict."
                ),
                "tools_called": [
                    "explain_ml_score",
                    "get_customer_history",
                    "adverse_media_search",
                    "deep_network_analysis",
                    "regulatory_policy_rag",
                    "find_similar_patterns",
                ],
            },
            "sar_report": {
                "transaction_id": "00000000-0000-0000-0000-000000000042",
                "customer_info": {
                    "account_id": "TR330006200119800006672356",
                    "bank_code": "GARBBTR",
                    "country": "TR",
                    "account_age_days": 254,
                    "prior_suspicious_flags": 2,
                    "kyc_risk_tier": "high",
                },
                "transaction_details": {
                    "amount": "29,500 TRY",
                    "channel": "online",
                    "timestamp": "2025-03-15T02:55:00+03:00",
                    "device_fingerprint": "AUTOMATION-BOT-001",
                    "ip_address": "77.79.99.1",
                    "receiver_bank": "ISBKTR",
                    "receiver_country": "TR",
                },
                "suspicious_indicators": [
                    "Amount 500 TRY below MASAK 30,000 TRY mandatory reporting threshold — deliberate structuring",
                    "Automated bot device fingerprint (AUTOMATION-BOT-001) confirms script-driven transaction",
                    "Deep night submission (02:55) with no customer interaction pattern",
                    "Circular fund flow: 6-node network graph with density 0.2 indicative of layering",
                    "2 prior suspicious flags on the sending account within the past 90 days",
                    "44 transactions in 30 days — 2.4× baseline velocity for account age",
                    "Pattern match: 2 prior structuring cases with identical typology (similarity 0.72)",
                ],
                "investigation_summary": (
                    "The investigation into Transaction 00000000-0000-0000-0000-000000000042 reveals "
                    "a convergence of structuring, automation, and network-layering signals. XGBoost "
                    "assigned fraud probability 0.7329. The amount (29,500 TRY) is 500 TRY below the "
                    "MASAK mandatory STR threshold — a threshold-avoidance pattern confirmed by automation "
                    "bot device fingerprint. Deep network analysis identified a 6-node circular flow "
                    "(density 0.2), consistent with money-laundering layering. Account has 2 prior "
                    "flags and 44 transactions in 30 days. Historical pattern matching found 2 identical "
                    "prior cases. MASAK Article 4 (Law 5549) and FATF Recommendation 13 mandate immediate STR."
                ),
                "regulatory_triggers": [
                    "MASAK Article 4, Law No. 5549 — STR filing obligation for threshold-avoidance structuring",
                    "FATF Recommendation 13 — Suspicious Transaction Reporting",
                    "FATF Recommendation 11 — Unusual Transaction Patterns",
                    "BDDK Circular on AML/CFT Controls (2021/12)",
                ],
                "recommended_action": (
                    "1. File Şüpheli İşlem Bildirimi (ŞİB) with MASAK immediately under Law 5549 Art. 4. "
                    "2. Place transaction hold and freeze outbound transfers on account TR330006200119800006672356. "
                    "3. Expand circular-flow analysis to map all 6 network nodes and identify connected accounts. "
                    "4. Review the 2 prior structuring cases (pattern match 0.72) for coordinated activity. "
                    "5. Initiate Enhanced Due Diligence (EDD): contact account holder and verify transaction purpose. "
                    "6. Escalate to senior compliance officer within 24 hours per BDDK procedure."
                ),
                "generated_at": "2025-03-15T03:12:00+00:00",
                "agent_model": "claude-haiku-4-5-20251001",
            },
            "token_usage": {"input_tokens": 18240, "output_tokens": 2810},
        },
        "elapsed_ms": 91200.0,
    },

    # ------------------------------------------------------------------
    # case_043 — HIGH / ESCALATE — sanctioned country
    # ------------------------------------------------------------------
    {
        "case": {
            "case_id": "case_043",
            "description": "Large wire to sanctioned country Iran, VPN, night",
            "tier": "high",
            "tags": ["aml", "sanctions", "geo_anomaly"],
            "expected_outcome": "escalate",
            "transaction": {
                "transaction_id": "00000000-0000-0000-0000-000000000043",
                "amount": 35000.0,
                "currency": "USD",
                "channel": "online",
                "sender_country": "TR",
                "receiver_country": "IR",
            },
        },
        "result": {
            "transaction_id": "00000000-0000-0000-0000-000000000043",
            "fraud_probability": 0.8912,
            "risk_tier": "high",
            "triage_action": "escalate",
            "shap_top_features": [
                {"feature": "V45", "contribution": 2.8341},
                {"feature": "C1", "contribution": 2.2018},
                {"feature": "card1", "contribution": 1.4122},
                {"feature": "TransactionAmt", "contribution": 0.9845},
                {"feature": "D1", "contribution": 0.6231},
                {"feature": "C5", "contribution": 0.4188},
                {"feature": "V307", "contribution": -0.2341},
            ],
            "processing_time_ms": 112650.0,
            "investigation": {
                "decision_hint": "suspicious",
                "confidence": 0.95,
                "evidence": [
                    "XGBoost fraud probability: 0.8912 — highest-confidence ML score in dataset",
                    "Wire transfer destination: Iran (IR) — FATF-listed jurisdiction with deficient AML/CFT controls",
                    "Receiver bank: BKMIIRTH — Iranian institution subject to international sanctions",
                    "IP address 185.220.101.34 identified as anonymous VPN exit node (AS208091)",
                    "Device fingerprint ANON-VPN-001 confirms deliberate anonymisation",
                    "Transfer amount 35,000 USD — significantly above MASAK reporting threshold",
                    "Deep night submission (04:10) with no observed legitimate business context",
                    "Adverse media: Iranian counterparty flagged in OFAC SDN-adjacent database",
                    "Customer history: first-ever cross-border transfer on this account; 0 prior international transactions",
                ],
                "red_flags": [
                    "sanctions_jurisdiction",
                    "sanctioned_counterparty_bank",
                    "vpn_anonymisation",
                    "large_cross_border_wire",
                    "first_international_transfer",
                    "adverse_media_match",
                    "night_transaction",
                ],
                "tools_called": [
                    "explain_ml_score",
                    "get_customer_history",
                    "adverse_media_search",
                    "deep_network_analysis",
                    "regulatory_policy_rag",
                    "get_geolocation_context",
                ],
                "reasoning_summary": (
                    "35,000 USD wire to Iran via anonymous VPN from an account with zero prior "
                    "international transaction history represents the clearest possible AML signal. "
                    "Destination jurisdiction (IR) is FATF-listed; receiver bank (BKMIIRTH) is under "
                    "international sanctions. Adverse media search flagged the Iranian counterparty. "
                    "FATF Recommendation 19 (high-risk jurisdictions) and OFAC/MASAK regulations "
                    "mandate immediate blocking and STR filing. Confidence: 0.95."
                ),
                "tool_trace": [
                    {
                        "tool": "explain_ml_score",
                        "args": {},
                        "result": '{"top_features": [{"feature": "V45", "shap": 2.83, "meaning": "cross-border anomaly"}, {"feature": "C1", "shap": 2.20, "meaning": "address count spike"}]}',
                    },
                    {
                        "tool": "get_customer_history",
                        "args": {"account_id": "TR330006200119800006672357"},
                        "result": '{"transactions_30d": 8, "avg_amount": 1200, "account_age_days": 620, "prior_flags": 0, "international_transfers": 0}',
                    },
                    {
                        "tool": "adverse_media_search",
                        "args": {"entity": "BKMIIRTH Iran"},
                        "result": '{"sanctions_match": true, "pep_flag": false, "sdn_adjacent": true, "ofac_listed": true}',
                    },
                    {
                        "tool": "deep_network_analysis",
                        "args": {"transaction_id": "00000000-0000-0000-0000-000000000043"},
                        "result": '{"nodes": 2, "edges": 1, "circular_flow": false, "cross_border_risk": "critical"}',
                    },
                    {
                        "tool": "regulatory_policy_rag",
                        "args": {"query": "high-risk jurisdiction sanctions Iran wire transfer reporting"},
                        "result": '{"excerpts": [{"text": "Countries should apply enhanced due diligence measures to business relationships and transactions with persons from countries for which this is called for by the FATF.", "citation": "FATF_40_RECOMMENDATIONS_2012.pdf, p.19", "source": "FATF_40_RECOMMENDATIONS_2012.pdf", "page": 19, "relevance_score": 0.88}, {"text": "Financial institutions shall immediately block and report transactions involving sanctioned jurisdictions or entities listed in international sanctions databases.", "citation": "MASAK_5549_Regulation.pdf, p.8", "source": "MASAK_5549_Regulation.pdf", "page": 8, "relevance_score": 0.82}]}',
                    },
                    {
                        "tool": "get_geolocation_context",
                        "args": {"ip_address": "185.220.101.34"},
                        "result": '{"city": "unknown", "country": "NL", "is_proxy": true, "is_vpn": true, "isp": "AS208091 Tor/VPN", "risk_score": 0.97}',
                    },
                ],
                "cited_sources": [
                    "FATF_40_RECOMMENDATIONS_2012.pdf, p.19",
                    "MASAK_5549_Regulation.pdf, p.8",
                ],
            },
            "fraud_decision": {
                "transaction_id": "00000000-0000-0000-0000-000000000043",
                "outcome": "escalate",
                "confidence": 0.95,
                "ml_score": 0.8912,
                "agent_used": "critical",
                "decision_hint": "suspicious",
                "evidence": [
                    "XGBoost fraud probability: 0.8912 — highest-confidence ML score in dataset",
                    "Wire transfer destination: Iran (IR) — FATF-listed jurisdiction with deficient AML/CFT controls",
                    "Receiver bank: BKMIIRTH — Iranian institution subject to international sanctions",
                    "IP address 185.220.101.34 identified as anonymous VPN exit node (AS208091)",
                    "Device fingerprint ANON-VPN-001 confirms deliberate anonymisation",
                    "Transfer amount 35,000 USD — significantly above MASAK reporting threshold",
                    "Deep night submission (04:10) with no observed legitimate business context",
                    "Adverse media: Iranian counterparty flagged in OFAC SDN-adjacent database",
                    "Customer history: first-ever cross-border transfer on this account",
                ],
                "red_flags": [
                    "sanctions_jurisdiction",
                    "sanctioned_counterparty_bank",
                    "vpn_anonymisation",
                    "large_cross_border_wire",
                    "first_international_transfer",
                    "adverse_media_match",
                    "night_transaction",
                ],
                "regulatory_citations": [
                    {
                        "source": "FATF_40_RECOMMENDATIONS_2012.pdf",
                        "article": None,
                        "page": 19,
                        "excerpt": "Countries should apply enhanced due diligence measures to business relationships and transactions with persons from countries for which this is called for by the FATF.",
                        "relevance_score": 0.88,
                    },
                    {
                        "source": "MASAK_5549_Regulation.pdf",
                        "article": None,
                        "page": 8,
                        "excerpt": "Financial institutions shall immediately block and report transactions involving sanctioned jurisdictions or entities listed in international sanctions databases.",
                        "relevance_score": 0.82,
                    },
                ],
                "reasoning": (
                    "35,000 USD wire to Iran via anonymous VPN from an account with zero prior "
                    "international transaction history represents the clearest possible AML signal. "
                    "Destination jurisdiction (IR) is FATF-listed; receiver bank (BKMIIRTH) is under "
                    "international sanctions. Adverse media search flagged the Iranian counterparty. "
                    "FATF Recommendation 19 and MASAK regulations mandate immediate blocking and STR filing."
                ),
                "tools_called": [
                    "explain_ml_score",
                    "get_customer_history",
                    "adverse_media_search",
                    "deep_network_analysis",
                    "regulatory_policy_rag",
                    "get_geolocation_context",
                ],
            },
            "sar_report": {
                "transaction_id": "00000000-0000-0000-0000-000000000043",
                "customer_info": {
                    "account_id": "TR330006200119800006672357",
                    "bank_code": "ISBKTR",
                    "country": "TR",
                    "account_age_days": 620,
                    "prior_suspicious_flags": 0,
                    "kyc_risk_tier": "critical",
                    "note": "Zero prior international transfers — first-ever cross-border wire",
                },
                "transaction_details": {
                    "amount": "35,000 USD",
                    "channel": "online",
                    "timestamp": "2025-03-15T04:10:00+03:00",
                    "device_fingerprint": "ANON-VPN-001",
                    "ip_address": "185.220.101.34 (VPN exit node, AS208091)",
                    "receiver_bank": "BKMIIRTH (Bank Melli Iran)",
                    "receiver_country": "IR",
                    "receiver_account": "IR062960000000100324200001",
                },
                "suspicious_indicators": [
                    "Destination is Iran (IR) — FATF-listed jurisdiction with deficient AML/CFT controls",
                    "Receiver bank BKMIIRTH (Bank Melli Iran) is subject to international sanctions",
                    "IP 185.220.101.34 is a Tor/VPN exit node — deliberate anonymisation of origin",
                    "35,000 USD exceeds MASAK mandatory reporting threshold (30,000 TRY equivalent)",
                    "First-ever international transfer on a 620-day domestic-only account",
                    "Deep night submission (04:10) with automated characteristics",
                    "Adverse media: counterparty flagged in OFAC SDN-adjacent database",
                ],
                "investigation_summary": (
                    "A 35,000 USD wire from a Turkish account with zero international transaction history "
                    "to sanctioned Iranian bank BKMIIRTH, routed via anonymous VPN at 04:10, represents "
                    "a critical AML event. XGBoost scored 0.8912. Adverse media confirmed OFAC adjacency. "
                    "FATF Recommendation 19 (enhanced due diligence for high-risk jurisdictions) and "
                    "MASAK sanctions provisions mandate immediate transaction block and STR filing."
                ),
                "regulatory_triggers": [
                    "FATF Recommendation 19 — Enhanced Due Diligence for High-Risk Jurisdictions",
                    "FATF Recommendation 13 — Suspicious Transaction Reporting",
                    "MASAK Sanctions Compliance (Law 7262, Art. 4)",
                    "OFAC SDN List — Bank Melli Iran designation",
                    "BDDK AML/CFT Regulation — Cross-border wire controls",
                ],
                "recommended_action": (
                    "1. Block transaction immediately — do not release funds to BKMIIRTH. "
                    "2. File ŞİB with MASAK under Law 7262 (sanctions) and Law 5549 Art. 4 (STR) today. "
                    "3. Notify BDDK within 24 hours per cross-border sanctions protocol. "
                    "4. Freeze all outbound transfers on account TR330006200119800006672357 pending investigation. "
                    "5. Initiate KYC review and contact account holder to establish business justification. "
                    "6. Report to OFAC via FinCEN if USD correspondent banking chain is involved."
                ),
                "generated_at": "2025-03-15T04:28:00+00:00",
                "agent_model": "claude-haiku-4-5-20251001",
            },
            "token_usage": {"input_tokens": 21480, "output_tokens": 3140},
        },
        "elapsed_ms": 118900.0,
    },
]

FIXTURE_INDEX: dict[str, dict] = {f["case"]["case_id"]: f for f in FIXTURES}
