"""FraudLens Streamlit Demo.

Run with:
    streamlit run demo/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_BASE = "http://localhost:8001/api/v1"
_GOLD_SET_PATH = Path(__file__).parent.parent / "tests" / "eval" / "gold_set.jsonl"
_MAX_HISTORY = 5

_HAIKU_INPUT_PRICE_PER_M = 0.80
_HAIKU_OUTPUT_PRICE_PER_M = 4.00

# Modernized color palette for outcomes and tiers
_OUTCOME_COLORS = {
    "approve": "#10B981",  # Emerald Green
    "decline": "#EF4444",  # Red
    "escalate": "#F97316",  # Orange
    "manual_review": "#F59E0B",  # Amber
    "unknown": "#6B7280",  # Gray
}
_OUTCOME_LABELS = {
    "approve": "APPROVED",
    "decline": "DECLINED",
    "escalate": "ESCALATED",
    "manual_review": "MANUAL REVIEW",
    "unknown": "UNKNOWN",
}
_TIER_COLORS = {
    "low": "#10B981",  # Emerald Green
    "medium": "#F59E0B",  # Amber
    "high": "#EF4444",  # Red
}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data
def load_preset_cases() -> list[dict[str, Any]]:
    """Load all gold-set cases for the preset dropdown."""
    if not _GOLD_SET_PATH.exists():
        return []
    cases = []
    with open(_GOLD_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    # Return all cases from the gold set
    return cases


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def call_api(transaction: dict[str, Any], raw_mode: bool = False) -> dict[str, Any] | None:
    """Calls the FraudLens API with the given transaction data."""
    try:
        with httpx.Client(timeout=120.0) as client:
            url = f"{_API_BASE}/transactions" + ("?raw_mode=true" if raw_mode else "")
            resp = client.post(url, json=transaction)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def fetch_daily_cost() -> dict[str, Any] | None:
    """Fetches daily cost metrics from the API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{_API_BASE}/observability/cost")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def load_css(file_name: Path) -> None:
    """Injects a CSS file into the Streamlit app."""
    if not file_name.exists():
        st.warning(f"CSS file not found: {file_name}")
        return
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import plotly.graph_objects as go


def _render_gauge(value: float, title: str, min_val: float, max_val: float, thresholds: dict[str, Any]) -> None:
    """Renders a Plotly gauge chart."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [min_val, max_val], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "#6B7280"},  # Gray bar
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [thresholds["low"][0], thresholds["low"][1]], "color": _TIER_COLORS["low"]},
                    {"range": [thresholds["medium"][0], thresholds["medium"][1]], "color": _TIER_COLORS["medium"]},
                    {"range": [thresholds["high"][0], thresholds["high"][1]], "color": _TIER_COLORS["high"]},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=200,
        font=dict(size=14),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_shap_waterfall(shap_features: list[dict[str, Any]], max_features: int = 5) -> None:
    """Renders a Plotly waterfall chart for SHAP feature contributions."""
    if not shap_features:
        st.caption("No SHAP features available.")
        return

    # Sort features by absolute contribution to show most impactful ones
    sorted_features = sorted(shap_features, key=lambda x: abs(x["contribution"]), reverse=True)
    top_features = sorted_features[:max_features]

    names = [f["feature"] for f in top_features]
    contribs = [f["contribution"] for f in top_features]
    text_values = [f"{c:+.4f}" for c in contribs]
    colors = [_OUTCOME_COLORS["decline"] if c > 0 else _OUTCOME_COLORS["approve"] for c in contribs]

    fig = go.Figure(
        go.Bar( # Changed from Waterfall to Bar
            orientation="h",
            y=names,
            x=contribs,
            text=text_values,
            textposition="outside",
            marker=dict(color=colors), # Use marker.color for go.Bar
        )
    )
    fig.update_layout(
        title_text="SHAP Feature Contributions",
        xaxis_title="Contribution to ML Score",
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        showlegend=False,
        yaxis={"autorange": "reversed"},  # Display largest contribution at the top
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _badge(text: str, bg_color: str) -> None:
    """Renders a styled badge using custom CSS."""
    st.markdown(
        f'<span class="badge" style="background-color:{bg_color};">{text}</span>',
        unsafe_allow_html=True,
    )


def _render_ml_section(resp: dict[str, Any]) -> None:
    """Renders the ML scoring section with gauge and waterfall charts."""
    prob = resp["fraud_probability"]
    tier = resp["risk_tier"]
    col1, col2 = st.columns([1, 2])
    with col1:
        _render_gauge(prob, "Fraud Probability", 0.0, 1.0, {"low": [0.0, 0.3], "medium": [0.3, 0.7], "high": [0.7, 1.0]})
        tier_color = _TIER_COLORS.get(tier, "#6B7280")
        _badge(f"{tier.upper()} RISK", tier_color)
    with col2:
        _render_shap_waterfall(resp.get("shap_top_features", []))


def _render_agent_section(resp: dict[str, Any]) -> None:
    """Renders the agent investigation section."""
    inv = resp.get("investigation")
    triage = resp.get("triage_action", "approve")
    tier = resp.get("risk_tier")

    if triage == "approve" or inv is None:
        st.info("No agent investigation — transaction was auto-approved by ML tier.")
        return
    
    st.subheader("Agent Details")
    agent_type = "Critical Agent" if triage == "escalate" else "Investigation Agent"
    agent_color = _TIER_COLORS.get(tier, "#6B7280")
    _badge(agent_type, agent_color)
    st.markdown("") # Add a small vertical space

    # Display decision hint and confidence using st.metric, which will be styled as cards
    fd = resp.get("fraud_decision") or {}
    confidence = fd.get("confidence", inv.get("confidence", 0.0))
    col1, col2 = st.columns(2)
    with col1:
        hint = fd.get("decision_hint", inv.get("decision_hint", ""))
        st.metric("Decision Hint", hint.replace("_", " ").title())
    with col2:
        _render_gauge(confidence, "Confidence", 0.0, 1.0, {"low": [0.7, 1.0], "medium": [0.4, 0.7], "high": [0.0, 0.4]})

    st.divider()
    st.subheader("Tool Execution Trace")
    tool_trace: list[dict[str, Any]] = inv.get("tool_trace", [])
    tools_called: list[str] = inv.get("tools_called", [])

    if tool_trace:
        for entry in tool_trace:
            tool_name = entry.get("tool", "unknown")
            with st.expander(f"Tool: `{tool_name}`"):
                st.json(entry.get("result", "No result"))
    elif tools_called:
        st.markdown("Tools called (no detailed trace available):")
        for t in tools_called:
            st.markdown(f"- `{t}`")
    else:
        st.info("No tools were called by the agent for this transaction.")


def _render_decision_section(resp: dict[str, Any]) -> None:
    fd = resp.get("fraud_decision")
    triage = resp.get("triage_action", "approve")

    if fd is None and triage == "approve":
        _badge(_OUTCOME_LABELS["approve"], _OUTCOME_COLORS["approve"])
        st.success("Transaction automatically approved — below ML fraud threshold.")
        return

    if fd is None:
        st.warning("No fraud decision available.")
        return
    
    outcome = fd.get("outcome", "unknown")
    outcome_color = _OUTCOME_COLORS.get(outcome, "#6B7280")
    outcome_label = _OUTCOME_LABELS.get(outcome, outcome.upper())

    st.markdown(
        f'<div style="text-align:center;padding:18px;">'
        f'<span style="background:{outcome_color};color:#fff;padding:10px 36px;'
        f'border-radius:12px;font-weight:900;font-size:1.6rem;">{outcome_label}</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    
    st.container(border=True).markdown(f"**Reasoning Summary:** {fd.get('reasoning', 'No summary provided.')}")

    st.divider()
    st.subheader("Investigation Findings")
    col_evidence, col_red_flags = st.columns(2)
    with col_evidence:
        st.markdown("##### Evidence")
        for i, ev in enumerate(fd.get("evidence", [])):
            st.markdown(f"{i+1}. {ev}")
    with col_red_flags:
        st.markdown("##### Red Flags")
        if fd.get("red_flags"):
            tags_html = " ".join(
                f'<span class="badge" style="background-color:#EF4444;">{rf}</span>' for rf in fd["red_flags"]
            )
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.caption("None")

    citations = fd.get("regulatory_citations", [])
    if citations:
        st.divider()
        st.markdown("##### Regulatory Citations")
        for cit in citations:
            src = cit.get("source", "")
            page = cit.get("page")
            excerpt = cit.get("excerpt", "")[:200]
            score = cit.get("relevance_score", 0.0)
            ref = f"{src}" + (f", p.{page}" if page else "")
            with st.expander(f"{ref} (relevance: {score:.2f})"):
                st.write(excerpt)

    if fd.get("key_risk_factors"):
        st.divider()
        st.markdown(f"**Key Risk Factors:** {fd['key_risk_factors']}")


def _render_sar_section(resp: dict[str, Any]) -> None:
    sar = resp.get("sar_report")
    if not sar:
        return

    st.markdown("---")
    st.subheader("Suspicious Activity Report")

    sections = {
        "Customer Info": sar.get("customer_info", {}),
        "Transaction Details": sar.get("transaction_details", {}),
        "Suspicious Indicators": sar.get("suspicious_indicators", []),
        "Regulatory Triggers": sar.get("regulatory_triggers", []),
    }
    for label, content in sections.items():
        with st.expander(label, expanded=False):
            if isinstance(content, list):
                for item in content:
                    st.markdown(f"- {item}")
            else:
                st.json(content)

    if sar.get("investigation_summary"):
        st.markdown("**Investigation Summary**")
        st.write(sar["investigation_summary"])

    if sar.get("recommended_action"):
        st.warning(f"**Recommended Action:** {sar['recommended_action']}")

    sar_json = json.dumps(sar, indent=2, default=str)
    st.download_button(
        label="Download SAR (JSON)",
        data=cast(str, sar_json), # Cast to str to satisfy mypy
        file_name=f"SAR_{sar.get('transaction_id', 'unknown')}.json",
        mime="application/json",
    )


def _render_observability_section(resp: dict[str, Any], elapsed_ms: float) -> None:
    st.markdown("---")
    st.subheader("Observability")
    
    usage = resp.get("token_usage") or {}
    input_tok = usage.get("input_tokens", 0)
    output_tok = usage.get("output_tokens", 0)
    cost_usd = (
        (input_tok * _HAIKU_INPUT_PRICE_PER_M + output_tok * _HAIKU_OUTPUT_PRICE_PER_M)
        / 1_000_000
    )

    tools_count = len((resp.get("investigation") or {}).get("tools_called", []))

    col1, col2, col3, col4 = st.columns(4) # Use st.metric for these
    with col1:
        st.metric("Total Latency", f"{elapsed_ms:.0f} ms", help="End-to-end latency including network and API processing.")
    with col2:
        st.metric("Estimated Cost", f"${cost_usd:.4f} USD", help="Estimated LLM token cost for this transaction.")
    with col3:
        st.metric("Tools Called", tools_count, help="Number of tools invoked by the agent.")
    with col4:
        st.metric("API Processing Time", f"{resp.get('processing_time_ms', 0):.0f} ms", help="Time spent by the FraudLens API processing the request.")

    settings_json = {}
    try:
        from fraudlens.core.config import get_settings  # noqa: PLC0415

        s = get_settings()
        if s.langsmith_tracing:
            project = s.langsmith_project
            base = s.langsmith_base_url
            tx_id = resp.get("transaction_id", "")
            settings_json = {"langsmith_trace": f"{base}/o/{project}?filter=transaction_id:{tx_id}"}
    except Exception:
        pass

    if settings_json.get("langsmith_trace"):
        st.markdown(f"[View LangSmith Trace]({settings_json['langsmith_trace']})")

    daily_cost = fetch_daily_cost()
    if daily_cost:
        st.markdown(f"**Today's total cost:** ${daily_cost.get('today', {}).get('cost_usd', 0.0):.4f} USD")


# ---------------------------------------------------------------------------
# Manual input form
# ---------------------------------------------------------------------------


def _build_manual_transaction() -> dict[str, Any]:
    """Builds a transaction dictionary from manual sidebar inputs."""
    st.sidebar.markdown("### Manual Transaction Input")
    customers = {
        "ACC-RETAIL-001 (Ayşe Y.)": "TR330006200119800006672315",
        "ACC-BUSINESS-002 (Mehmet K.)": "TR330006200119800006672340",
        "ACC-VIP-003 (Zeynep A.)": "TR330006200119800006672355",
    }
    customer_label = st.sidebar.selectbox("Customer", list(customers.keys()))
    sender_id = customers[customer_label] if customer_label else "UNKNOWN"

    amount = st.sidebar.number_input("Amount", min_value=1.0, max_value=500000.0, value=250.0, step=50.0)
    currency = st.sidebar.selectbox("Currency", ["TRY", "USD", "EUR"])
    tx_type = st.sidebar.selectbox("Transaction Type", ["purchase", "transfer", "payment", "withdrawal", "deposit"])
    channel = st.sidebar.selectbox("Channel", ["online", "mobile", "atm", "branch", "api"])
    country = st.sidebar.selectbox("Receiver Country", ["TR", "US", "DE", "NL", "AE", "RU", "IR", "VG", "KP"]) # type: ignore
    hour = st.sidebar.slider("Hour of Day", 0, 23, 14)
    is_night = st.sidebar.checkbox("Night transaction (00-06)", value=(hour < 6))

    mcc_map = {"grocery": "5411", "restaurant": "5812", "electronics": "5732", "crypto": "6051", "hotel": "7011", "airline": "4511", "pharmacy": "5912"}
    mcc_key = st.sidebar.selectbox("Merchant Category", list(mcc_map.keys()))

    ts = f"2025-03-15T{hour:02d}:00:00+03:00"
    needs_ip = channel in ("online", "mobile")
    needs_mcc = tx_type == "purchase"

    txn: dict[str, Any] = {
        "timestamp": ts,
        "amount": amount,
        "currency": currency,
        "transaction_type": tx_type,
        "channel": channel,
        "sender_account_id": sender_id,
        "sender_bank_code": "AKBNKTR",
        "sender_country": "TR",
        "receiver_account_id": f"RCV-{country}-{int(amount):06d}",
        "receiver_bank_code": "GARBBTR" if country == "TR" else "INTLBANK", # type: ignore
        "receiver_country": country,
    }
    if needs_ip:
        txn["ip_address"] = "78.181.10.22" if not is_night else "185.220.101.5"
    if needs_mcc:
        txn["merchant_id"] = f"MERCH-{mcc_key.upper()}-001"
        txn["merchant_category_code"] = mcc_map[mcc_key]
        txn["merchant_name"] = mcc_key.title() + " Store"
    
    return txn


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="FraudLens Demo", layout="wide")
    load_css(Path(__file__).parent / "assets" / "style.css")
    st.title("FraudLens — AML / Fraud Detection Demo")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar ----------------------------------------------------------------
    st.sidebar.title("FraudLens")
    mode = st.sidebar.radio("Input Mode", ["Preset Cases", "Manual Input"], index=0)

    preset_cases = load_preset_cases()
    preset_labels = {c["case_id"]: f"{c['case_id']} - {c['description']}" for c in preset_cases}

    transaction: dict[str, Any] | None = None
    selected_case_info: dict[str, Any] | None = None

    if mode == "Preset Cases":
        if not preset_cases:
            st.sidebar.warning("gold_set.jsonl not found or empty.")
        else:
            selected_id = st.sidebar.selectbox(
                "Select Scenario",
                list(preset_labels.keys()),
                format_func=lambda k: preset_labels[k],
                index=0,
            )
            selected_case = next(c for c in preset_cases if c["case_id"] == selected_id)
            transaction = selected_case["transaction"]
            selected_case_info = selected_case
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("##### Selected Case Summary")
            st.sidebar.markdown(f"**Case ID:** {selected_case_info['case_id']}")
            st.sidebar.markdown(f"**Description:** {selected_case_info['description']}")            
            st.sidebar.markdown(f"**Risk Tier:** {selected_case_info['tier'].upper()}")
            st.sidebar.markdown(f"**Expected Triage:** {_OUTCOME_LABELS.get(selected_case_info['expected_triage'], 'UNKNOWN')}")
            st.sidebar.markdown(f"**Expected Outcome:** {_OUTCOME_LABELS.get(selected_case_info['expected_outcome'], 'UNKNOWN')}")
            if selected_case_info.get("expected_sar"):
                st.sidebar.markdown("**Expected SAR:** Yes")
            else:
                st.sidebar.markdown("**Expected SAR:** No")
            
            st.sidebar.markdown("---")
            with st.sidebar.expander("View Raw Transaction JSON"):
                st.json(transaction)
    else:
        transaction = _build_manual_transaction()

    if st.sidebar.button("Run Investigation", type="primary", use_container_width=True):
        if transaction:
            with st.spinner("Investigating..."):
                import time as _time  # noqa: PLC0415

                t0 = _time.perf_counter() # Start timer for total latency
                resp = call_api(transaction, raw_mode=selected_case_info.get("raw_mode", False) if mode == "Preset Cases" else False)
                elapsed_ms = (_time.perf_counter() - t0) * 1000

            if resp:
                entry = {"transaction": transaction, "response": resp, "elapsed_ms": elapsed_ms}
                st.session_state.history.insert(0, entry)
                st.session_state.history = st.session_state.history[:_MAX_HISTORY] # Keep only max_history items
                st.session_state.latest = entry
            else:
                st.error("API call failed. Please check the console for errors or ensure the API is running.")
                return


    if st.session_state.history:
        history_labels = [
            f"{i + 1}. {h['response'].get('triage_action', '?').upper()} - {h['transaction'].get('amount')} {h['transaction'].get('currency')}"
            for i, h in enumerate(st.session_state.history)
        ]
        selected_hist = st.sidebar.selectbox(f"History (last {_MAX_HISTORY})", range(len(history_labels)), format_func=lambda i: history_labels[i])
        active = st.session_state.history[selected_hist]
    else:
        active = None

    if not active: # This should ideally not happen if history is managed correctly
        st.info("No investigation run yet. Select a preset case or fill the manual form and click 'Run Investigation'.")
        return
    
    resp = active["response"]
    elapsed_ms = active["elapsed_ms"]

    # Main panels ------------------------------------------------------------

    tab_ml, tab_agent, tab_decision, tab_obs = st.tabs(["ML Score", "Agent Trace", "Final Decision", "Observability"])

    with tab_ml: # ML Scoring Tab
        st.header("ML Scoring")
        _render_ml_section(resp)

    with tab_agent: # Agent Trace Tab
        st.header("Agent Investigation")
        _render_agent_section(resp)

    with tab_decision: # Final Decision Tab
        st.header("Final Decision")
        _render_decision_section(resp)
        _render_sar_section(resp)

    with tab_obs:
        _render_observability_section(resp, elapsed_ms)


if __name__ == "__main__":
    main()
