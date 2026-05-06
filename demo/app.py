"""FraudLens Compliance Dashboard.

Run with:
    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE = "http://localhost:8001/api/v1"
_GOLD_SET_PATH = Path(__file__).parent.parent / "tests" / "eval" / "gold_set.jsonl"
_MAX_HISTORY = 8
_POLL_INTERVAL = 0.4
_POLL_TIMEOUT = 180.0

_HAIKU_IN_PRICE = 0.80    # USD per 1M input tokens
_HAIKU_OUT_PRICE = 4.00   # USD per 1M output tokens

_OUTCOME_LABEL: dict[str, str] = {
    "approve": "APPROVED",
    "decline": "DECLINED",
    "escalate": "ESCALATED",
    "manual_review": "MANUAL REVIEW",
    "unknown": "UNKNOWN",
}
_TIER_COLOR: dict[str, str] = {
    "low": "#10B981",
    "medium": "#F59E0B",
    "high": "#EF4444",
    "unknown": "#64748B",
}

# Full system pipeline layout:
#
#   [XGBoost] → [Triage] → ┬─ [Investigation] ─┬ → [Synthesize] → [SAR] → [Done]
#                           └─ [Critical]       ┘
#
# Investigation and Critical are always shown as parallel branches (fork).
# Only one fires per transaction; the other is "skipped" (visually muted).
# This makes the intent-based multi-agent architecture visible from the loading screen.

# Node index reference: 0=XGBoost 1=Triage | fork: 2=Investigation 3=Critical | 4=Synthesize 5=SAR 6=Done
_ALL_NODE_LABELS = ["XGBoost", "Triage", "Investigation", "Critical", "Synthesize", "SAR", "Done"]

# State per node: "active" | "done" | "skipped" | (absent = "pending")
_PIPELINE_STATES: dict[str, dict[str, dict[int, str]]] = {
    "low": {
        "queued":  {},
        "scoring": {0: "active"},
        "routing": {0: "done", 1: "active", 2: "skipped", 3: "skipped", 4: "skipped", 5: "skipped"},
        "done":    {0: "done", 1: "done",   2: "skipped", 3: "skipped", 4: "skipped", 5: "skipped", 6: "done"},
    },
    "medium": {
        "queued":        {},
        "scoring":       {0: "active"},
        "routing":       {0: "done", 1: "active", 3: "skipped", 5: "skipped"},
        "investigating": {0: "done", 1: "done", 2: "active",  3: "skipped", 5: "skipped"},
        "synthesizing":  {0: "done", 1: "done", 2: "done",   3: "skipped", 4: "active", 5: "skipped"},
        "done":          {0: "done", 1: "done", 2: "done",   3: "skipped", 4: "done",   5: "skipped", 6: "done"},
    },
    "high": {
        "queued":       {},
        "scoring":      {0: "active"},
        "routing":      {0: "done", 1: "active", 2: "skipped"},
        "critical":     {0: "done", 1: "done",   2: "skipped", 3: "active"},
        "synthesizing": {0: "done", 1: "done",   2: "skipped", 3: "done",   4: "active"},
        "sar":          {0: "done", 1: "done",   2: "skipped", 3: "done",   4: "done",  5: "active"},
        "done":         {0: "done", 1: "done",   2: "skipped", 3: "done",   4: "done",  5: "done",  6: "done"},
    },
    "unknown": {
        "queued":        {},
        "scoring":       {0: "active"},
        "routing":       {0: "done", 1: "active"},
        "investigating": {0: "done", 1: "done", 2: "active"},
        "critical":      {0: "done", 1: "done", 3: "active"},
        "synthesizing":  {0: "done", 1: "done", 4: "active"},
        "sar":           {0: "done", 1: "done", 4: "done", 5: "active"},
        "done":          {0: "done", 1: "done", 6: "done"},
    },
}

# Tool purpose descriptions — shown in investigation timeline cards
_TOOL_PURPOSE: dict[str, str] = {
    "get_customer_history":      "Customer behavior baseline",
    "find_similar_patterns":     "Historical fraud pattern match",
    "check_merchant_reputation": "Merchant risk & chargeback rate",
    "get_geolocation_context":   "IP/device location verification",
    "explain_ml_score":          "SHAP feature attribution",
    "deep_network_analysis":     "Transaction network graph",
    "regulatory_policy_rag":     "BDDK/FATF regulatory lookup",
    "adverse_media_search":      "Sanctions & PEP screening",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md_to_safe_html(text: str) -> str:
    """Convert a subset of markdown to safe HTML for injection into styled divs."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"`(.+?)`",       r"<code>\1</code>",      text)
    return text.replace("\n", "<br>")


def _escape_html(text: str) -> str:
    """Escape HTML special characters and normalise newlines."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# ---------------------------------------------------------------------------
# Data & CSS loaders
# ---------------------------------------------------------------------------


@st.cache_data
def load_preset_cases() -> list[dict[str, Any]]:
    """Load all gold-set evaluation cases."""
    if not _GOLD_SET_PATH.exists():
        return []
    cases: list[dict[str, Any]] = []
    with open(_GOLD_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_css(path: Path) -> None:
    """Inject a CSS file into the Streamlit app."""
    if not path.exists():
        return
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def submit_job(transaction: dict[str, Any], raw_mode: bool = False) -> str | None:
    """POST /transactions and return job_id, or None on error."""
    url = f"{_API_BASE}/transactions" + ("?raw_mode=true" if raw_mode else "")
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=transaction)
            resp.raise_for_status()
            return str(resp.json()["job_id"])
    except Exception as exc:
        st.error(f"Failed to submit transaction: {exc}")
        return None


def poll_job(job_id: str) -> dict[str, Any] | None:
    """GET /jobs/{job_id} and return state dict, or None on network error."""
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{_API_BASE}/jobs/{job_id}")
            resp.raise_for_status()
            return resp.json()  # type: ignore[return-value]
    except Exception as exc:
        st.error(f"Polling error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Loading-phase renderers
# ---------------------------------------------------------------------------


def _node_html(idx: int, node_states: dict[int, str]) -> str:
    """Return the HTML string for a single pipeline node."""
    label = _ALL_NODE_LABELS[idx]
    state = node_states.get(idx, "pending")
    if state == "done":
        cls, prefix = "fl-node-done", "✓ "
    elif state == "active":
        cls, prefix = "fl-node-active", ""
    elif state == "skipped":
        cls, prefix = "fl-node-skipped", ""
    else:
        cls, prefix = "fl-node-pending", ""
    return f'<div class="fl-pipeline-node {cls}">{prefix}{label}</div>'


def render_pipeline_nodes(status: str, tier: str) -> None:
    """Render the full system pipeline with a fork for the Investigation/Critical branch.

    Layout:
      [XGBoost] → [Triage] → ┬─ [Investigation] ─┬ → [Synthesize] → [SAR] → [Done]
                              └─ [Critical]       ┘

    All nodes are always shown. The current tier+status determine which nodes are
    active, done, or skipped — making the multi-agent routing visible at all times.
    """
    node_states: dict[int, str] = _PIPELINE_STATES.get(
        tier, _PIPELINE_STATES["unknown"]
    ).get(status, {})

    arrow = '<span class="fl-pipeline-arrow">→</span>'

    fork_html = (
        '<div class="fl-pipeline-fork">'
        + _node_html(2, node_states)
        + '<div class="fl-pipeline-fork-or">OR</div>'
        + _node_html(3, node_states)
        + '</div>'
    )

    parts = (
        ['<div class="fl-pipeline">']
        + [_node_html(0, node_states), arrow, _node_html(1, node_states), arrow]
        + [fork_html, arrow]
        + [_node_html(4, node_states), arrow, _node_html(5, node_states), arrow]
        + [_node_html(6, node_states)]
        + ['</div>']
    )
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_thought_bubble(thought: str) -> None:
    """Render full agent thought with markdown support — no character cap."""
    html = _md_to_safe_html(thought)
    st.markdown(
        '<p class="fl-thought-label">🤔 Agent Thinking</p>'
        f'<div class="fl-thought-bubble">{html}</div>',
        unsafe_allow_html=True,
    )


def render_tool_tracker(completed: list[str], current: str | None) -> None:
    """Render done/active tool badges."""
    if not completed and not current:
        return
    parts = ['<div class="fl-tool-badges">']
    for tool in completed:
        parts.append(f'<span class="fl-tool-badge fl-badge-done">✓ {tool}</span>')
    if current:
        parts.append(f'<span class="fl-tool-badge fl-badge-active">⟳ {current}</span>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Result-phase renderers
# ---------------------------------------------------------------------------


def _gauge_fig(value: float) -> go.Figure:
    """Dark-themed Plotly gauge for fraud probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"color": "#E2E8F0", "size": 34}, "valueformat": ".3f"},
        gauge={
            "axis": {
                "range": [0, 1],
                "tickcolor": "#64748B",
                "tickfont": {"color": "#64748B", "size": 10},
                "tickwidth": 1,
            },
            "bar": {"color": "#6C63FF", "thickness": 0.28},
            "bgcolor": "#151822",
            "borderwidth": 0,
            "steps": [
                {"range": [0.0, 0.3], "color": "rgba(16,185,129,0.12)"},
                {"range": [0.3, 0.7], "color": "rgba(245,158,11,0.12)"},
                {"range": [0.7, 1.0], "color": "rgba(239,68,68,0.12)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#1E2130",
        plot_bgcolor="#1E2130",
        font={"color": "#E2E8F0"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=260,
    )
    return fig


def _shap_fig(features: list[dict[str, Any]]) -> go.Figure:
    """Dark-themed horizontal bar chart for SHAP contributions."""
    sorted_f = sorted(features, key=lambda x: abs(x["contribution"]), reverse=True)[:7]
    names = [f["feature"] for f in sorted_f]
    contribs = [f["contribution"] for f in sorted_f]
    colors = ["rgba(239,68,68,0.85)" if c > 0 else "rgba(16,185,129,0.85)" for c in contribs]

    fig = go.Figure(go.Bar(
        orientation="h",
        y=names,
        x=contribs,
        text=[f"{c:+.4f}" for c in contribs],
        textposition="outside",
        textfont={"color": "#94A3B8", "size": 10},
        marker={"color": colors},
    ))
    fig.update_layout(
        paper_bgcolor="#1E2130",
        plot_bgcolor="#1E2130",
        font={"color": "#E2E8F0"},
        xaxis={
            "gridcolor": "#2D3250",
            "zerolinecolor": "#4A5568",
            "tickfont": {"color": "#64748B", "size": 10},
            "title": {"text": "Contribution", "font": {"color": "#64748B", "size": 11}},
        },
        yaxis={
            "gridcolor": "#2D3250",
            "autorange": "reversed",
            "tickfont": {"color": "#E2E8F0", "size": 10},
        },
        margin={"l": 10, "r": 70, "t": 20, "b": 20},
        height=260,
        showlegend=False,
    )
    return fig


_OUTCOME_COLOR: dict[str, str] = {
    "approve": "#10B981",
    "decline": "#EF4444",
    "escalate": "#F97316",
    "manual_review": "#F59E0B",
    "unknown": "#64748B",
}


def render_verdict_hero(result: dict[str, Any]) -> None:
    """Render 3 equal full-width verdict cards: ML Score | Decision | Confidence."""
    fd = result.get("fraud_decision") or {}
    triage = result.get("triage_action", "approve")
    outcome = fd.get("outcome") or ("approve" if triage == "approve" else "unknown")

    label = _OUTCOME_LABEL.get(outcome, outcome.upper())
    prob = result.get("fraud_probability", 0.0)
    inv = result.get("investigation") or {}
    confidence = fd.get("confidence") or inv.get("confidence")
    decision_color = _OUTCOME_COLOR.get(outcome, "#64748B")
    conf_str = f"{confidence:.0%}" if confidence is not None else "—"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="fl-verdict-card">'
            f'<div class="fl-verdict-label">ML Score</div>'
            f'<div class="fl-verdict-value">{prob:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="fl-verdict-card">'
            f'<div class="fl-verdict-label">Decision</div>'
            f'<div class="fl-verdict-value" style="color:{decision_color};">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="fl-verdict-card">'
            f'<div class="fl-verdict-label">Confidence</div>'
            f'<div class="fl-verdict-value">{conf_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_ml_section(result: dict[str, Any]) -> None:
    """Render fraud probability gauge and SHAP feature bar chart."""
    prob = result.get("fraud_probability", 0.0)
    shap_features = result.get("shap_top_features") or []

    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.plotly_chart(_gauge_fig(prob), use_container_width=True, config={"displayModeBar": False})
    with col2:
        if shap_features:
            st.markdown(
                '<div class="fl-subsection-label">SHAP Feature Contributions</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_shap_fig(shap_features), use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No SHAP features available.")


def render_investigation_timeline(result: dict[str, Any]) -> None:
    """Render rich tool trace cards (3 per row) with sequence, purpose, and expandable result."""
    inv = result.get("investigation")
    if not inv:
        return

    tool_trace: list[dict[str, Any]] = inv.get("tool_trace") or []
    tools_called: list[str] = inv.get("tools_called") or []

    if tool_trace:
        per_row = 3
        for row_start in range(0, len(tool_trace), per_row):
            row_entries = tool_trace[row_start : row_start + per_row]
            cols = st.columns(len(row_entries))
            for col, (idx, entry) in zip(cols, enumerate(row_entries, start=row_start + 1)):
                tool_name = entry.get("tool", "unknown")
                purpose = _TOOL_PURPOSE.get(tool_name, "")
                args = entry.get("args") or {}
                raw_result = entry.get("result", "")

                args_preview = (
                    json.dumps(args, default=str)[:120]
                    if isinstance(args, dict) and args
                    else str(args)[:120]
                )
                args_esc = _escape_html(args_preview)

                with col:
                    st.markdown(
                        f'<div class="fl-tool-card fl-fade-in">'
                        f'<div class="fl-tool-card-header">'
                        f'<span class="fl-tool-seq">{idx}</span>'
                        f'<span class="fl-tool-card-name">{tool_name}</span>'
                        f'</div>'
                        f'<div class="fl-tool-purpose">{purpose}</div>'
                        f'<div class="fl-tool-card-result">{args_esc}…</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if raw_result:
                        with st.expander("→ Result"):
                            if isinstance(raw_result, dict):
                                st.json(raw_result)
                            else:
                                st.markdown(str(raw_result))
    elif tools_called:
        parts = ['<div class="fl-tool-badges">']
        for t in tools_called:
            parts.append(f'<span class="fl-tool-badge fl-badge-done">✓ {t}</span>')
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

    # reasoning_summary intentionally omitted here —
    # the synthesized reasoning already appears in render_decision_detail()


def render_decision_detail(result: dict[str, Any]) -> None:
    """Render evidence list, red flag tags, and regulatory citation cards."""
    fd = result.get("fraud_decision")
    triage = result.get("triage_action", "approve")

    if fd is None:
        if triage == "approve":
            st.success("Transaction auto-approved — below ML fraud threshold.")
        return

    reasoning = fd.get("reasoning", "")
    if reasoning:
        st.markdown(
            f'<div class="fl-card" style="font-size:0.88rem;color:#CBD5E1;">'
            f'<span style="color:#64748B;font-size:0.72rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;">Reasoning</span><br><br>'
            f'{reasoning}'
            f'</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)

    _REG_PREFIXES = ("FATF", "BDDK", "MASAK", "Rec.", "Article", "Madde", "§")

    with col1:
        st.markdown('<div class="fl-section-title">Evidence</div>', unsafe_allow_html=True)
        plain_n = 0
        for ev in fd.get("evidence") or []:
            # Regulatory citation items belong in Regulatory Citations section only
            if any(ev.lstrip().startswith(p) for p in _REG_PREFIXES):
                continue
            plain_n += 1
            st.markdown(f"{plain_n}. {ev}")

    with col2:
        st.markdown('<div class="fl-section-title">Red Flags</div>', unsafe_allow_html=True)
        flags = fd.get("red_flags") or []
        if flags:
            html = " ".join(f'<span class="fl-red-flag">{rf}</span>' for rf in flags)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<span style="color:#64748B;font-size:0.82rem;">None detected</span>',
                unsafe_allow_html=True,
            )

        citations = fd.get("regulatory_citations") or []
        if citations:
            st.markdown(
                '<div class="fl-section-title" style="margin-top:18px;">Regulatory Citations</div>',
                unsafe_allow_html=True,
            )
            for cit in citations:
                src = cit.get("source", "")
                page = cit.get("page")
                excerpt = cit.get("excerpt", "")   # full excerpt, no truncation
                score = cit.get("relevance_score", 0.0)
                ref = src.replace(".pdf", "") + (f" · p.{page}" if page else "")
                score_color = (
                    "#10B981" if score >= 0.7
                    else "#F59E0B" if score >= 0.5
                    else "#64748B"
                )
                # <details>/<summary> gives native browser collapse — no JS needed
                st.markdown(
                    f'<details class="fl-citation-card">'
                    f'<summary>'
                    f'<span class="fl-citation-ref">{_escape_html(ref)}</span>'
                    f'<span class="fl-citation-score" style="color:{score_color};">score {score:.2f}</span>'
                    f'</summary>'
                    f'<div class="fl-citation-excerpt">{_escape_html(excerpt)}</div>'
                    f'</details>',
                    unsafe_allow_html=True,
                )


def render_sar_section(result: dict[str, Any]) -> None:
    """Render SAR report section (only when outcome == escalate)."""
    sar = result.get("sar_report")
    fd = result.get("fraud_decision")
    outcome = fd.get("outcome") if fd else None
    if not sar or outcome != "escalate":
        return

    col_title, col_dl = st.columns([5, 1])
    with col_title:
        st.markdown(
            '<div class="fl-section-title">Suspicious Activity Report</div>',
            unsafe_allow_html=True,
        )
    with col_dl:
        st.download_button(
            "↓ SAR JSON",
            data=json.dumps(sar, indent=2, default=str),
            file_name=f"SAR_{result.get('transaction_id', 'unknown')}.json",
            mime="application/json",
        )

    # Recommended Action — fixed/expanded at top
    rec = sar.get("recommended_action")
    if rec:
        st.markdown(
            '<div class="fl-subsection-label">Recommended Action</div>',
            unsafe_allow_html=True,
        )
        if isinstance(rec, list):
            for item in rec:
                st.markdown(f"- {item}")
        else:
            st.markdown(str(rec))

    st.markdown("<br>", unsafe_allow_html=True)

    collapsible: list[tuple[str, Any]] = [
        ("Customer Info", sar.get("customer_info")),
        ("Transaction Details", sar.get("transaction_details")),
        ("Suspicious Indicators", sar.get("suspicious_indicators")),
        ("Investigation Summary", sar.get("investigation_summary")),
        ("Regulatory Triggers", sar.get("regulatory_triggers")),
    ]
    for label, content in collapsible:
        if not content:
            continue
        with st.expander(label, expanded=False):
            if isinstance(content, list):
                for item in content:
                    st.markdown(f"- {item}")
            elif isinstance(content, dict):
                st.json(content)
            else:
                st.markdown(str(content))


def render_observability_footer(result: dict[str, Any], elapsed_ms: float) -> None:
    """Render 4-metric observability footer."""
    usage = result.get("token_usage") or {}
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    cost = (in_tok * _HAIKU_IN_PRICE + out_tok * _HAIKU_OUT_PRICE) / 1_000_000
    tools_n = len((result.get("investigation") or {}).get("tools_called") or [])
    proc_ms = result.get("processing_time_ms", 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Latency", f"{elapsed_ms:.0f} ms")
    with col2:
        st.metric("Est. LLM Cost", f"${cost:.5f}")
    with col3:
        st.metric("Tools Called", str(tools_n))
    with col4:
        st.metric("API Processing", f"{proc_ms:.0f} ms")


def render_results(entry: dict[str, Any]) -> None:
    """Render the full results page for a completed investigation."""
    result = entry["result"]
    elapsed_ms = entry["elapsed_ms"]

    render_verdict_hero(result)
    st.divider()

    st.markdown('<div class="fl-section-title">ML Intelligence</div>', unsafe_allow_html=True)
    render_ml_section(result)
    st.divider()

    triage = result.get("triage_action", "approve")
    if triage != "approve" and result.get("investigation"):
        st.markdown('<div class="fl-section-title">Investigation Timeline</div>', unsafe_allow_html=True)
        render_investigation_timeline(result)
        st.divider()

    st.markdown('<div class="fl-section-title">Decision Detail</div>', unsafe_allow_html=True)
    render_decision_detail(result)

    fd = result.get("fraud_decision")
    if fd and fd.get("outcome") == "escalate":
        st.divider()
        render_sar_section(result)

    st.divider()
    st.markdown('<div class="fl-section-title">Observability</div>', unsafe_allow_html=True)
    render_observability_footer(result, elapsed_ms)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _outcome_for_result(result: dict[str, Any]) -> str:
    """Extract the final outcome from a completed result dict."""
    fd = result.get("fraud_decision")
    if fd and fd.get("outcome"):
        return str(fd["outcome"])
    return "approve" if result.get("triage_action") == "approve" else "unknown"


def render_sidebar(cases: list[dict[str, Any]]) -> None:
    """Render the full sidebar: logo, case selector, run button, history."""
    with st.sidebar:
        st.markdown(
            '<div class="fl-sidebar-header">'
            '<div style="font-size:1.35rem;font-weight:800;color:#E2E8F0;letter-spacing:-0.02em;">'
            'FraudLens</div>'
            '<div style="font-size:0.72rem;color:#64748B;margin-top:3px;">'
            'AML · Compliance · Intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if not cases:
            st.warning("gold_set.jsonl not found or empty.")
            return

        _TIER_ORDER = {"high": 0, "medium": 1, "low": 2}
        sorted_cases = sorted(
            cases,
            key=lambda c: (_TIER_ORDER.get(c.get("tier", ""), 99), c["case_id"]),
        )

        labels = {
            c["case_id"]: (
                f"[{c['tier'].upper()}]  {c['case_id']} — "
                + (c["description"][:38] + "…" if len(c["description"]) > 38 else c["description"])
            )
            for c in sorted_cases
        }

        ids = list(labels.keys())

        # Ensure session state holds a valid id before the widget renders
        if st.session_state.get("selected_case_id") not in ids:
            st.session_state.selected_case_id = ids[0]

        selected_id: str = st.selectbox(
            "Case",
            ids,
            format_func=lambda k: labels[k],
            key="selected_case_id",
            label_visibility="collapsed",
        )
        selected_case = next(c for c in cases if c["case_id"] == selected_id)

        tier = selected_case.get("tier", "unknown")
        tier_color = _TIER_COLOR.get(tier, "#64748B")
        tags = selected_case.get("tags") or []
        tags_html = " ".join(f'<span class="fl-tag">{t}</span>' for t in tags)
        desc = selected_case.get("description", "")

        st.markdown(
            f'<div class="fl-case-card">'
            f'<span class="fl-tier-badge" style="background:{tier_color};">{tier.upper()}</span>'
            f'<div style="font-size:0.81rem;color:#CBD5E1;margin-bottom:7px;">{desc}</div>'
            f'{tags_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        view_entry = st.session_state.get("view_entry")
        if view_entry and view_entry.get("case", {}).get("case_id") == selected_id:
            expected = selected_case.get("expected_outcome", "")
            actual = _outcome_for_result(view_entry["result"])
            if expected and actual:
                correct = expected == actual
                fd_r = view_entry["result"].get("fraud_decision") or {}
                inv_r = view_entry["result"].get("investigation") or {}
                conf = fd_r.get("confidence") or inv_r.get("confidence")
                conf_str = (
                    f'<span class="fl-compare-conf">· conf {conf:.0%}</span>'
                    if conf is not None else ""
                )
                icon = "✓" if correct else "✗"
                cls = "fl-compare-correct" if correct else "fl-compare-wrong"
                exp_label = _OUTCOME_LABEL.get(expected, expected.upper())
                st.markdown(
                    f'<div class="fl-compare-row {cls}">'
                    f'{icon} Expected: {exp_label} {conf_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        is_running = bool(st.session_state.get("job_id"))
        if st.button(
            "⟳ Running…" if is_running else "Run Investigation",
            type="primary",
            use_container_width=True,
            disabled=is_running,
        ):
            job_id = submit_job(
                selected_case["transaction"],
                raw_mode=selected_case.get("raw_mode", False),
            )
            if job_id:
                st.session_state.job_id = job_id
                st.session_state.job_start_t = time.time()
                st.session_state.last_case = selected_case
                st.session_state.tier_hint = "unknown"
                st.session_state.view_entry = None
                st.rerun()

        history: list[dict[str, Any]] = st.session_state.get("history", [])
        if history:
            st.divider()
            st.markdown('<div class="fl-section-title">Recent Cases</div>', unsafe_allow_html=True)
            for i, h_entry in enumerate(history):
                h_case = h_entry.get("case") or {}
                h_result = h_entry.get("result") or {}
                h_outcome = _outcome_for_result(h_result)
                h_label = _OUTCOME_LABEL.get(h_outcome, h_outcome.upper())
                h_cid = h_case.get("case_id", f"run-{i+1}")
                h_tier = h_case.get("tier", "?")

                if st.button(
                    f"{h_label}  ·  {h_cid}  [{h_tier.upper()}]",
                    key=f"hist_{i}",
                    use_container_width=True,
                ):
                    st.session_state.view_entry = h_entry
                    st.session_state.selected_case_id = h_cid
                    st.rerun()


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------


def run_polling_loop() -> None:
    """Single-pass loading view — renders once then calls st.rerun() to poll again.

    Each script execution is a fresh render cycle, so no previous result content
    can bleed through. Avoids the ghost-content problem caused by while-True polling.
    """
    job_id: str = st.session_state.job_id
    t0: float = st.session_state.job_start_t
    tier_hint: str = st.session_state.get("tier_hint", "unknown")

    state = poll_job(job_id)
    if state is None:
        st.session_state.job_id = None
        st.rerun()
        return

    status = state["status"]

    if status == "investigating":
        tier_hint = "medium"
        st.session_state.tier_hint = "medium"
    elif status in ("critical", "sar"):
        tier_hint = "high"
        st.session_state.tier_hint = "high"

    st.markdown('<div class="fl-loading-wrapper">', unsafe_allow_html=True)
    render_pipeline_nodes(status, tier_hint)

    thought = state.get("thought") or ""
    if status in ("investigating", "critical", "synthesizing", "sar") and thought:
        render_thought_bubble(thought)

    completed = state.get("completed_tools") or []
    current = state.get("current_tool")
    if status in ("investigating", "critical") and (completed or current):
        render_tool_tracker(completed, current)

    stage = state.get("stage_label", "")
    if stage:
        st.markdown(
            f'<div class="fl-stage-label">↳ {stage}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if status == "done":
        elapsed_ms = (time.time() - t0) * 1000
        entry: dict[str, Any] = {
            "case": st.session_state.last_case,
            "result": state["result"],
            "elapsed_ms": elapsed_ms,
        }
        st.session_state.history.insert(0, entry)
        st.session_state.history = st.session_state.history[:_MAX_HISTORY]
        st.session_state.view_entry = entry
        st.session_state.job_id = None
        st.rerun()
        return

    if status == "error":
        st.error(f"Pipeline error: {state.get('error', 'Unknown error')}")
        st.session_state.job_id = None
        st.rerun()
        return

    if time.time() - t0 > _POLL_TIMEOUT:
        st.warning("Investigation timed out after 3 minutes.")
        st.session_state.job_id = None
        st.rerun()
        return

    time.sleep(_POLL_INTERVAL)
    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """Initialize session state with safe defaults."""
    defaults: dict[str, Any] = {
        "history": [],
        "job_id": None,
        "job_start_t": None,
        "last_case": None,
        "view_entry": None,
        "tier_hint": "unknown",
        "selected_case_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main() -> None:
    """Run the FraudLens compliance dashboard."""
    st.set_page_config(layout="wide", page_title="FraudLens", page_icon="🔍")
    load_css(Path(__file__).parent / "assets" / "style.css")
    _init_state()

    cases = load_preset_cases()
    render_sidebar(cases)

    if st.session_state.job_id:
        run_polling_loop()
    elif st.session_state.view_entry:
        render_results(st.session_state.view_entry)
    else:
        st.markdown(
            '<div style="text-align:center;padding:100px 0 40px;">'
            '<div style="font-size:1.9rem;font-weight:700;color:#2D3250;letter-spacing:-0.02em;">'
            'Select a case and run an investigation</div>'
            '<div style="font-size:0.85rem;color:#64748B;margin-top:10px;">'
            'FraudLens · AML / Fraud Detection · Compliance Dashboard'
            '</div></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
