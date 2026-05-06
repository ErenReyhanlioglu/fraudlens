"""FraudLens — Hugging Face Spaces demo.

Standalone Streamlit app with pre-computed fixtures (no live backend).
Reuses all render_* functions from demo/app.py so the UI is identical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Make demo/ importable regardless of CWD (HF Spaces runs from repo root)
_DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(_DEMO_DIR))

from app import (  # noqa: E402
    _OUTCOME_COLOR,
    _OUTCOME_LABEL,
    _TIER_COLOR,
    load_css,
    render_decision_detail,
    render_investigation_timeline,
    render_ml_section,
    render_observability_footer,
    render_results,
    render_sar_section,
    render_verdict_hero,
)
from fixtures import FIXTURES  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="FraudLens Demo", page_icon="🔍")
load_css(_DEMO_DIR / "assets" / "style.css")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _outcome_badge(outcome: str) -> str:
    label = _OUTCOME_LABEL.get(outcome, outcome.upper())
    color = _OUTCOME_COLOR.get(outcome, "#64748B")
    return f'<span style="color:{color};font-weight:700;">{label}</span>'


with st.sidebar:
    st.markdown(
        '<div class="fl-sidebar-header">'
        '<div style="font-size:1.35rem;font-weight:800;color:#E2E8F0;letter-spacing:-0.02em;">'
        "FraudLens</div>"
        '<div style="font-size:0.72rem;color:#64748B;margin-top:3px;">'
        "AML · Compliance · Intelligence</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.2);'
        'border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:0.78rem;color:#94A3B8;">'
        "⚡ <strong style='color:#A5B4FC;'>Demo mode</strong> — pre-computed results, no live backend."
        "</div>",
        unsafe_allow_html=True,
    )

    _TIER_ORDER = {"high": 0, "medium": 1, "low": 2}
    sorted_fixtures = sorted(
        FIXTURES,
        key=lambda f: (_TIER_ORDER.get(f["case"]["tier"], 99), f["case"]["case_id"]),
    )

    labels = {
        f["case"]["case_id"]: (
            f"[{f['case']['tier'].upper()}]  {f['case']['case_id']} — "
            + (
                f["case"]["description"][:38] + "…"
                if len(f["case"]["description"]) > 38
                else f["case"]["description"]
            )
        )
        for f in sorted_fixtures
    }
    ids = list(labels.keys())

    if st.session_state.get("selected_case_id") not in ids:
        st.session_state.selected_case_id = ids[0]

    selected_id: str = st.selectbox(
        "Case",
        ids,
        format_func=lambda k: labels[k],
        key="selected_case_id",
        label_visibility="collapsed",
    )

    entry = next(f for f in FIXTURES if f["case"]["case_id"] == selected_id)
    case = entry["case"]
    result = entry["result"]

    tier = case.get("tier", "unknown")
    tier_color = _TIER_COLOR.get(tier, "#64748B")
    tags_html = " ".join(
        f'<span class="fl-tag">{t}</span>' for t in (case.get("tags") or [])
    )
    fd = result.get("fraud_decision") or {}
    outcome = fd.get("outcome") or ("approve" if result.get("triage_action") == "approve" else "unknown")

    st.markdown(
        f'<div class="fl-case-card">'
        f'<span class="fl-tier-badge" style="background:{tier_color};">{tier.upper()}</span>'
        f'<div style="font-size:0.81rem;color:#CBD5E1;margin-bottom:7px;">{case["description"]}</div>'
        f"{tags_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    expected = case.get("expected_outcome", "")
    correct = expected == outcome
    icon = "✓" if correct else "✗"
    cls = "fl-compare-correct" if correct else "fl-compare-wrong"
    exp_label = _OUTCOME_LABEL.get(expected, expected.upper())
    st.markdown(
        f'<div class="fl-compare-row {cls}">{icon} Expected: {exp_label}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main — render pre-computed result
# ---------------------------------------------------------------------------

render_results(entry)
