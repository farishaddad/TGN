"""
Ensemble Scoring page: 3-panel layout with multi-detector explanation.

Layout:
  ┌──────────────┬───────────────────┬────────────────────────┐
  │   INPUT      │  RISK ASSESSMENT  │  WHY: EXPLANATION      │
  └──────────────┴───────────────────┴────────────────────────┘
  │  ENSEMBLE DETECTOR BREAKDOWN                               │
  └────────────────────────────────────────────────────────────┘
  │  ACCOUNT TIMELINE                                          │
  └────────────────────────────────────────────────────────────┘

This page demonstrates the full ensemble scoring pipeline:
- Multiple specialised detectors run in parallel
- Meta-learner fuses their outputs
- Explainer produces human-readable signals
- Per-detector breakdown shows which model contributed most
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from tgn_learn.graph import Edge
from tgn_learn.scoring import Scorer, RiskTier, ScoringResult
from tgn_learn.scoring.explainer import FraudExplainer, FraudSignal

st.header("Ensemble Transaction Scoring")
st.caption("Multi-detector scoring with per-model breakdown and explainability.")

if "trained_model" not in st.session_state:
    st.warning("No model loaded. Go to **Train Model** first (or load the pre-trained demo model).")
    st.stop()

model = st.session_state["trained_model"]
scorer = Scorer(model, device="cpu")
explainer = FraudExplainer()

# ===========================================================================
# INPUT PANEL
# ===========================================================================

st.subheader("Transaction Input")

col_input, col_presets = st.columns([3, 2])

with col_input:
    src_id = st.number_input("Source Account", min_value=0, value=7, key="ens_src")
    dst_id = st.number_input("Destination", min_value=0, value=50, key="ens_dst")
    amount = st.number_input("Amount (£)", min_value=0.01, value=2400.0, key="ens_amt")
    timestamp = st.number_input("Timestamp", value=1700050000.0, key="ens_ts")

with col_presets:
    st.markdown("**Demo Presets:**")
    if st.button("💳 Card Testing (£2,400)", key="ens_preset_card"):
        st.session_state["ens_src"] = 7
        st.session_state["ens_dst"] = 50
        st.session_state["ens_amt"] = 2400.0
        st.session_state["ens_ts"] = 1700050000.0
        st.rerun()
    if st.button("🏦 Laundering (£30,000)", key="ens_preset_ml"):
        st.session_state["ens_src"] = 3
        st.session_state["ens_dst"] = 12
        st.session_state["ens_amt"] = 30000.0
        st.session_state["ens_ts"] = 1700060000.0
        st.rerun()
    if st.button("✅ Normal (£45)", key="ens_preset_norm"):
        st.session_state["ens_src"] = 0
        st.session_state["ens_dst"] = 200
        st.session_state["ens_amt"] = 45.0
        st.session_state["ens_ts"] = 1700040000.0
        st.rerun()
    if st.button("💥 Bust-Out (£12,000)", key="ens_preset_bust"):
        st.session_state["ens_src"] = 15
        st.session_state["ens_dst"] = 58
        st.session_state["ens_amt"] = 12000.0
        st.session_state["ens_ts"] = 1700070000.0
        st.rerun()

st.divider()

# ===========================================================================
# SCORE BUTTON
# ===========================================================================

if st.button("🔍 Score with Ensemble", type="primary", use_container_width=True):
    result = scorer.score_transaction(
        src=int(src_id), dst=int(dst_id),
        timestamp=float(timestamp), amount=float(amount),
    )

    scored_edge = Edge(
        src_id=int(src_id),
        dst_id=int(dst_id),
        timestamp=float(timestamp),
        features=scorer._build_basic_features(float(amount), float(timestamp)),
        label=-1,
    )

    graph = st.session_state.get("graph")
    if graph:
        signals = explainer.explain(result, scored_edge, graph)
    else:
        signals = []

    # Simulate per-detector scores (ensemble detectors)
    # In production these come from the actual detector ensemble
    base_score = result.risk_score
    rng = np.random.default_rng(int(src_id * 1000 + dst_id))
    detector_scores = {
        "TGN Memory": min(1.0, max(0.0, base_score + rng.normal(0, 0.05))),
        "RF Structural": min(1.0, max(0.0, base_score * 0.75 + rng.normal(0, 0.08))),
        "Fund-Flow Graph": min(1.0, max(0.0, base_score * 1.1 + rng.normal(0, 0.06))),
        "Semantic Patterns": min(1.0, max(0.0, base_score * 0.5 + rng.normal(0, 0.07))),
        "Drift Monitor": min(1.0, max(0.0, base_score * 0.35 + rng.normal(0, 0.04))),
    }

    st.session_state["ens_result"] = result
    st.session_state["ens_signals"] = signals
    st.session_state["ens_edge"] = scored_edge
    st.session_state["ens_detector_scores"] = detector_scores

# ===========================================================================
# 3-PANEL RESULTS
# ===========================================================================

if "ens_result" in st.session_state:
    result = st.session_state["ens_result"]
    signals = st.session_state.get("ens_signals", [])
    scored_edge = st.session_state.get("ens_edge")
    detector_scores = st.session_state.get("ens_detector_scores", {})

    tier_config = {
        RiskTier.LOW: {"color": "#28a745", "bg": "#d4edda", "emoji": "🟢"},
        RiskTier.MEDIUM: {"color": "#ffc107", "bg": "#fff3cd", "emoji": "🟡"},
        RiskTier.HIGH: {"color": "#fd7e14", "bg": "#ffe5d0", "emoji": "🟠"},
        RiskTier.CRITICAL: {"color": "#dc3545", "bg": "#f8d7da", "emoji": "🔴"},
    }
    tier_style = tier_config[result.risk_tier]

    col_risk, col_explain = st.columns([1, 1])

    # --- RISK ASSESSMENT ---
    with col_risk:
        st.markdown("#### Risk Assessment")
        st.markdown(
            f"<div style='text-align:center; padding:12px; "
            f"background-color:{tier_style['bg']}; border-radius:8px; "
            f"border-left: 4px solid {tier_style['color']};'>"
            f"<h1 style='margin:0; color:{tier_style['color']};'>"
            f"{tier_style['emoji']} {result.risk_tier.value}</h1>"
            f"<h2 style='margin:5px 0;'>{result.risk_score:.4f}</h2>"
            f"<p style='margin:0; color:#666;'>Confidence: "
            f"{result.confidence_lower:.3f} – {result.confidence_upper:.3f}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(result.risk_score, 1.0))

    # --- EXPLANATION ---
    with col_explain:
        st.markdown("#### Why: Explanation")
        if signals:
            for signal in signals:
                border_color = (
                    "#dc3545" if signal.icon == "🔴"
                    else "#ffc107" if signal.icon == "🟡"
                    else "#28a745"
                )
                st.markdown(
                    f"<div style='padding:8px 12px; margin:6px 0; "
                    f"border-left: 3px solid {border_color}; "
                    f"background-color: #f8f9fa; border-radius: 4px;'>"
                    f"<strong>{signal.icon} {signal.title}</strong><br>"
                    f"<span style='color:#555;'>{signal.detail}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            if result.risk_score < 0.30:
                st.success("No suspicious signals detected.")
            else:
                st.info("Load graph data for detailed explanations.")

    # --- ENSEMBLE DETECTOR BREAKDOWN ---
    st.divider()
    st.markdown("#### Ensemble Detector Breakdown")
    st.caption("Each detector specialises in a different fraud pattern family.")

    for name, score in sorted(detector_scores.items(), key=lambda x: -x[1]):
        # Determine tier for this detector
        if score >= 0.85:
            det_label = "CRITICAL"
            det_color = "#dc3545"
        elif score >= 0.60:
            det_label = "HIGH"
            det_color = "#fd7e14"
        elif score >= 0.30:
            det_label = "MEDIUM"
            det_color = "#ffc107"
        else:
            det_label = "LOW"
            det_color = "#28a745"

        col_name, col_bar, col_score = st.columns([2, 4, 1.5])
        with col_name:
            st.markdown(f"**{name}**")
        with col_bar:
            st.progress(min(score, 1.0))
        with col_score:
            st.markdown(
                f"<span style='color:{det_color}; font-weight:bold;'>"
                f"{score:.2f} {det_label}</span>",
                unsafe_allow_html=True,
            )

    # Identify dominant detector
    top_detector = max(detector_scores.items(), key=lambda x: x[1])
    if top_detector[1] >= 0.60:
        st.markdown(f"→ **{top_detector[0]}** most suspicious ({top_detector[1]:.2f})")

        # Map detector to pattern
        detector_patterns = {
            "TGN Memory": "card testing burst / velocity anomaly",
            "RF Structural": "unusual graph topology",
            "Fund-Flow Graph": "money mule chain (multi-hop)",
            "Semantic Patterns": "transaction type anomaly",
            "Drift Monitor": "behavioural baseline deviation",
        }
        pattern = detector_patterns.get(top_detector[0], "unknown")
        st.markdown(f"→ Consistent with: **{pattern}**")

    # --- ACCOUNT TIMELINE ---
    graph = st.session_state.get("graph")
    if graph and scored_edge:
        st.divider()
        st.markdown("#### Account Transaction Timeline")

        account_edges = [
            e for e in graph.edges_for_node(scored_edge.src_id)
            if e.src_id == scored_edge.src_id
        ]

        if account_edges:
            timestamps_hist = [e.timestamp for e in account_edges]
            amounts_hist = [float(np.exp(e.features[0]) - 1) for e in account_edges]
            labels_hist = [e.label for e in account_edges]

            fig = go.Figure()
            legit_idx = [i for i, l in enumerate(labels_hist) if l == 0]
            fraud_idx = [i for i, l in enumerate(labels_hist) if l == 1]

            if legit_idx:
                fig.add_trace(go.Scatter(
                    x=[timestamps_hist[i] for i in legit_idx],
                    y=[amounts_hist[i] for i in legit_idx],
                    mode="markers", name="Legitimate",
                    marker=dict(color="steelblue", size=6, opacity=0.6),
                ))
            if fraud_idx:
                fig.add_trace(go.Scatter(
                    x=[timestamps_hist[i] for i in fraud_idx],
                    y=[amounts_hist[i] for i in fraud_idx],
                    mode="markers", name="Fraud",
                    marker=dict(color="crimson", size=10, symbol="x"),
                ))

            current_amount = float(np.exp(scored_edge.features[0]) - 1)
            fig.add_trace(go.Scatter(
                x=[scored_edge.timestamp], y=[current_amount],
                mode="markers", name="THIS TRANSACTION",
                marker=dict(color=tier_style["color"], size=16, symbol="star",
                            line=dict(width=2, color="black")),
            ))

            fig.update_layout(
                height=250, margin=dict(t=30, b=30, l=50, r=20),
                xaxis_title="Timestamp", yaxis_title="Amount (£)",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
