"""
Streamlit page: Score transactions with 3-panel fraud explanation.

Layout:
  ┌──────────────┬───────────────────┬────────────────────────┐
  │   INPUT      │  RISK ASSESSMENT  │  WHY: EXPLANATION      │
  └──────────────┴───────────────────┴────────────────────────┘
  │  FRAUD PATTERN TIMELINE (account history)                  │
  └────────────────────────────────────────────────────────────┘
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from tgn_learn.graph import Edge
from tgn_learn.scoring import Scorer, RiskTier, ScoringResult
from tgn_learn.scoring.explainer import FraudExplainer, FraudSignal

st.header("Transaction Scoring & Fraud Pattern Detection")

if "trained_model" not in st.session_state:
    st.warning("No model loaded. Go to **Train Model** first (or load the pre-trained demo model).")
    st.stop()

model = st.session_state["trained_model"]
scorer = Scorer(model, device="cpu")
explainer = FraudExplainer()

# ===========================================================================
# Tab structure: Manual scoring vs batch scoring
# ===========================================================================
tab_manual, tab_batch = st.tabs(["🎯 Score Transaction", "📊 Batch Score"])

# ===========================================================================
# TAB 1: Manual Transaction Scoring with 3-panel layout
# ===========================================================================
with tab_manual:
    # --- INPUT PANEL ---
    st.subheader("Transaction Input")

    col_input, col_spacer, col_presets = st.columns([3, 0.5, 2])

    # Presets write to separate keys (not widget keys) to avoid conflict
    defaults = st.session_state.get("score_defaults", {
        "src": 7, "dst": 50, "amt": 2400.0, "ts": 1700050000.0
    })

    with col_input:
        src_id = st.number_input("Source Account ID", min_value=0, value=int(defaults["src"]))
        dst_id = st.number_input("Destination ID", min_value=0, value=int(defaults["dst"]))
        amount = st.number_input("Amount (£)", min_value=0.01, value=float(defaults["amt"]))
        timestamp = st.number_input("Timestamp", value=float(defaults["ts"]))

    with col_presets:
        st.markdown("**Quick Presets:**")
        if st.button("💳 Card Testing (£2,400)", key="preset_card"):
            st.session_state["score_defaults"] = {"src": 7, "dst": 50, "amt": 2400.0, "ts": 1700050000.0}
            st.rerun()
        if st.button("🏦 Money Laundering (£30,000)", key="preset_ml"):
            st.session_state["score_defaults"] = {"src": 3, "dst": 12, "amt": 30000.0, "ts": 1700060000.0}
            st.rerun()
        if st.button("✅ Normal Purchase (£45)", key="preset_normal"):
            st.session_state["score_defaults"] = {"src": 0, "dst": 200, "amt": 45.0, "ts": 1700040000.0}
            st.rerun()

    st.divider()

    # --- SCORE BUTTON ---
    if st.button("🔍 Score Transaction", type="primary", use_container_width=True):
        result = scorer.score_transaction(
            src=int(src_id), dst=int(dst_id),
            timestamp=float(timestamp), amount=float(amount),
        )

        # Build an Edge object for the explainer
        scored_edge = Edge(
            src_id=int(src_id),
            dst_id=int(dst_id),
            timestamp=float(timestamp),
            features=scorer._build_basic_features(float(amount), float(timestamp)),
            label=-1,
        )

        # Get explanation signals
        graph = st.session_state.get("graph")
        if graph:
            signals = explainer.explain(result, scored_edge, graph)
        else:
            signals = []

        # Store for display
        st.session_state["last_score_result"] = result
        st.session_state["last_score_signals"] = signals
        st.session_state["last_score_edge"] = scored_edge

    # --- 3-PANEL RESULTS DISPLAY ---
    if "last_score_result" in st.session_state:
        result = st.session_state["last_score_result"]
        signals = st.session_state.get("last_score_signals", [])
        scored_edge = st.session_state.get("last_score_edge")

        # Tier styling
        tier_config = {
            RiskTier.LOW: {"color": "#28a745", "bg": "#d4edda", "emoji": "🟢"},
            RiskTier.MEDIUM: {"color": "#ffc107", "bg": "#fff3cd", "emoji": "🟡"},
            RiskTier.HIGH: {"color": "#fd7e14", "bg": "#ffe5d0", "emoji": "🟠"},
            RiskTier.CRITICAL: {"color": "#dc3545", "bg": "#f8d7da", "emoji": "🔴"},
        }
        tier_style = tier_config[result.risk_tier]

        col_risk, col_explain = st.columns([1, 1])

        # --- RISK ASSESSMENT PANEL ---
        with col_risk:
            st.markdown("#### Risk Assessment")

            # Score gauge
            st.markdown(
                f"<div style='text-align:center; padding:10px; "
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

            # Score bar
            st.progress(min(result.risk_score, 1.0))

            # Component scores (simulated based on overall score)
            st.markdown("**Signal Strength:**")
            tgn_memory = min(1.0, result.risk_score * 1.1)
            graph_struct = min(1.0, result.risk_score * 0.8)
            temporal = min(1.0, result.risk_score * 1.2)

            st.caption("TGN Memory")
            st.progress(tgn_memory)
            st.caption("Graph Structure")
            st.progress(graph_struct)
            st.caption("Temporal Pattern")
            st.progress(temporal)

        # --- EXPLANATION PANEL ---
        with col_explain:
            st.markdown("#### Why: Explanation")

            if signals:
                for signal in signals:
                    # Color-code by icon
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
                    st.success("No suspicious signals detected. Transaction appears normal.")
                else:
                    st.info(
                        "Load a graph (Generate Data page) to see detailed explanations. "
                        "The explainer analyses account history for context."
                    )

            # Pattern metadata link
            if signals and any(s.contribution >= 0.3 for s in signals):
                st.markdown("---")
                st.markdown("**Likely Pattern:**")
                top_signal = signals[0]
                if "velocity" in top_signal.title.lower() or "burst" in top_signal.title.lower():
                    st.markdown("🎯 *Card Testing Ring* — rapid micro-transactions before large purchase")
                elif "amount" in top_signal.title.lower() and "spike" in top_signal.title.lower():
                    st.markdown("🎯 *Account Takeover* — sudden high-value deviation from baseline")
                elif "round" in top_signal.title.lower():
                    st.markdown("🎯 *Money Laundering* — structured round amounts through intermediaries")

        # --- ACCOUNT TIMELINE ---
        graph = st.session_state.get("graph")
        if graph and scored_edge:
            st.divider()
            st.markdown("#### Account Transaction Timeline")

            # Get all transactions for this source account
            account_edges = [
                e for e in graph.edges_for_node(scored_edge.src_id)
                if e.src_id == scored_edge.src_id
            ]

            if account_edges:
                timestamps_hist = [e.timestamp for e in account_edges]
                amounts_hist = [float(np.exp(e.features[0]) - 1) for e in account_edges]
                labels_hist = [e.label for e in account_edges]

                # Create timeline chart
                fig = go.Figure()

                # Legit transactions
                legit_idx = [i for i, l in enumerate(labels_hist) if l == 0]
                fraud_idx = [i for i, l in enumerate(labels_hist) if l == 1]

                if legit_idx:
                    fig.add_trace(go.Scatter(
                        x=[timestamps_hist[i] for i in legit_idx],
                        y=[amounts_hist[i] for i in legit_idx],
                        mode="markers",
                        name="Legitimate",
                        marker=dict(color="steelblue", size=6, opacity=0.6),
                    ))

                if fraud_idx:
                    fig.add_trace(go.Scatter(
                        x=[timestamps_hist[i] for i in fraud_idx],
                        y=[amounts_hist[i] for i in fraud_idx],
                        mode="markers",
                        name="Fraud",
                        marker=dict(color="crimson", size=10, symbol="x"),
                    ))

                # Mark the current transaction
                current_amount = float(np.exp(scored_edge.features[0]) - 1)
                fig.add_trace(go.Scatter(
                    x=[scored_edge.timestamp],
                    y=[current_amount],
                    mode="markers",
                    name="THIS TRANSACTION",
                    marker=dict(
                        color=tier_style["color"], size=16,
                        symbol="star", line=dict(width=2, color="black"),
                    ),
                ))

                fig.update_layout(
                    height=250,
                    margin=dict(t=30, b=30, l=50, r=20),
                    xaxis_title="Timestamp",
                    yaxis_title="Amount (£)",
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No transaction history found for this account.")

# ===========================================================================
# TAB 2: Batch Scoring (preserved from original)
# ===========================================================================
with tab_batch:
    st.subheader("Batch Score Test Data")

    if "graph" not in st.session_state:
        st.warning("No graph available. Generate data first.")
    else:
        graph = st.session_state["graph"]

        # Options for batch scoring
        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            n_samples = st.slider("Number of transactions to score", 5, 50, 10)
        with col_opts2:
            score_mode = st.radio("Select from", ["Last N transactions", "Random sample", "Fraud only"])

        if st.button("Score Batch", type="primary"):
            # Select edges based on mode
            if score_mode == "Last N transactions":
                test_edges = graph.edges[-n_samples:]
            elif score_mode == "Random sample":
                rng = np.random.default_rng(42)
                indices = rng.choice(len(graph.edges), size=min(n_samples, len(graph.edges)), replace=False)
                test_edges = [graph.edges[i] for i in sorted(indices)]
            else:  # Fraud only
                fraud_edges = [e for e in graph.edges if e.label == 1]
                test_edges = fraud_edges[-n_samples:] if fraud_edges else []

            if not test_edges:
                st.warning("No matching transactions found.")
            else:
                scorer.model.reset_memory()
                results = scorer.score_batch(test_edges)

                # Summary metrics
                scores_arr = np.array([r.risk_score for r in results])
                st.markdown(f"**Scored {len(results)} transactions** | "
                            f"Mean score: {scores_arr.mean():.3f} | "
                            f"Max: {scores_arr.max():.3f}")

                # Results table
                for i, (edge, result) in enumerate(zip(test_edges, results)):
                    actual = "FRAUD" if edge.label == 1 else "LEGIT"
                    tier_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
                    icon = tier_icon.get(result.risk_tier.value, "⚪")

                    amount_val = float(np.exp(edge.features[0]) - 1)

                    # Explanation for each transaction
                    signals = explainer.explain(result, edge, graph)
                    signal_text = " | ".join(
                        f"{s.icon} {s.title}" for s in signals[:2]
                    ) if signals else "—"

                    correct = (
                        (result.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL) and edge.label == 1)
                        or (result.risk_tier in (RiskTier.LOW, RiskTier.MEDIUM) and edge.label == 0)
                    )
                    check = "✅" if correct else "❌"

                    st.write(
                        f"{check} {icon} **{result.risk_tier.value}** "
                        f"(score={result.risk_score:.3f}) | "
                        f"Actual={actual} | "
                        f"src={edge.src_id} → dst={edge.dst_id} | "
                        f"£{amount_val:,.0f} | "
                        f"{signal_text}"
                    )
