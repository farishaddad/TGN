"""
Streamlit page: Score individual transactions.
"""

import streamlit as st
import numpy as np

from tgn_learn.graph import Edge
from tgn_learn.scoring import Scorer, RiskTier

st.header("Score Transactions")

if "trained_model" not in st.session_state:
    st.warning("No model trained yet. Go to **Train Model** first.")
    st.stop()

model = st.session_state["trained_model"]
scorer = Scorer(model, device="cpu")

st.markdown("Score individual transactions or select from generated test data.")

tab1, tab2 = st.tabs(["Manual Input", "Score Test Data"])

with tab1:
    st.subheader("Manual Transaction Input")
    col1, col2 = st.columns(2)

    with col1:
        src_id = st.number_input("Source Account ID", min_value=0, value=0)
        dst_id = st.number_input("Destination ID", min_value=0, value=50)

    with col2:
        amount = st.number_input("Amount ($)", min_value=0.01, value=500.0)
        timestamp = st.number_input("Timestamp", value=1700050000.0)

    if st.button("Score Transaction", type="primary"):
        result = scorer.score_transaction(
            src=int(src_id), dst=int(dst_id),
            timestamp=float(timestamp), amount=float(amount),
        )

        # Display result with color-coded tier
        tier_colors = {
            RiskTier.LOW: "green",
            RiskTier.MEDIUM: "orange",
            RiskTier.HIGH: "red",
            RiskTier.CRITICAL: "darkred",
        }
        color = tier_colors[result.risk_tier]

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{result.risk_score:.4f}")
        col2.markdown(f"**Risk Tier:** :{color}[{result.risk_tier.value}]")
        col3.metric("Confidence", f"{result.confidence_lower:.3f} - {result.confidence_upper:.3f}")

        # Score gauge
        st.progress(min(result.risk_score, 1.0))

with tab2:
    st.subheader("Score Test Data")

    if "graph" not in st.session_state:
        st.warning("No graph available.")
    else:
        graph = st.session_state["graph"]
        # Get last 10 edges as test samples
        test_edges = graph.edges[-10:]

        if st.button("Score Last 10 Transactions"):
            scorer.model.reset_memory()
            results = scorer.score_batch(test_edges)

            for i, (edge, result) in enumerate(zip(test_edges, results)):
                actual = "FRAUD" if edge.label == 1 else "LEGIT"
                tier_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
                icon = tier_icon.get(result.risk_tier.value, "⚪")

                st.write(
                    f"{icon} **{result.risk_tier.value}** "
                    f"(score={result.risk_score:.3f}) | "
                    f"Actual={actual} | "
                    f"src={edge.src_id} → dst={edge.dst_id} | "
                    f"amount={np.exp(edge.features[0]) - 1:.0f}"
                )
