"""
Ensemble page: Generate Data with Demo Mode.

Identical demo scenarios to the standard page but optimised for
the ensemble demo flow — generates data and pre-configures session
state for the ensemble scoring pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tgn_learn.generators import BankSimGenerator, GeneratorRegistry
from tgn_learn.generators.base import GeneratorConfig

# ---------------------------------------------------------------------------
# Demo Scenarios — pre-scripted fraud narratives
# ---------------------------------------------------------------------------

DEMO_SCENARIOS = {
    "Card Testing Ring": {
        "description": (
            "A compromised card is tested with 8 micro-transactions before "
            "a £2,400 purchase. The ensemble's TGN Memory detector catches the "
            "velocity burst while the Fund-Flow detector sees the chain."
        ),
        "config": GeneratorConfig(
            num_accounts=50, num_merchants=15,
            num_transactions=2000, fraud_rate=0.04, seed=42,
        ),
        "patterns": ["card_testing", "account_takeover"],
        "highlight_account": 7,
        "ensemble_focus": "TGN Memory + Fund-Flow Graph",
    },
    "Money Laundering Network": {
        "description": (
            "£180k layered through 6 intermediary accounts over 48 hours. "
            "The Fund-Flow Graph detector identifies the fan-out→fan-in topology "
            "invisible to feature-based models."
        ),
        "config": GeneratorConfig(
            num_accounts=80, num_merchants=20,
            num_transactions=3000, fraud_rate=0.03, seed=99,
        ),
        "patterns": ["money_laundering"],
        "ensemble_focus": "Fund-Flow Graph + RF Structural",
    },
    "Bust-Out Fraud": {
        "description": (
            "Account builds 3 months of legitimate history, then maxes out. "
            "The Drift Monitor catches baseline deviation that the RF "
            "Structural detector misses due to clean history."
        ),
        "config": GeneratorConfig(
            num_accounts=60, num_merchants=18,
            num_transactions=2500, fraud_rate=0.025, seed=77,
        ),
        "patterns": ["bust_out", "synthetic_identity"],
        "highlight_account": 15,
        "ensemble_focus": "Drift Monitor + TGN Memory",
    },
    "Multi-Pattern Attack": {
        "description": (
            "Combined scenario: card testing on one account + laundering "
            "chain across intermediaries. Tests ensemble's ability to "
            "detect concurrent independent attacks."
        ),
        "config": GeneratorConfig(
            num_accounts=100, num_merchants=25,
            num_transactions=4000, fraud_rate=0.05, seed=123,
        ),
        "patterns": ["card_testing", "money_laundering", "account_takeover"],
        "highlight_account": 7,
        "ensemble_focus": "All detectors active",
    },
}

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

st.header("Generate Data — Ensemble Mode")
st.caption("Pre-scripted fraud scenarios designed to showcase ensemble detector strengths.")

# ---------------------------------------------------------------------------
# Demo Mode (always on in ensemble pages)
# ---------------------------------------------------------------------------

st.info(
    "**Ensemble Demo Mode** — Each scenario is tuned to highlight different "
    "detectors in the ensemble. The focus detector(s) are listed with each scenario."
)

scenario_name = st.selectbox(
    "Select Fraud Scenario",
    options=list(DEMO_SCENARIOS.keys()),
    format_func=lambda x: f"🎬 {x}",
)
scenario = DEMO_SCENARIOS[scenario_name]

# Scenario details
st.markdown(f"**{scenario_name}:** {scenario['description']}")

col_info, col_focus = st.columns([2, 1])
with col_info:
    st.caption(
        f"Accounts: {scenario['config'].num_accounts} | "
        f"Merchants: {scenario['config'].num_merchants} | "
        f"Transactions: {scenario['config'].num_transactions} | "
        f"Fraud rate: {scenario['config'].fraud_rate:.0%}"
    )
with col_focus:
    st.markdown(f"🎯 **Detectors tested:** {scenario['ensemble_focus']}")

if "highlight_account" in scenario:
    st.markdown(f"🔍 **Watch account {scenario['highlight_account']}** — the attack target.")

st.divider()

# ---------------------------------------------------------------------------
# Generate Button
# ---------------------------------------------------------------------------

if st.button("▶️ Generate Ensemble Demo Scenario", type="primary", use_container_width=True):
    config = scenario["config"]
    gen = BankSimGenerator(config, patterns=scenario["patterns"])

    with st.spinner(f"Generating {scenario_name}..."):
        graph = gen.generate()

    st.session_state["graph"] = graph
    st.session_state["gen_config"] = config
    st.session_state["demo_scenario"] = scenario_name
    st.session_state["ensemble_focus"] = scenario["ensemble_focus"]
    if "highlight_account" in scenario:
        st.session_state["highlight_account"] = scenario["highlight_account"]

    st.success(
        f"Generated **{scenario_name}** — "
        f"{graph.num_edges} transactions, {graph.num_fraud} fraudulent. "
        f"Ready for ensemble scoring."
    )

# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------

if "graph" in st.session_state:
    graph = st.session_state["graph"]

    st.divider()
    st.subheader("Generated Network")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", graph.num_nodes)
    col2.metric("Transactions", graph.num_edges)
    col3.metric("Fraud", f"{graph.num_fraud} ({graph.fraud_rate:.1%})")
    col4.metric("Legitimate", graph.num_legit)

    # Temporal distribution
    edges = graph.edges
    timestamps = np.array([e.timestamp for e in edges])
    labels = np.array([e.label for e in edges])

    n_bins = 40
    bins = np.linspace(timestamps.min(), timestamps.max(), n_bins + 1)
    legit_hist, _ = np.histogram(timestamps[labels == 0], bins=bins)
    fraud_hist, _ = np.histogram(timestamps[labels == 1], bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers, y=legit_hist, name="Legitimate",
        marker_color="steelblue", opacity=0.7
    ))
    fig.add_trace(go.Bar(
        x=bin_centers, y=fraud_hist, name="Fraud",
        marker_color="crimson", opacity=0.9
    ))
    fig.update_layout(
        barmode="stack", height=250,
        xaxis_title="Time", yaxis_title="Count",
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Next steps
    st.markdown("**Next:** Go to **Train Model** → Load Pre-trained → **Score Transactions (Ensemble)**")
