"""
Streamlit page: Generate synthetic fraud networks.

Lets users configure and generate synthetic transaction networks
with various fraud patterns injected. Includes a Demo Mode with
pre-scripted fraud scenarios for live presentations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from tgn_learn.generators import BankSimGenerator, PaySimGenerator, GeneratorRegistry
from tgn_learn.generators.base import GeneratorConfig

# ---------------------------------------------------------------------------
# Demo Mode — Pre-scripted scenarios for live demos
# ---------------------------------------------------------------------------
DEMO_SCENARIOS = {
    "Card Testing Ring": {
        "description": "A compromised card is tested with 8 micro-transactions before a £2,400 purchase",
        "config": GeneratorConfig(
            num_accounts=50, num_merchants=15,
            num_transactions=2000, fraud_rate=0.04, seed=42,
        ),
        "patterns": ["card_testing", "account_takeover"],
        "highlight_account": 7,
    },
    "Money Laundering Network": {
        "description": "£180k layered through 6 intermediary accounts over 48 hours",
        "config": GeneratorConfig(
            num_accounts=80, num_merchants=20,
            num_transactions=3000, fraud_rate=0.03, seed=99,
        ),
        "patterns": ["money_laundering"],
    },
    "Bust-Out Fraud": {
        "description": "Account builds credit history over 3 months, then maxes out instantly",
        "config": GeneratorConfig(
            num_accounts=60, num_merchants=18,
            num_transactions=2500, fraud_rate=0.025, seed=77,
        ),
        "patterns": ["bust_out", "synthetic_identity"],
    },
}

st.header("Generate Synthetic Fraud Data")

st.markdown("""
Generate a synthetic transaction network with configurable fraud patterns.
The generated graph will be available for exploration, training, and scoring.
""")

# --- Demo Mode ---
demo_mode = st.toggle("🎯 Demo Mode", value=False, help="Use pre-scripted scenarios for live presentations")

if demo_mode:
    st.info("**Demo Mode** — Select a pre-scripted fraud scenario with named accounts and a narrated fraud story.")

    scenario_name = st.selectbox(
        "Select Scenario",
        options=list(DEMO_SCENARIOS.keys()),
        format_func=lambda x: f"🎬 {x}",
    )
    scenario = DEMO_SCENARIOS[scenario_name]

    st.markdown(f"**{scenario_name}:** {scenario['description']}")
    st.caption(
        f"Accounts: {scenario['config'].num_accounts} | "
        f"Merchants: {scenario['config'].num_merchants} | "
        f"Transactions: {scenario['config'].num_transactions} | "
        f"Fraud rate: {scenario['config'].fraud_rate:.0%} | "
        f"Patterns: {', '.join(scenario['patterns'])}"
    )

    if "highlight_account" in scenario:
        st.markdown(f"🔍 **Watch account {scenario['highlight_account']}** — this is the compromised card.")

    if st.button("▶️ Generate Demo Scenario", type="primary"):
        config = scenario["config"]
        gen = BankSimGenerator(config, patterns=scenario["patterns"])
        graph = gen.generate()
        st.session_state["graph"] = graph
        st.session_state["gen_config"] = config
        st.session_state["demo_scenario"] = scenario_name
        if "highlight_account" in scenario:
            st.session_state["highlight_account"] = scenario["highlight_account"]
        st.success(f"Generated **{scenario_name}** scenario — {graph.num_edges} transactions, {graph.num_fraud} fraudulent.")

else:
    # --- Quick Start ---
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Quick Start", type="primary"):
            config = GeneratorConfig(
                num_accounts=200, num_merchants=30,
                num_transactions=5000, fraud_rate=0.03, seed=42,
            )
            gen = BankSimGenerator(config)
            graph = gen.generate()
            st.session_state["graph"] = graph
            st.session_state["gen_config"] = config
            st.success("Generated 5000 transactions with 3% fraud rate!")

    with col2:
        st.caption("One-click generation with good defaults (200 accounts, 5000 txns, 3% fraud)")

    st.divider()

    # --- Configuration ---
    st.subheader("Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        generator_type = st.selectbox(
            "Generator",
            options=["banksim", "paysim"],
            help="BankSim: merchant fraud patterns. PaySim: mobile money patterns.",
        )
        num_accounts = st.slider("Accounts", 50, 1000, 200, step=50)
        num_merchants = st.slider("Merchants", 10, 100, 30, step=10)

    with col2:
        num_transactions = st.slider("Transactions", 500, 20000, 5000, step=500)
        fraud_rate = st.slider("Fraud Rate (%)", 1, 20, 3) / 100.0
        seed = st.number_input("Random Seed", value=42, step=1)

    with col3:
        # Pattern selection (BankSim)
        if generator_type == "banksim":
            st.markdown("**Fraud Patterns:**")
            patterns = []
            if st.checkbox("Account Takeover", value=True):
                patterns.append("account_takeover")
            if st.checkbox("Card Testing", value=True):
                patterns.append("card_testing")
            if st.checkbox("Money Laundering", value=True):
                patterns.append("money_laundering")
            if st.checkbox("Synthetic Identity", value=True):
                patterns.append("synthetic_identity")
            if st.checkbox("Bust Out", value=True):
                patterns.append("bust_out")
        else:
            st.markdown("**Fraud Patterns:**")
            patterns = []
            if st.checkbox("Money Mule", value=True):
                patterns.append("money_mule")
            if st.checkbox("Structuring", value=True):
                patterns.append("structuring")
            if st.checkbox("Account Drainage", value=True):
                patterns.append("account_drainage")

    # --- Generate Button ---
    if st.button("Generate Network"):
        config = GeneratorConfig(
            num_accounts=num_accounts,
            num_merchants=num_merchants,
            num_transactions=num_transactions,
            fraud_rate=fraud_rate,
            seed=int(seed),
        )

        gen = GeneratorRegistry.create(generator_type, config, patterns=patterns or None)

        with st.spinner("Generating..."):
            graph = gen.generate()

        st.session_state["graph"] = graph
        st.session_state["gen_config"] = config
        st.success(f"Generated {graph.num_edges} transactions!")

# --- Display Results ---
if "graph" in st.session_state:
    graph = st.session_state["graph"]

    st.divider()
    st.subheader("Generated Network Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", graph.num_nodes)
    col2.metric("Transactions", graph.num_edges)
    col3.metric("Fraud", f"{graph.num_fraud} ({graph.fraud_rate:.1%})")
    col4.metric("Legitimate", graph.num_legit)

    # Node type breakdown
    st.markdown("**Node Types:**")
    types = graph.node_types()
    st.write(types)

    # Temporal distribution
    st.subheader("Temporal Distribution")

    edges = graph.edges
    timestamps = np.array([e.timestamp for e in edges])
    labels = np.array([e.label for e in edges])

    # Bin into time windows
    n_bins = 50
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
        barmode="stack",
        xaxis_title="Timestamp",
        yaxis_title="Transaction Count",
        height=300,
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Amount distribution
    st.subheader("Amount Distribution")
    amounts = np.array([e.features[0] for e in edges])  # log_amount
    df_amounts = pd.DataFrame({
        "log_amount": amounts,
        "label": ["Fraud" if l == 1 else "Legit" for l in labels]
    })
    fig2 = px.histogram(
        df_amounts, x="log_amount", color="label",
        nbins=40, barmode="overlay", opacity=0.7,
        color_discrete_map={"Legit": "steelblue", "Fraud": "crimson"},
    )
    fig2.update_layout(height=300, margin=dict(t=30, b=30))
    st.plotly_chart(fig2, use_container_width=True)
