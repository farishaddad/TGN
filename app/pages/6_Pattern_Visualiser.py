"""
Streamlit page: Fraud Pattern Visualiser.

Shows fraud patterns as step-through graph animations.
The presenter clicks through each step of an attack, watching
the graph evolve and the TGN memory respond.

Uses plotly + networkx for graph layout.
"""

import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go

st.header("Fraud Pattern Visualiser")
st.markdown("Step through a fraud attack and see how TGN detects it in real time.")

# ---------------------------------------------------------------------------
# Pattern Definitions (self-contained — does not require generated data)
# ---------------------------------------------------------------------------

PATTERNS = {
    "Card Testing Ring": {
        "description": (
            "A stolen card is validated with rapid micro-transactions before "
            "a large fraudulent purchase. TGN's memory tracks the burst and "
            "flags the final transaction as CRITICAL."
        ),
        "steps": [
            {
                "title": "Step 1/4: Normal baseline",
                "narrative": "Account 7 has a normal transaction history — coffee, groceries, fuel.",
                "edges": [
                    (7, 50, 12.50, False),
                    (7, 51, 45.00, False),
                    (7, 52, 38.00, False),
                    (7, 50, 8.90, False),
                ],
                "highlight": [],
                "memory_level": 0.05,
            },
            {
                "title": "Step 2/4: Card testing begins",
                "narrative": (
                    "The attacker starts testing the stolen card with micro-transactions "
                    "(£0.50–£1.00) at multiple merchants. Each one looks harmless alone."
                ),
                "edges": [
                    (7, 53, 0.50, True),
                    (7, 54, 0.75, True),
                    (7, 55, 1.00, True),
                ],
                "highlight": [(7, 53), (7, 54), (7, 55)],
                "memory_level": 0.35,
            },
            {
                "title": "Step 3/4: TGN memory rises — anomaly detected",
                "narrative": (
                    "Five more micro-transactions in under 4 minutes. TGN's memory "
                    "vector for account 7 now encodes a velocity anomaly. The model "
                    "knows something is wrong BEFORE the large transaction arrives."
                ),
                "edges": [
                    (7, 56, 0.50, True),
                    (7, 57, 1.00, True),
                    (7, 58, 0.50, True),
                    (7, 59, 0.75, True),
                    (7, 60, 1.00, True),
                ],
                "highlight": [(7, 56), (7, 57), (7, 58), (7, 59), (7, 60)],
                "memory_level": 0.72,
            },
            {
                "title": "Step 4/4: Large transaction — BLOCKED",
                "narrative": (
                    "The attacker attempts a £2,400 purchase. TGN instantly scores "
                    "this as CRITICAL (0.94) because the memory already captured the "
                    "card-testing burst. A logistic regression would score £2,400 at "
                    "a new merchant as only MEDIUM risk."
                ),
                "edges": [
                    (7, 61, 2400.00, True),
                ],
                "highlight": [(7, 61)],
                "memory_level": 0.94,
            },
        ],
        "comparison": (
            "**Why TGN caught this and a simple model wouldn't:**\n\n"
            "A logistic regression scores each transaction independently. "
            "Transaction #1 (£0.50 to merchant A) looks completely normal. "
            "TGN's memory module tracked that 8 micro-transactions happened in 4 minutes, "
            "making transaction #9 (£2,400) instantly recognisable as card testing."
        ),
    },
    "Money Laundering Chain": {
        "description": (
            "Funds are layered through 4 intermediary accounts over 12 hours. "
            "Each individual transfer looks legitimate, but the fund-flow chain "
            "reveals the laundering topology."
        ),
        "steps": [
            {
                "title": "Step 1/4: Initial deposit",
                "narrative": "Account 1 receives a large deposit (£50,000) — the proceeds of crime.",
                "edges": [
                    (0, 1, 50000.00, True),
                ],
                "highlight": [(0, 1)],
                "memory_level": 0.15,
            },
            {
                "title": "Step 2/4: First layer — split across intermediaries",
                "narrative": (
                    "Account 1 immediately splits the funds across 3 intermediary accounts. "
                    "Each transfer is below reporting thresholds."
                ),
                "edges": [
                    (1, 2, 18000.00, True),
                    (1, 3, 17000.00, True),
                    (1, 4, 15000.00, True),
                ],
                "highlight": [(1, 2), (1, 3), (1, 4)],
                "memory_level": 0.45,
            },
            {
                "title": "Step 3/4: Second layer — consolidation",
                "narrative": (
                    "The intermediaries forward funds to a consolidation account. "
                    "TGN sees the fan-out → fan-in topology and flags the chain."
                ),
                "edges": [
                    (2, 5, 17500.00, True),
                    (3, 5, 16500.00, True),
                    (4, 5, 14500.00, True),
                ],
                "highlight": [(2, 5), (3, 5), (4, 5)],
                "memory_level": 0.78,
            },
            {
                "title": "Step 4/4: Final extraction — BLOCKED",
                "narrative": (
                    "Account 5 attempts to withdraw the consolidated £48,500. "
                    "TGN scores this CRITICAL — the full chain path is encoded "
                    "in the graph structure."
                ),
                "edges": [
                    (5, 6, 48500.00, True),
                ],
                "highlight": [(5, 6)],
                "memory_level": 0.91,
            },
        ],
        "comparison": (
            "**Why TGN caught this and a simple model wouldn't:**\n\n"
            "Each individual transfer (£17,000 from account 1 to account 2) "
            "is below suspicious thresholds when viewed alone. TGN encodes the "
            "full graph neighbourhood — it sees that account 5 received funds from "
            "3 accounts that ALL received from account 1 within 2 hours. "
            "The fan-out → fan-in topology is a textbook layering pattern."
        ),
    },
    "Bust-Out Fraud": {
        "description": (
            "An account builds 3 months of legitimate credit history, then "
            "suddenly maxes out with rapid high-value purchases."
        ),
        "steps": [
            {
                "title": "Step 1/3: Building legitimate history",
                "narrative": (
                    "Account 15 establishes a normal spending pattern over weeks — "
                    "regular purchases, consistent amounts, reliable repayment."
                ),
                "edges": [
                    (15, 50, 85.00, False),
                    (15, 51, 120.00, False),
                    (15, 52, 45.00, False),
                    (15, 53, 200.00, False),
                    (15, 50, 90.00, False),
                ],
                "highlight": [],
                "memory_level": 0.03,
            },
            {
                "title": "Step 2/3: Sudden behaviour change",
                "narrative": (
                    "After 3 months, the account suddenly makes 4 high-value purchases "
                    "in rapid succession — completely different from its baseline."
                ),
                "edges": [
                    (15, 54, 3500.00, True),
                    (15, 55, 4200.00, True),
                    (15, 56, 2800.00, True),
                    (15, 57, 5100.00, True),
                ],
                "highlight": [(15, 54), (15, 55), (15, 56), (15, 57)],
                "memory_level": 0.68,
            },
            {
                "title": "Step 3/3: Max-out attempt — BLOCKED",
                "narrative": (
                    "Final attempt: £12,000 electronics purchase. TGN flags CRITICAL "
                    "because the memory encodes the sharp deviation from 3 months of "
                    "baseline behaviour."
                ),
                "edges": [
                    (15, 58, 12000.00, True),
                ],
                "highlight": [(15, 58)],
                "memory_level": 0.89,
            },
        ],
        "comparison": (
            "**Why TGN caught this and a simple model wouldn't:**\n\n"
            "The account's features (age: 90 days, prior transactions: 50+) look "
            "like a trustworthy customer. A feature-based model would give it low risk. "
            "TGN's memory retains the *trajectory* — it knows this account's baseline "
            "is £50–200 transactions and detects the sudden 25x deviation."
        ),
    },
}


# ---------------------------------------------------------------------------
# Pattern Selection
# ---------------------------------------------------------------------------

pattern_name = st.selectbox(
    "Select Fraud Pattern",
    options=list(PATTERNS.keys()),
    format_func=lambda x: f"🎬 {x}",
)
pattern = PATTERNS[pattern_name]

st.markdown(f"*{pattern['description']}*")
st.divider()

# ---------------------------------------------------------------------------
# Step-Through Controls
# ---------------------------------------------------------------------------

total_steps = len(pattern["steps"])

if "vis_step" not in st.session_state:
    st.session_state["vis_step"] = 0
if "vis_pattern" not in st.session_state:
    st.session_state["vis_pattern"] = pattern_name

# Reset step when pattern changes
if st.session_state["vis_pattern"] != pattern_name:
    st.session_state["vis_step"] = 0
    st.session_state["vis_pattern"] = pattern_name

current_step = st.session_state["vis_step"]

# Navigation
col_prev, col_step_label, col_next, col_reset = st.columns([1, 2, 1, 1])

with col_prev:
    if st.button("⬅️ Previous", disabled=(current_step == 0)):
        st.session_state["vis_step"] = max(0, current_step - 1)
        st.rerun()

with col_step_label:
    step_data = pattern["steps"][current_step]
    st.markdown(f"### {step_data['title']}")

with col_next:
    if st.button("Next ➡️", disabled=(current_step >= total_steps - 1), type="primary"):
        st.session_state["vis_step"] = min(total_steps - 1, current_step + 1)
        st.rerun()

with col_reset:
    if st.button("🔄 Reset"):
        st.session_state["vis_step"] = 0
        st.rerun()

# Step progress
st.progress((current_step + 1) / total_steps)

# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------

st.markdown(f"_{step_data['narrative']}_")

# ---------------------------------------------------------------------------
# Graph Visualisation
# ---------------------------------------------------------------------------

# Build cumulative graph up to current step
G = nx.DiGraph()
all_edges_so_far = []
for step_idx in range(current_step + 1):
    for edge in pattern["steps"][step_idx]["edges"]:
        src, dst, amount, is_fraud = edge
        all_edges_so_far.append((src, dst, amount, is_fraud, step_idx))
        G.add_edge(src, dst, amount=amount, fraud=is_fraud, step=step_idx)

# Layout
pos = nx.spring_layout(G, seed=42, k=2.0)

# Build plotly traces
fig = go.Figure()

# Draw edges
current_highlights = set(
    (e[0], e[1]) for e in pattern["steps"][current_step].get("highlight", [])
)

for src, dst, amount, is_fraud, step_idx in all_edges_so_far:
    x0, y0 = pos[src]
    x1, y1 = pos[dst]

    is_highlighted = (src, dst) in current_highlights
    is_current_step = (step_idx == current_step)

    if is_highlighted:
        color = "crimson"
        width = 4
    elif is_fraud:
        color = "rgba(220, 53, 69, 0.4)"
        width = 2
    else:
        color = "rgba(150, 150, 150, 0.4)"
        width = 1

    fig.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="text",
        text=f"£{amount:,.0f}" if is_current_step else "",
        showlegend=False,
    ))

    # Amount label on highlighted edges
    if is_highlighted:
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        fig.add_annotation(
            x=mid_x, y=mid_y,
            text=f"£{amount:,.0f}",
            showarrow=False,
            font=dict(size=10, color="crimson"),
            bgcolor="white",
            bordercolor="crimson",
            borderwidth=1,
        )

# Draw nodes
for node in G.nodes():
    x, y = pos[node]

    # Check if this node is involved in current step highlights
    is_involved = any(node in (e[0], e[1]) for e in current_highlights)

    if is_involved:
        color = "crimson"
        size = 20
    else:
        color = "steelblue"
        size = 12

    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode="markers+text",
        marker=dict(color=color, size=size, line=dict(width=1, color="white")),
        text=str(node),
        textposition="top center",
        textfont=dict(size=9),
        hoverinfo="text",
        hovertext=f"Account {node}",
        showlegend=False,
    ))

fig.update_layout(
    height=400,
    margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TGN Memory Gauge
# ---------------------------------------------------------------------------

col_memory, col_status = st.columns([2, 1])

with col_memory:
    st.markdown("**TGN Memory — Anomaly Score for Primary Account:**")
    memory_level = step_data["memory_level"]
    st.progress(memory_level)

    # Color-coded label
    if memory_level < 0.30:
        st.markdown(f"🟢 **{memory_level:.2f}** — Normal")
    elif memory_level < 0.60:
        st.markdown(f"🟡 **{memory_level:.2f}** — Elevated")
    elif memory_level < 0.85:
        st.markdown(f"🟠 **{memory_level:.2f}** — High")
    else:
        st.markdown(f"🔴 **{memory_level:.2f}** — CRITICAL")

with col_status:
    if memory_level >= 0.85:
        st.error("🚨 **BLOCKED**")
        st.caption("Transaction would be declined in production.")
    elif memory_level >= 0.60:
        st.warning("⚠️ **REVIEW**")
        st.caption("Flagged for manual investigation.")
    else:
        st.success("✓ **PASS**")
        st.caption("Transaction proceeds normally.")

# ---------------------------------------------------------------------------
# Comparison (show on final step)
# ---------------------------------------------------------------------------

if current_step == total_steps - 1:
    st.divider()
    st.markdown(pattern["comparison"])
