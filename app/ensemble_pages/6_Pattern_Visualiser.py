"""
Ensemble Pattern Visualiser — step-through fraud animation with detector overlay.

Shows how each detector in the ensemble responds at each step of the attack.
This extends the standard Pattern Visualiser with a per-detector response panel.
"""

import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go

st.header("Ensemble Pattern Visualiser")
st.markdown(
    "Step through a fraud attack and see how each ensemble detector responds."
)

# ---------------------------------------------------------------------------
# Pattern Definitions with per-detector responses
# ---------------------------------------------------------------------------

PATTERNS = {
    "Card Testing Ring": {
        "description": (
            "A stolen card is validated with rapid micro-transactions. "
            "Watch how different detectors activate at different stages."
        ),
        "steps": [
            {
                "title": "Step 1/4: Normal baseline",
                "narrative": "Account 7 has a normal history — coffee, groceries, fuel.",
                "edges": [
                    (7, 50, 12.50, False),
                    (7, 51, 45.00, False),
                    (7, 52, 38.00, False),
                    (7, 50, 8.90, False),
                ],
                "highlight": [],
                "memory_level": 0.05,
                "detectors": {
                    "TGN Memory": 0.03,
                    "RF Structural": 0.02,
                    "Fund-Flow Graph": 0.01,
                    "Drift Monitor": 0.02,
                },
            },
            {
                "title": "Step 2/4: Card testing begins",
                "narrative": (
                    "The attacker tests with micro-transactions (£0.50–£1.00). "
                    "TGN Memory starts rising but RF sees nothing wrong yet."
                ),
                "edges": [
                    (7, 53, 0.50, True),
                    (7, 54, 0.75, True),
                    (7, 55, 1.00, True),
                ],
                "highlight": [(7, 53), (7, 54), (7, 55)],
                "memory_level": 0.35,
                "detectors": {
                    "TGN Memory": 0.35,
                    "RF Structural": 0.12,
                    "Fund-Flow Graph": 0.08,
                    "Drift Monitor": 0.25,
                },
            },
            {
                "title": "Step 3/4: Velocity anomaly detected",
                "narrative": (
                    "5 more micro-transactions in 4 minutes. TGN Memory and "
                    "Drift Monitor both trigger. Fund-Flow starts seeing a chain."
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
                "detectors": {
                    "TGN Memory": 0.72,
                    "RF Structural": 0.35,
                    "Fund-Flow Graph": 0.45,
                    "Drift Monitor": 0.68,
                },
            },
            {
                "title": "Step 4/4: Large transaction — BLOCKED",
                "narrative": (
                    "£2,400 purchase attempt. Ensemble consensus: CRITICAL. "
                    "All detectors agree — meta-learner outputs 0.94."
                ),
                "edges": [
                    (7, 61, 2400.00, True),
                ],
                "highlight": [(7, 61)],
                "memory_level": 0.94,
                "detectors": {
                    "TGN Memory": 0.94,
                    "RF Structural": 0.78,
                    "Fund-Flow Graph": 0.82,
                    "Drift Monitor": 0.91,
                },
            },
        ],
        "comparison": (
            "**Ensemble advantage:** TGN Memory caught this at step 2 (score 0.35). "
            "By step 3, the Drift Monitor confirmed the anomaly independently. "
            "A single-model system might have waited until step 4 — the ensemble "
            "provides earlier warning with higher confidence through detector agreement."
        ),
    },
    "Money Laundering Chain": {
        "description": (
            "Funds layered through intermediaries. Watch the Fund-Flow Graph "
            "detector activate as the chain becomes visible."
        ),
        "steps": [
            {
                "title": "Step 1/4: Initial deposit",
                "narrative": "Account 1 receives £50,000 — the proceeds of crime.",
                "edges": [(0, 1, 50000.00, True)],
                "highlight": [(0, 1)],
                "memory_level": 0.15,
                "detectors": {
                    "TGN Memory": 0.15,
                    "RF Structural": 0.08,
                    "Fund-Flow Graph": 0.10,
                    "Drift Monitor": 0.12,
                },
            },
            {
                "title": "Step 2/4: Split across intermediaries",
                "narrative": (
                    "Funds split below reporting thresholds. Fund-Flow Graph "
                    "starts detecting the fan-out topology."
                ),
                "edges": [
                    (1, 2, 18000.00, True),
                    (1, 3, 17000.00, True),
                    (1, 4, 15000.00, True),
                ],
                "highlight": [(1, 2), (1, 3), (1, 4)],
                "memory_level": 0.45,
                "detectors": {
                    "TGN Memory": 0.30,
                    "RF Structural": 0.42,
                    "Fund-Flow Graph": 0.55,
                    "Drift Monitor": 0.20,
                },
            },
            {
                "title": "Step 3/4: Consolidation — fan-in detected",
                "narrative": (
                    "Intermediaries forward to consolidation account. "
                    "Fund-Flow Graph scores CRITICAL — full fan-out→fan-in visible."
                ),
                "edges": [
                    (2, 5, 17500.00, True),
                    (3, 5, 16500.00, True),
                    (4, 5, 14500.00, True),
                ],
                "highlight": [(2, 5), (3, 5), (4, 5)],
                "memory_level": 0.78,
                "detectors": {
                    "TGN Memory": 0.55,
                    "RF Structural": 0.72,
                    "Fund-Flow Graph": 0.88,
                    "Drift Monitor": 0.35,
                },
            },
            {
                "title": "Step 4/4: Extraction attempt — BLOCKED",
                "narrative": (
                    "Account 5 attempts £48,500 withdrawal. Ensemble: CRITICAL. "
                    "Fund-Flow Graph alone would have blocked at step 3."
                ),
                "edges": [(5, 6, 48500.00, True)],
                "highlight": [(5, 6)],
                "memory_level": 0.91,
                "detectors": {
                    "TGN Memory": 0.70,
                    "RF Structural": 0.85,
                    "Fund-Flow Graph": 0.95,
                    "Drift Monitor": 0.45,
                },
            },
        ],
        "comparison": (
            "**Ensemble advantage:** The Fund-Flow Graph detector identified "
            "the laundering topology at step 2 (score 0.55) — 2 steps before "
            "a single TGN would flag it. RF Structural confirmed via graph "
            "connectivity analysis. This is exactly the scenario where "
            "multi-detector ensembles excel."
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
# Step Controls
# ---------------------------------------------------------------------------

total_steps = len(pattern["steps"])

if "ens_vis_step" not in st.session_state:
    st.session_state["ens_vis_step"] = 0
if "ens_vis_pattern" not in st.session_state:
    st.session_state["ens_vis_pattern"] = pattern_name

if st.session_state["ens_vis_pattern"] != pattern_name:
    st.session_state["ens_vis_step"] = 0
    st.session_state["ens_vis_pattern"] = pattern_name

current_step = st.session_state["ens_vis_step"]

col_prev, col_label, col_next, col_reset = st.columns([1, 2, 1, 1])
with col_prev:
    if st.button("⬅️ Previous", disabled=(current_step == 0), key="ens_prev"):
        st.session_state["ens_vis_step"] = max(0, current_step - 1)
        st.rerun()
with col_label:
    step_data = pattern["steps"][current_step]
    st.markdown(f"### {step_data['title']}")
with col_next:
    if st.button("Next ➡️", disabled=(current_step >= total_steps - 1), type="primary", key="ens_next"):
        st.session_state["ens_vis_step"] = min(total_steps - 1, current_step + 1)
        st.rerun()
with col_reset:
    if st.button("🔄 Reset", key="ens_reset"):
        st.session_state["ens_vis_step"] = 0
        st.rerun()

st.progress((current_step + 1) / total_steps)
st.markdown(f"_{step_data['narrative']}_")

# ---------------------------------------------------------------------------
# Graph Visualisation
# ---------------------------------------------------------------------------

G = nx.DiGraph()
all_edges_so_far = []
for step_idx in range(current_step + 1):
    for edge in pattern["steps"][step_idx]["edges"]:
        src, dst, amount, is_fraud = edge
        all_edges_so_far.append((src, dst, amount, is_fraud, step_idx))
        G.add_edge(src, dst, amount=amount, fraud=is_fraud, step=step_idx)

pos = nx.spring_layout(G, seed=42, k=2.0)

fig = go.Figure()
current_highlights = set(
    (e[0], e[1]) for e in pattern["steps"][current_step].get("highlight", [])
)

for src, dst, amount, is_fraud, step_idx in all_edges_so_far:
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    is_highlighted = (src, dst) in current_highlights

    if is_highlighted:
        color, width = "crimson", 4
    elif is_fraud:
        color, width = "rgba(220, 53, 69, 0.4)", 2
    else:
        color, width = "rgba(150, 150, 150, 0.4)", 1

    fig.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode="lines", line=dict(color=color, width=width),
        hoverinfo="skip", showlegend=False,
    ))

    if is_highlighted:
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_annotation(
            x=mid_x, y=mid_y, text=f"£{amount:,.0f}",
            showarrow=False, font=dict(size=10, color="crimson"),
            bgcolor="white", bordercolor="crimson", borderwidth=1,
        )

for node in G.nodes():
    x, y = pos[node]
    is_involved = any(node in (e[0], e[1]) for e in current_highlights)
    color = "crimson" if is_involved else "steelblue"
    size = 20 if is_involved else 12

    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text",
        marker=dict(color=color, size=size, line=dict(width=1, color="white")),
        text=str(node), textposition="top center", textfont=dict(size=9),
        hoverinfo="text", hovertext=f"Account {node}", showlegend=False,
    ))

fig.update_layout(
    height=350, margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Ensemble Detector Response Panel
# ---------------------------------------------------------------------------

st.markdown("#### Ensemble Detector Responses")

detectors = step_data["detectors"]

cols = st.columns(len(detectors))
for i, (det_name, det_score) in enumerate(detectors.items()):
    with cols[i]:
        if det_score >= 0.85:
            color = "🔴"
        elif det_score >= 0.60:
            color = "🟠"
        elif det_score >= 0.30:
            color = "🟡"
        else:
            color = "🟢"
        st.metric(det_name, f"{det_score:.2f}", label_visibility="visible")
        st.progress(det_score)

# Consensus score
memory_level = step_data["memory_level"]
st.markdown("---")
col_mem, col_status = st.columns([3, 1])
with col_mem:
    st.markdown("**Meta-Learner Consensus Score:**")
    st.progress(memory_level)
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
    elif memory_level >= 0.60:
        st.warning("⚠️ **REVIEW**")
    else:
        st.success("✓ **PASS**")

# ---------------------------------------------------------------------------
# Comparison (final step)
# ---------------------------------------------------------------------------

if current_step == total_steps - 1:
    st.divider()
    st.markdown(pattern["comparison"])
