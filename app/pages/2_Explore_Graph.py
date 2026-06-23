"""
Streamlit page: Interactive graph visualization.

Lets users explore the generated transaction network with
filtering by time range, node type, and fraud edges.
"""

import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go

st.header("Graph Explorer")

if "graph" not in st.session_state:
    st.warning("No graph generated yet. Go to **Generate Data** first.")
    st.stop()

graph = st.session_state["graph"]

st.markdown(f"Exploring graph with **{graph.num_nodes}** nodes and **{graph.num_edges}** edges")

# --- Filters ---
st.subheader("Filters")
col1, col2, col3 = st.columns(3)

with col1:
    t_min, t_max = graph.time_range
    time_range = st.slider(
        "Time Range",
        min_value=float(t_min), max_value=float(t_max),
        value=(float(t_min), float(t_max)),
    )

with col2:
    show_fraud_only = st.checkbox("Show only fraud-connected nodes", value=False)
    max_nodes = st.slider("Max nodes to display", 10, 200, 50)

with col3:
    node_types_available = list(graph.node_types().keys())
    selected_types = st.multiselect(
        "Node types", node_types_available, default=node_types_available
    )

# --- Build subgraph for visualization ---
edges_in_range = graph.edges_in_range(time_range[0], time_range[1])

if show_fraud_only:
    fraud_nodes = set()
    for e in edges_in_range:
        if e.label == 1:
            fraud_nodes.add(e.src_id)
            fraud_nodes.add(e.dst_id)
    edges_in_range = [e for e in edges_in_range if e.src_id in fraud_nodes or e.dst_id in fraud_nodes]

# Filter by node type
valid_nodes = set()
for n in graph.nodes:
    if n.node_type in selected_types:
        valid_nodes.add(n.node_id)

edges_in_range = [e for e in edges_in_range if e.src_id in valid_nodes and e.dst_id in valid_nodes]

# Limit nodes
involved_nodes = set()
for e in edges_in_range:
    involved_nodes.add(e.src_id)
    involved_nodes.add(e.dst_id)
    if len(involved_nodes) >= max_nodes:
        break

edges_display = [e for e in edges_in_range if e.src_id in involved_nodes and e.dst_id in involved_nodes]

st.info(f"Displaying {len(involved_nodes)} nodes, {len(edges_display)} edges")

# --- Build NetworkX graph for layout ---
G = nx.Graph()
for nid in involved_nodes:
    node = graph.get_node(nid)
    G.add_node(nid, node_type=node.node_type if node else "unknown")

for e in edges_display:
    G.add_edge(e.src_id, e.dst_id, label=e.label)

if len(G.nodes) == 0:
    st.warning("No nodes match the current filters.")
    st.stop()

# Spring layout
pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(len(G.nodes)))

# --- Plotly visualization ---
# Color map by node type
color_map = {"account": "#4A90D9", "merchant": "#27AE60", "device": "#F39C12"}

# Edge traces
edge_x, edge_y = [], []
fraud_edge_x, fraud_edge_y = [], []

for e in edges_display:
    if e.src_id in pos and e.dst_id in pos:
        x0, y0 = pos[e.src_id]
        x1, y1 = pos[e.dst_id]
        if e.label == 1:
            fraud_edge_x.extend([x0, x1, None])
            fraud_edge_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

fig = go.Figure()

# Normal edges
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(width=0.5, color="#CCCCCC"),
    hoverinfo="none", name="Legitimate",
))

# Fraud edges
fig.add_trace(go.Scatter(
    x=fraud_edge_x, y=fraud_edge_y, mode="lines",
    line=dict(width=2, color="#E74C3C"),
    hoverinfo="none", name="Fraud",
))

# Node traces by type
for node_type in selected_types:
    node_x, node_y, node_text = [], [], []
    for nid in involved_nodes:
        node = graph.get_node(nid)
        if node and node.node_type == node_type and nid in pos:
            x, y = pos[nid]
            node_x.append(x)
            node_y.append(y)
            n_edges = len(graph.edges_for_node(nid))
            node_text.append(f"ID: {nid}<br>Type: {node_type}<br>Edges: {n_edges}")

    color = color_map.get(node_type, "#9B59B6")
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=10, color=color, line=dict(width=1, color="white")),
        text=node_text, hoverinfo="text",
        name=node_type.capitalize(),
    ))

fig.update_layout(
    showlegend=True,
    height=600,
    margin=dict(t=20, b=20, l=20, r=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

# --- Node details on click ---
st.subheader("Node Inspector")
node_id_input = st.number_input(
    "Enter Node ID to inspect", min_value=0,
    max_value=graph.num_nodes - 1, value=0
)

node = graph.get_node(node_id_input)
if node:
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Type:** {node.node_type}")
        st.write(f"**Metadata:** {node.metadata}")
    with col2:
        node_edges = graph.edges_for_node(node_id_input)
        fraud_count = sum(1 for e in node_edges if e.label == 1)
        st.write(f"**Total edges:** {len(node_edges)}")
        st.write(f"**Fraud edges:** {fraud_count}")
        if node_edges:
            amounts = [e.features[0] for e in node_edges]
            st.write(f"**Avg log_amount:** {np.mean(amounts):.2f}")
