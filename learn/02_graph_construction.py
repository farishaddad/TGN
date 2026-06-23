#!/usr/bin/env python3
"""
Script 02: Graph Construction
==============================

Build temporal graphs from scratch and understand the Node/Edge/TemporalGraph
data model. You'll learn:

- How nodes represent entities (accounts, merchants, devices)
- How edges represent temporal interactions with features
- How to convert to PyG TemporalData for model training
- How temporal splitting works for causal evaluation

Run: python learn/02_graph_construction.py
"""

import numpy as np
import torch

from tgn_learn.graph import Edge, Node, TemporalGraph, EDGE_FEAT_DIM


def main():
    print("=" * 70)
    print("  02: Graph Construction — From Scratch to PyG")
    print("=" * 70)
    print()

    # ─── Section 1: Creating Nodes ────────────────────────────────────────
    print("─── Creating Nodes ───")
    print()
    print("  Nodes represent entities in the transaction network:")
    print("  - Accounts: customers, companies")
    print("  - Merchants: shops, services")
    print("  - Devices: phones, computers (optional)")
    print()

    alice = Node(node_id=0, node_type="account", metadata={"name": "Alice", "age_days": 365})
    bob = Node(node_id=1, node_type="account", metadata={"name": "Bob", "age_days": 30})
    shop = Node(node_id=2, node_type="merchant", metadata={"name": "CoffeeShop", "category": "food"})
    store = Node(node_id=3, node_type="merchant", metadata={"name": "Electronics", "category": "tech"})

    print(f"  Alice: {alice}")
    print(f"  Bob:   {bob}")
    print(f"  Shop:  {shop}")
    print(f"  Store: {store}")
    print()

    # ─── Section 2: Creating Edges ────────────────────────────────────────
    print("─── Creating Temporal Edges ───")
    print()
    print(f"  Each edge has a {EDGE_FEAT_DIM}-dimensional feature vector encoding:")
    print("  [0] log_amount, [1] normalized_amount, [2-5] time cyclical,")
    print("  [6] channel, [7] international, [8] velocity, [9] deviation, ...")
    print()

    # Create a simple feature vector
    def make_features(amount: float, channel: float = 0.5) -> np.ndarray:
        feat = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
        feat[0] = np.log1p(amount)
        feat[1] = min(amount / 10000, 1.0)
        feat[6] = channel
        return feat

    edges = [
        Edge(src_id=0, dst_id=2, timestamp=1000.0, features=make_features(5.50, 0.2),
             label=0, edge_type="purchase"),
        Edge(src_id=1, dst_id=2, timestamp=2000.0, features=make_features(4.25, 0.2),
             label=0, edge_type="purchase"),
        Edge(src_id=0, dst_id=3, timestamp=3000.0, features=make_features(1200.0, 0.5),
             label=0, edge_type="purchase"),
        Edge(src_id=1, dst_id=3, timestamp=4000.0, features=make_features(8500.0, 0.5),
             label=1, edge_type="purchase"),  # Fraud!
        Edge(src_id=0, dst_id=2, timestamp=5000.0, features=make_features(6.00, 0.2),
             label=0, edge_type="purchase"),
    ]

    for e in edges:
        status = "FRAUD" if e.label == 1 else "legit"
        amount = np.exp(e.features[0]) - 1
        print(f"  t={e.timestamp:.0f}: {e.src_id}→{e.dst_id} ${amount:.2f} [{status}]")
    print()

    # ─── Section 3: Building the Graph ────────────────────────────────────
    print("─── Building TemporalGraph ───")
    print()

    graph = TemporalGraph()
    for node in [alice, bob, shop, store]:
        graph.add_node(node)
    graph.add_edges(edges)

    print(graph.summary())
    print()

    # ─── Section 4: Querying the Graph ────────────────────────────────────
    print("─── Querying ───")
    print()
    print(f"  Alice's edges: {len(graph.edges_for_node(0))}")
    print(f"  Bob's edges:   {len(graph.edges_for_node(1))}")
    print(f"  Time range:    {graph.time_range}")
    print(f"  Edges in [2000, 4000]: {len(graph.edges_in_range(2000, 4000))}")
    print()

    # ─── Section 5: Converting to PyG ────────────────────────────────────
    print("─── Converting to PyG TemporalData ───")
    print()
    print("  PyG's TemporalData is the native input format for TGN training.")
    print("  It stores src, dst, timestamps, messages, and labels as tensors.")
    print()

    data = graph.to_pyg_temporal_data()
    print(f"  data.src:  shape={data.src.shape}, dtype={data.src.dtype}")
    print(f"  data.dst:  shape={data.dst.shape}, dtype={data.dst.dtype}")
    print(f"  data.t:    shape={data.t.shape}, dtype={data.t.dtype}")
    print(f"  data.msg:  shape={data.msg.shape}, dtype={data.msg.dtype}")
    print(f"  data.y:    shape={data.y.shape}, values={data.y.tolist()}")
    print()

    # ─── Section 6: Temporal Splitting ────────────────────────────────────
    print("─── Temporal Splitting (70/15/15) ───")
    print()
    print("  Unlike random splitting, temporal splitting preserves causality:")
    print("  - Train on PAST transactions")
    print("  - Validate on NEAR-FUTURE")
    print("  - Test on FAR-FUTURE")
    print()

    train, val, test = graph.temporal_split(0.70, 0.15)
    print(f"  Train: {train.num_edges} edges, time [{train.time_range[0]:.0f}, {train.time_range[1]:.0f}]")
    print(f"  Val:   {val.num_edges} edges, time [{val.time_range[0]:.0f}, {val.time_range[1]:.0f}]")
    print(f"  Test:  {test.num_edges} edges, time [{test.time_range[0]:.0f}, {test.time_range[1]:.0f}]")
    print()
    print("  This prevents 'future leakage' — the model never sees future data during training.")
    print()

    # ─── Key Takeaways ───────────────────────────────────────────────────
    print("─── Key Takeaways ───")
    print()
    print("  1. Nodes = entities, Edges = temporal interactions")
    print("  2. Edge features encode: amount, time patterns, channel, velocity")
    print("  3. TemporalGraph maintains chronological ordering automatically")
    print("  4. to_pyg_temporal_data() converts to native TGN training format")
    print("  5. Temporal splitting is essential for evaluating sequential models")
    print()
    print("Next: 03_tgn_architecture.py — Understand TGN components")


if __name__ == "__main__":
    main()
