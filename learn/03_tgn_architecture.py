#!/usr/bin/env python3
"""
Script 03: TGN Architecture Deep Dive
=======================================

Walk through each component of the Temporal Graph Network and understand
how they work together for fraud detection. You'll learn:

- TimeEncoder: How timestamps become learnable representations
- TGNMemory: How per-node state evolves with each interaction
- GraphAttentionEmbedding: How neighbors provide context
- Scoring Heads: How embeddings become fraud predictions

Run: python learn/03_tgn_architecture.py
"""

import torch
import numpy as np

from tgn_learn.model import (
    TGNConfig, TimeEncoder, GraphAttentionEmbedding,
    LinkPredictor, NodeClassifier, TGNFraudDetector,
)


def main():
    print("=" * 70)
    print("  03: TGN Architecture Deep Dive")
    print("=" * 70)
    print()

    config = TGNConfig(memory_dim=64, embedding_dim=64, time_dim=32, edge_feat_dim=20)
    print(f"  Config: memory={config.memory_dim}, embed={config.embedding_dim}, "
          f"time={config.time_dim}, edge_feat={config.edge_feat_dim}")
    print()

    # ─── Component 1: Time Encoder ────────────────────────────────────────
    print("─── Component 1: Time Encoder ───")
    print()
    print("  Maps scalar timestamps → high-dimensional vectors using learnable")
    print("  cosine frequencies spanning many time scales (seconds to years).")
    print()

    time_enc = TimeEncoder(time_dim=config.time_dim)
    print(f"  Parameters: {sum(p.numel() for p in time_enc.parameters())}")

    # Show how different time deltas produce different encodings
    times = torch.tensor([0.0, 60.0, 3600.0, 86400.0, 604800.0])
    encodings = time_enc(times)
    print(f"  Input times:  {times.tolist()} (0s, 1min, 1hr, 1day, 1week)")
    print(f"  Output shape: {encodings.shape}")
    print(f"  Encoding similarity (cosine):")

    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            sim = torch.nn.functional.cosine_similarity(
                encodings[i:i+1], encodings[j:j+1]
            ).item()
            print(f"    {times[i]:.0f}s vs {times[j]:.0f}s: {sim:.4f}")
    print()

    # ─── Component 2: TGN Memory ─────────────────────────────────────────
    print("─── Component 2: TGN Memory ───")
    print()
    print("  The key innovation of TGN: each node maintains a state vector")
    print("  that evolves with every interaction. This captures behavioral history.")
    print()
    print("  Memory update flow:")
    print("    1. New interaction arrives: (src, dst, t, features)")
    print("    2. Message function computes: msg = f(memory[src], memory[dst], features, t)")
    print("    3. Aggregator combines messages: agg = LastAggregator(messages)")
    print("    4. Memory updated: memory[node] = GRU(memory[node], agg)")
    print()

    model = TGNFraudDetector(num_nodes=20, config=config)
    print(f"  Model created with 20 nodes, memory_dim={config.memory_dim}")

    # Show memory before and after interactions
    z_before, _ = model.memory(torch.tensor([0, 1, 2]))
    print(f"  Initial memory[0] norm: {z_before[0].norm():.4f}")

    # Simulate an interaction
    src = torch.tensor([0])
    dst = torch.tensor([1])
    t = torch.tensor([1000.0])
    msg = torch.randn(1, 20)
    model(src, dst, t, msg)

    z_after, _ = model.memory(torch.tensor([0, 1, 2]))
    print(f"  After interaction memory[0] norm: {z_after[0].norm():.4f}")
    print(f"  Memory[2] (uninvolved) changed: {not torch.allclose(z_before[2], z_after[2])}")
    print()

    # ─── Component 3: Graph Attention Embedding ───────────────────────────
    print("─── Component 3: Graph Attention Embedding ───")
    print()
    print("  TransformerConv aggregates information from temporal neighbors:")
    print("  - Multi-head attention weights relevant neighbors more")
    print("  - Edge features include message content + time encoding")
    print("  - Output: contextual embedding for each node")
    print()

    gnn = model.gnn
    print(f"  GNN parameters: {sum(p.numel() for p in gnn.parameters())}")
    print(f"  Attention heads: {config.num_heads}")
    print(f"  Output dimension: {config.embedding_dim}")
    print()

    # ─── Component 4: Scoring Heads ───────────────────────────────────────
    print("─── Component 4: Scoring Heads ───")
    print()
    print("  Two parallel heads operating on learned embeddings:")
    print()

    print("  LinkPredictor: Scores transactions (is this edge anomalous?)")
    print(f"    Architecture: Linear({config.embedding_dim}) + Linear({config.embedding_dim}) → ReLU → Linear(1)")
    link_pred = model.link_pred
    z_src = torch.randn(1, config.embedding_dim)
    z_dst = torch.randn(1, config.embedding_dim)
    link_score = link_pred(z_src, z_dst)
    print(f"    Example output (logit): {link_score.item():.4f}")
    print(f"    As probability: {link_score.sigmoid().item():.4f}")
    print()

    print("  NodeClassifier: Scores accounts (is this node risky?)")
    print(f"    Architecture: Linear({config.embedding_dim}, 128) → ReLU → Dropout → Linear(128, 1)")
    node_pred = model.node_pred
    z = torch.randn(1, config.embedding_dim)
    node_score = node_pred(z)
    print(f"    Example output (logit): {node_score.item():.4f}")
    print(f"    As probability: {node_score.sigmoid().item():.4f}")
    print()

    # ─── Full Model Summary ───────────────────────────────────────────────
    print("─── Full Model Summary ───")
    print()
    print(f"  Total trainable parameters: {model.total_parameters:,}")
    print()
    print("  Forward pass flow:")
    print("    Input: (src_ids, dst_ids, timestamps, edge_features)")
    print("    1. Look up memory states for all involved nodes")
    print("    2. Compute embeddings (z_src, z_dst)")
    print("    3. Score links: link_pred(z_src, z_dst) → anomaly logit")
    print("    4. Score nodes: node_pred(z_src) → risk logit")
    print("    5. Update memory with new interactions")
    print("    Output: (pos_score, neg_score, node_scores)")
    print()

    # ─── Key Takeaways ───────────────────────────────────────────────────
    print("─── Key Takeaways ───")
    print()
    print("  1. TGN Memory is the core innovation — nodes remember their history")
    print("  2. Time encoding lets the model learn temporal patterns at any scale")
    print("  3. Graph attention weighs recent/relevant neighbors more")
    print("  4. Dual heads enable both transaction-level and account-level detection")
    print("  5. The model updates memory DURING inference — it's always learning")
    print()
    print("Next: 04_training_loop.py — Train the model with loss visualization")


if __name__ == "__main__":
    main()
