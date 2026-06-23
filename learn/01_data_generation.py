#!/usr/bin/env python3
"""
Script 01: Synthetic Fraud Data Generation
==========================================

This script demonstrates how to generate synthetic transaction networks
with injected fraud patterns for TGN training. You'll learn:

- How to configure and run synthetic generators
- What fraud patterns look like in graph form
- How to inspect the temporal and structural properties of the data

Run: python learn/01_data_generation.py
"""

import numpy as np

from tgn_learn.generators import BankSimGenerator, PaySimGenerator, GeneratorRegistry
from tgn_learn.generators.base import GeneratorConfig


def main():
    print("=" * 70)
    print("  01: Synthetic Fraud Data Generation")
    print("=" * 70)
    print()

    # ─── Section 1: Available Generators ──────────────────────────────────
    print("─── Available Generators ───")
    print()
    for info in GeneratorRegistry.list_generators():
        print(f"  {info['name']:10s} — {info['description']}")
        print(f"             Patterns: {', '.join(info['fraud_patterns'])}")
        print()

    # ─── Section 2: Generate a BankSim Network ────────────────────────────
    print("─── Generating BankSim Network ───")
    print()

    config = GeneratorConfig(
        num_accounts=200,
        num_merchants=30,
        num_transactions=5000,
        fraud_rate=0.03,   # 3% fraud
        seed=42,           # Reproducible
        duration_days=30,  # 30-day simulation
    )

    gen = BankSimGenerator(config)
    graph = gen.generate()
    print(graph.summary())
    print()

    # ─── Section 3: Inspect Fraud Patterns ────────────────────────────────
    print("─── Fraud Pattern Breakdown ───")
    print()

    pattern_counts = {}
    for edge in graph.edges:
        if edge.label == 1:
            pattern = edge.metadata.get("pattern", "unknown")
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    for pattern, count in sorted(pattern_counts.items()):
        print(f"  {pattern:25s}: {count:4d} edges ({count/graph.num_fraud*100:.1f}%)")
    print(f"  {'TOTAL':25s}: {graph.num_fraud:4d}")
    print()

    # ─── Section 4: Temporal Properties ───────────────────────────────────
    print("─── Temporal Properties ───")
    print()

    t_min, t_max = graph.time_range
    duration_days = (t_max - t_min) / 86400
    print(f"  Time span: {duration_days:.1f} days")
    print(f"  Avg transactions/day: {graph.num_edges / duration_days:.0f}")

    # Check fraud temporal distribution
    fraud_timestamps = [e.timestamp for e in graph.edges if e.label == 1]
    legit_timestamps = [e.timestamp for e in graph.edges if e.label == 0]

    print(f"  Fraud avg time offset: {(np.mean(fraud_timestamps) - t_min)/86400:.1f} days from start")
    print(f"  Legit avg time offset: {(np.mean(legit_timestamps) - t_min)/86400:.1f} days from start")
    print()

    # ─── Section 5: Reproducibility ──────────────────────────────────────
    print("─── Reproducibility Check ───")
    print()

    graph2 = BankSimGenerator(config).generate()
    match = all(
        e1.src_id == e2.src_id and e1.label == e2.label
        for e1, e2 in zip(graph.edges, graph2.edges)
    )
    print(f"  Same seed produces identical graph: {match}")
    print()

    # ─── Section 6: PaySim Comparison ─────────────────────────────────────
    print("─── PaySim Generator (Mobile Money) ───")
    print()

    paysim_config = GeneratorConfig(
        num_accounts=150, num_transactions=3000,
        fraud_rate=0.05, seed=99,
    )
    paysim_graph = PaySimGenerator(paysim_config).generate()
    print(paysim_graph.summary())
    print()

    # ─── Key Takeaways ───────────────────────────────────────────────────
    print("─── Key Takeaways ───")
    print()
    print("  1. Synthetic generators let you control fraud rate and patterns")
    print("  2. Different patterns have different temporal signatures:")
    print("     - Card testing: rapid bursts of tiny amounts")
    print("     - Money laundering: layered chains between accounts")
    print("     - Account takeover: sudden high-value spike")
    print("  3. Seeded generation ensures reproducible experiments")
    print("  4. Multiple generators (BankSim, PaySim) cover different scenarios")
    print()
    print("Next: 02_graph_construction.py — Build graphs from scratch")


if __name__ == "__main__":
    main()
