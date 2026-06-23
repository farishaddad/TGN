#!/usr/bin/env python3
"""
Script 05: Inference & Scoring
===============================

Load a trained model, score transactions, calibrate scores, and
classify risk tiers. You'll learn:

- How to use the Scorer class for single/batch inference
- How isotonic calibration improves probability estimates
- How risk tiers map scores to actions (allow/review/hold/block)

Run: python learn/05_inference_scoring.py
"""

import numpy as np

from tgn_learn.generators import BankSimGenerator
from tgn_learn.generators.base import GeneratorConfig
from tgn_learn.graph import Edge
from tgn_learn.model import TGNConfig, TGNFraudDetector
from tgn_learn.scoring import Scorer, RiskTier
from tgn_learn.training import TGNTrainer, TrainingConfig


def main():
    print("=" * 70)
    print("  05: Inference & Scoring")
    print("=" * 70)
    print()

    # Train a quick model first
    print("─── Training a model (quick) ───")
    config = GeneratorConfig(
        num_accounts=100, num_merchants=20,
        num_transactions=2000, fraud_rate=0.05, seed=42,
    )
    graph = BankSimGenerator(config).generate()

    trainer = TGNTrainer(
        TrainingConfig(epochs=10, batch_size=200, patience=5),
        TGNConfig(memory_dim=32, embedding_dim=32, time_dim=16),
    )
    results = trainer.train(graph, verbose=False)
    print(f"  Trained model: Val AUC-PR={results['best_metrics'].auc_pr:.4f}")
    print()

    # Create scorer
    scorer = Scorer(results["model"], device="cpu")

    # ─── Single Transaction Scoring ───────────────────────────────────────
    print("─── Single Transaction Scoring ───")
    print()
    print("  Score a single transaction by providing src, dst, amount, timestamp:")
    print()

    result = scorer.score_transaction(src=0, dst=50, timestamp=1700050000.0, amount=100.0)
    print(f"  Normal purchase ($100): {result}")

    result = scorer.score_transaction(src=0, dst=50, timestamp=1700050001.0, amount=9500.0)
    print(f"  Large purchase ($9500): {result}")
    print()

    # ─── Batch Scoring ────────────────────────────────────────────────────
    print("─── Batch Scoring ───")
    print()
    print("  Score multiple transactions at once for efficiency:")
    print()

    scorer.model.reset_memory()
    test_edges = graph.edges[-10:]
    batch_results = scorer.score_batch(test_edges)

    print(f"  {'Src→Dst':<12} {'Amount':>8} {'Actual':>8} {'Score':>8} {'Tier':<10}")
    print(f"  {'-'*55}")
    for edge, res in zip(test_edges, batch_results):
        amount = np.exp(edge.features[0]) - 1
        actual = "FRAUD" if edge.label == 1 else "legit"
        print(f"  {edge.src_id:>3}→{edge.dst_id:<6} ${amount:>7.0f} {actual:>8} "
              f"{res.risk_score:>8.4f} {res.risk_tier.value:<10}")
    print()

    # ─── Risk Tier System ─────────────────────────────────────────────────
    print("─── Risk Tier System ───")
    print()
    print("  Tiers map continuous scores to discrete actions:")
    print()
    print("  Score Range    Tier      Action")
    print("  ──────────────────────────────────────────")
    print("  0.00 - 0.30    LOW       Allow automatically")
    print("  0.30 - 0.60    MEDIUM    Flag for review")
    print("  0.60 - 0.85    HIGH      Hold for investigation")
    print("  0.85 - 1.00    CRITICAL  Block immediately")
    print()

    # ─── Calibration ──────────────────────────────────────────────────────
    print("─── Score Calibration ───")
    print()
    print("  Raw model outputs are not well-calibrated probabilities.")
    print("  Isotonic regression maps raw scores → true probabilities")
    print("  using validation data.")
    print()

    # Simulate calibration data
    val_scores = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7, 0.75,
                          0.8, 0.85, 0.9, 0.92, 0.95, 0.12, 0.22, 0.65, 0.78, 0.88])
    val_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 0, 0, 1, 1, 1])

    scorer.calibrate(val_scores, val_labels)
    print("  Calibration fitted on 20 validation samples.")
    print()
    print("  Before vs After calibration:")
    for raw in [0.3, 0.5, 0.7, 0.9]:
        cal = scorer._calibrate_score(raw)
        print(f"    Raw={raw:.1f} → Calibrated={cal:.3f}")
    print()

    # ─── Key Takeaways ───────────────────────────────────────────────────
    print("─── Key Takeaways ───")
    print()
    print("  1. Scorer wraps the model with user-friendly API")
    print("  2. Batch scoring is more efficient than repeated single calls")
    print("  3. Calibration maps model outputs to true probabilities")
    print("  4. Risk tiers convert probabilities to actionable decisions")
    print("  5. Confidence bounds indicate prediction reliability")
    print()
    print("Next: 06_mint_transfer.py — Multi-network transfer learning")


if __name__ == "__main__":
    main()
