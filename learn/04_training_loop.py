#!/usr/bin/env python3
"""
Script 04: Training Loop
=========================

Full TGN training with loss visualization, temporal splitting, and
class weighting for extreme imbalance. You'll learn:

- How contrastive + supervised loss work together
- Why temporal splitting prevents future leakage
- How class weighting handles 98:2 imbalance
- How early stopping uses AUC-PR

Run: python learn/04_training_loop.py
"""

from tgn_learn.generators import BankSimGenerator
from tgn_learn.generators.base import GeneratorConfig
from tgn_learn.model import TGNConfig
from tgn_learn.training import TGNTrainer, TrainingConfig


def main():
    print("=" * 70)
    print("  04: Training Loop — From Data to Fraud Detector")
    print("=" * 70)
    print()

    # ─── Section 1: Generate Training Data ────────────────────────────────
    print("─── Step 1: Generate Training Data ───")
    print()

    gen_config = GeneratorConfig(
        num_accounts=200, num_merchants=30,
        num_transactions=3000, fraud_rate=0.05, seed=42,
    )
    graph = BankSimGenerator(gen_config).generate()
    print(f"  {graph.summary()}")
    print()

    # ─── Section 2: Understand the Loss Function ──────────────────────────
    print("─── Step 2: Understanding the Combined Loss ───")
    print()
    print("  The TGN uses TWO losses combined:")
    print()
    print("  1. CONTRASTIVE LINK LOSS (unsupervised):")
    print("     - For each real edge (src→dst), sample a random negative (src→rand)")
    print("     - Train model to score real edges higher than fake ones")
    print("     - This learns STRUCTURAL patterns (who transacts with whom)")
    print()
    print("  2. SUPERVISED NODE LOSS (supervised):")
    print("     - For nodes involved in labeled fraud, predict the label")
    print("     - Uses class-weighted BCE to handle 95:5 imbalance")
    print("     - This learns BEHAVIORAL patterns (fraud vs normal accounts)")
    print()
    print("  Combined: loss = 0.5 * link_loss + 0.5 * node_loss")
    print()

    # ─── Section 3: Configure and Train ───────────────────────────────────
    print("─── Step 3: Training ───")
    print()

    train_config = TrainingConfig(
        epochs=15,
        batch_size=200,
        learning_rate=1e-3,
        patience=8,
        link_loss_weight=0.5,
        node_loss_weight=0.5,
    )
    model_config = TGNConfig(memory_dim=64, embedding_dim=64, time_dim=32)

    trainer = TGNTrainer(train_config, model_config)
    results = trainer.train(graph, verbose=True)

    # ─── Section 4: Analyze Results ───────────────────────────────────────
    print()
    print("─── Step 4: Results Analysis ───")
    print()

    # Loss progression
    print("  Loss Progression:")
    for i, record in enumerate(results["history"]):
        bar = "█" * int(record["train_loss"] * 20)
        print(f"    Epoch {record['epoch']:2d}: {record['train_loss']:.4f} {bar}")

    print()
    print(f"  Best Validation: {results['best_metrics']}")
    print(f"  Test Results:    {results['test_metrics']}")
    print()

    # ─── Section 5: Why AUC-PR? ──────────────────────────────────────────
    print("─── Step 5: Why AUC-PR (not accuracy)? ───")
    print()
    print("  With 95% legitimate transactions, a model predicting 'always legit'")
    print("  gets 95% accuracy but catches ZERO fraud. We need better metrics:")
    print()
    print("  - AUC-PR: Area under Precision-Recall curve")
    print("    → Measures ability to FIND fraud without too many false alarms")
    print("    → Robust to class imbalance (unlike accuracy or AUC-ROC)")
    print()
    print("  - Precision: Of flagged transactions, how many are actually fraud?")
    print("  - Recall: Of all fraud, how much did we catch?")
    print("  - F1: Harmonic mean of precision and recall")
    print()

    # ─── Key Takeaways ───────────────────────────────────────────────────
    print("─── Key Takeaways ───")
    print()
    print("  1. Combined loss learns both structural and behavioral patterns")
    print("  2. Temporal splitting prevents the model from seeing the future")
    print("  3. Class weighting (pos_weight) is essential for imbalanced data")
    print("  4. AUC-PR is the right metric for fraud detection")
    print("  5. Early stopping prevents overfitting to training patterns")
    print()
    print("Next: 05_inference_scoring.py — Score transactions and calibrate")


if __name__ == "__main__":
    main()
