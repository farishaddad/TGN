#!/usr/bin/env python3
"""
Script 06: MiNT Transfer Learning
===================================

Demonstrates Multi-Network Training — training a shared TGN across
multiple synthetic fraud networks, then evaluating zero-shot transfer
and fine-tuning on unseen networks. You'll learn:

- Why fraud patterns transfer across networks
- How gradient accumulation works across multiple graphs
- Zero-shot evaluation on completely unseen networks
- Fine-tuning: freeze backbone, adapt heads to new network

Run: python learn/06_mint_transfer.py
"""

from tgn_learn.model import TGNConfig
from tgn_learn.training.mint import MiNTConfig, MiNTTrainer


def main():
    print("=" * 70)
    print("  06: MiNT — Multi-Network Transfer Learning")
    print("=" * 70)
    print()

    # ─── Concept Introduction ─────────────────────────────────────────────
    print("─── The MiNT Concept ───")
    print()
    print("  Key insight: Fraud patterns are universal across networks!")
    print("  Account takeover looks similar whether it's a bank, mobile money,")
    print("  or cryptocurrency network. MiNT exploits this by training ONE model")
    print("  across MANY networks simultaneously.")
    print()
    print("  Pipeline:")
    print("    1. Generate N training networks (different seeds/generators)")
    print("    2. Train shared model: accumulate gradients across all networks")
    print("    3. Evaluate zero-shot on unseen test networks")
    print("    4. Fine-tune: freeze memory+GNN, retrain only scoring heads")
    print("    5. Compare against single-network baselines")
    print()
    print("  Expected outcome: MiNT model generalizes better than models")
    print("  trained on individual networks, especially with fine-tuning.")
    print()

    # ─── Run MiNT Pipeline ────────────────────────────────────────────────
    print("─── Running MiNT Pipeline ───")
    print()

    mint_config = MiNTConfig(
        num_train_networks=5,
        num_test_networks=2,
        epochs=8,
        batch_size=200,
        fine_tune_epochs=5,
        transactions_per_network=1500,
        accounts_per_network=100,
        fraud_rate=0.05,
        seed=42,
    )

    model_config = TGNConfig(memory_dim=32, embedding_dim=32, time_dim=16)
    trainer = MiNTTrainer(mint_config, model_config)

    result = trainer.run(verbose=True)

    # ─── Analysis ─────────────────────────────────────────────────────────
    print()
    print("─── Transfer Learning Analysis ───")
    print()
    print("  Comparing approaches:")
    print()
    print(f"  {'Network':<25} {'Zero-Shot':>12} {'Fine-Tuned':>12} {'From Scratch':>12}")
    print(f"  {'-'*65}")

    for name in result.zero_shot_metrics:
        zs = result.zero_shot_metrics[name].auc_pr
        ft = result.fine_tuned_metrics.get(name, result.zero_shot_metrics[name]).auc_pr
        sn = result.single_network_metrics.get(name, result.zero_shot_metrics[name]).auc_pr
        print(f"  {name:<25} {zs:>10.4f}   {ft:>10.4f}   {sn:>10.4f}")

    print()
    print("  Interpretation:")
    print("  - Zero-shot: Model has NEVER seen this network's data")
    print("  - Fine-tuned: Scoring heads adapted with a few epochs on target")
    print("  - From scratch: Trained only on this one network (baseline)")
    print()
    print("  When fine-tuned > from scratch, it means the shared backbone")
    print("  learned useful representations that transfer across domains.")
    print()

    # ─── Key Takeaways ───────────────────────────────────────────────────
    print("─── Key Takeaways ───")
    print()
    print("  1. Fraud patterns transfer across networks (temporal burst,")
    print("     layered flows, velocity anomalies are universal)")
    print("  2. Multi-network training learns more robust features")
    print("  3. Fine-tuning adapts quickly to new domains")
    print("  4. Freezing backbone prevents catastrophic forgetting")
    print("  5. This approach scales to 84+ networks in production (ScalingTGNs)")
    print()
    print("Congratulations! You've completed the TGN learning journey.")
    print("Try the Streamlit app for interactive exploration: streamlit run app/main.py")


if __name__ == "__main__":
    main()
