"""
Generate the demo checkpoint: checkpoints/demo_model.pt

Trains a TGN model on BankSim data with seed=42, 5K transactions, 20 epochs.
This checkpoint is used by the "Load Pre-trained Demo Model" button in the app.

Usage:
    python -m scripts.generate_demo_checkpoint
    # or
    python scripts/generate_demo_checkpoint.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tgn_learn.generators import BankSimGenerator
from tgn_learn.generators.base import GeneratorConfig
from tgn_learn.model import TGNConfig
from tgn_learn.training import TGNTrainer, TrainingConfig


def main():
    print("=" * 60)
    print("Generating demo checkpoint: checkpoints/demo_model.pt")
    print("=" * 60)

    # Generate data with seed=42, 5K transactions (matches DEMO_READINESS spec)
    config = GeneratorConfig(
        num_accounts=200,
        num_merchants=30,
        num_transactions=5000,
        fraud_rate=0.03,
        seed=42,
    )
    print(f"\n[1/3] Generating data: {config.num_transactions} transactions, "
          f"seed={config.seed}, fraud_rate={config.fraud_rate:.0%}")

    gen = BankSimGenerator(config)
    graph = gen.generate()
    print(f"  -> {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"{graph.num_fraud} fraud ({graph.fraud_rate:.1%})")

    # Train with 20 epochs
    train_config = TrainingConfig(
        epochs=20,
        batch_size=200,
        learning_rate=1e-3,
        patience=20,  # Don't early-stop for demo — run full 20 epochs
        checkpoint_dir="checkpoints",
        device="cpu",
    )
    model_config = TGNConfig(
        memory_dim=64,
        embedding_dim=64,
        time_encoder_type="pragma",
    )

    print(f"\n[2/3] Training: {train_config.epochs} epochs, "
          f"batch_size={train_config.batch_size}, lr={train_config.learning_rate}")

    trainer = TGNTrainer(train_config, model_config)
    results = trainer.train(graph, verbose=True)

    print(f"\n[3/3] Saving checkpoint...")

    # Save as demo_model.pt
    checkpoint_path = Path("checkpoints/demo_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save({
        "model_state_dict": results["model"].state_dict(),
        "model_config": model_config,
        "training_config": train_config,
        "num_nodes": graph.num_nodes,
        "best_val_metric": results["best_metrics"].auc_pr,
    }, str(checkpoint_path))

    print(f"  -> Saved to {checkpoint_path}")
    print(f"  -> Best val AUC-PR: {results['best_metrics'].auc_pr:.4f}")
    print(f"  -> Test AUC-PR: {results['test_metrics'].auc_pr:.4f}")
    print("\nDone! Demo checkpoint is ready.")


if __name__ == "__main__":
    main()
