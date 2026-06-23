"""CLI entry point for training.

Usage:
    python -m tgn_learn.train --epochs 10
"""

import argparse

from tgn_learn.generators import BankSimGenerator
from tgn_learn.generators.base import GeneratorConfig
from tgn_learn.model import TGNConfig
from tgn_learn.training import TGNTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train TGN fraud detector")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--transactions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Generate synthetic data
    print("=" * 60)
    print("TGN Fraud Detection — Training")
    print("=" * 60)
    print()

    gen_config = GeneratorConfig(
        num_accounts=100, num_merchants=20,
        num_transactions=args.transactions,
        fraud_rate=0.05, seed=args.seed,
    )
    gen = BankSimGenerator(gen_config)
    graph = gen.generate()
    print(f"Generated: {graph.summary()}")
    print()

    # Train
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model_config = TGNConfig()
    trainer = TGNTrainer(train_config, model_config)
    results = trainer.train(graph, verbose=True)

    print()
    print("=" * 60)
    print(f"Best validation: {results['best_metrics']}")
    print(f"Test results:    {results['test_metrics']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
