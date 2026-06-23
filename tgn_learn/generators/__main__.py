"""CLI entry point for generating synthetic fraud data.

Usage:
    python -m tgn_learn.generators --type banksim --accounts 100 --transactions 5000 --fraud-rate 0.02
"""

import argparse

from .base import GeneratorConfig
from .registry import GeneratorRegistry


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud data")
    parser.add_argument("--type", default="banksim", help="Generator type")
    parser.add_argument("--accounts", type=int, default=100)
    parser.add_argument("--merchants", type=int, default=20)
    parser.add_argument("--transactions", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = GeneratorConfig(
        num_accounts=args.accounts,
        num_merchants=args.merchants,
        num_transactions=args.transactions,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    gen = GeneratorRegistry.create(args.type, config)
    print(f"Generating with {gen.name} ({gen.description})...")
    print(f"  Patterns: {gen.fraud_patterns}")
    print()

    graph = gen.generate()
    print(graph.summary())


if __name__ == "__main__":
    main()
