"""
BankSim Generator — Synthetic bank transaction data with 5 fraud patterns.

Generates a realistic temporal graph of bank transactions between accounts
and merchants, then injects configurable fraud patterns:

1. Account Takeover: Sudden burst of high-value transactions from a compromised account
2. Card Testing: Rapid small-value transactions to test stolen card validity
3. Money Laundering: Layered transactions through intermediary accounts
4. Synthetic Identity: New accounts with fabricated identities making immediate large purchases
5. Bust Out: Gradual build-up of legitimate activity followed by sudden maxed-out fraud

Adapted from SAIL_FSOS BankSimGenerator for learning purposes.
"""

from __future__ import annotations

import numpy as np

from tgn_learn.graph import Edge, Node, TemporalGraph, EDGE_FEAT_DIM
from .base import BaseGenerator, GeneratorConfig


class BankSimGenerator(BaseGenerator):
    """
    Synthetic bank transaction generator with 5 injectable fraud patterns.

    Each generated graph contains:
    - Account nodes (customers)
    - Merchant nodes (shops/services)
    - Transaction edges with temporal features (amount, time-of-day, channel, etc.)
    - Fraud labels injected according to selected patterns

    Example:
        >>> config = GeneratorConfig(num_accounts=100, num_transactions=5000, seed=42)
        >>> gen = BankSimGenerator(config)
        >>> graph = gen.generate()
        >>> print(f"Fraud rate: {graph.fraud_rate:.2%}")
    """

    name = "banksim"
    description = "Synthetic bank transactions with merchant category fraud patterns"
    fraud_patterns = [
        "account_takeover",
        "card_testing",
        "money_laundering",
        "synthetic_identity",
        "bust_out",
    ]

    def __init__(self, config: GeneratorConfig, patterns: list[str] | None = None):
        """
        Args:
            config: Generator configuration
            patterns: Which fraud patterns to inject. None = all patterns.
        """
        super().__init__(config)
        self.patterns = patterns or self.fraud_patterns

    def generate(self) -> TemporalGraph:
        """Generate a temporal graph with injected fraud patterns."""
        rng = np.random.default_rng(self.config.seed)
        graph = TemporalGraph()

        # --- Create nodes ---
        merchant_categories = [
            "grocery", "electronics", "restaurant", "gas_station",
            "online_retail", "travel", "entertainment", "healthcare",
        ]

        for i in range(self.config.num_accounts):
            graph.add_node(Node(
                node_id=i,
                node_type="account",
                metadata={
                    "balance": float(rng.lognormal(8, 1.5)),
                    "age_days": int(rng.exponential(365)),
                    "risk_score": float(rng.beta(2, 10)),
                },
            ))

        for i in range(self.config.num_merchants):
            mid = self.config.num_accounts + i
            graph.add_node(Node(
                node_id=mid,
                node_type="merchant",
                metadata={
                    "category": merchant_categories[i % len(merchant_categories)],
                    "avg_transaction": float(rng.lognormal(3, 1)),
                },
            ))

        # --- Generate legitimate transactions ---
        num_legit = int(self.config.num_transactions * (1 - self.config.fraud_rate))
        legit_edges = self._generate_legitimate(rng, num_legit)

        # --- Inject fraud patterns ---
        num_fraud = self.config.num_transactions - num_legit
        fraud_edges = self._inject_fraud_patterns(rng, num_fraud)

        # --- Add all edges ---
        graph.add_edges(legit_edges)
        graph.add_edges(fraud_edges)

        return graph

    def _generate_legitimate(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """Generate legitimate transaction edges with realistic patterns."""
        edges = []
        duration_seconds = self.config.duration_days * 86400
        merchant_ids = list(range(
            self.config.num_accounts,
            self.config.num_accounts + self.config.num_merchants,
        ))

        for _ in range(count):
            src = int(rng.integers(0, self.config.num_accounts))
            dst = int(rng.choice(merchant_ids))
            # Transactions cluster around business hours
            time_offset = self._sample_business_time(rng, duration_seconds)
            timestamp = self.config.start_timestamp + time_offset
            amount = float(rng.lognormal(3.5, 1.2))  # Median ~$33

            features = self._encode_features(
                rng, amount, timestamp, channel="pos", is_fraud=False
            )
            edges.append(Edge(
                src_id=src, dst_id=dst, timestamp=timestamp,
                features=features, label=0, edge_type="purchase",
            ))

        return edges

    def _inject_fraud_patterns(self, rng: np.random.Generator, total_fraud: int) -> list[Edge]:
        """Distribute fraud count across active patterns and inject."""
        if not self.patterns or total_fraud == 0:
            return []

        per_pattern = max(1, total_fraud // len(self.patterns))
        edges: list[Edge] = []

        for pattern in self.patterns:
            count = min(per_pattern, total_fraud - len(edges))
            if count <= 0:
                break

            if pattern == "account_takeover":
                edges.extend(self._pattern_account_takeover(rng, count))
            elif pattern == "card_testing":
                edges.extend(self._pattern_card_testing(rng, count))
            elif pattern == "money_laundering":
                edges.extend(self._pattern_money_laundering(rng, count))
            elif pattern == "synthetic_identity":
                edges.extend(self._pattern_synthetic_identity(rng, count))
            elif pattern == "bust_out":
                edges.extend(self._pattern_bust_out(rng, count))

        return edges

    def _pattern_account_takeover(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """
        Account Takeover: A compromised account suddenly makes multiple
        high-value transactions in a short time window.
        """
        edges = []
        # Pick a victim account
        victim = int(rng.integers(0, self.config.num_accounts))
        # Burst happens in a short window (1-2 hours)
        burst_start = self.config.start_timestamp + float(
            rng.uniform(0.3, 0.8) * self.config.duration_days * 86400
        )
        merchant_ids = list(range(
            self.config.num_accounts,
            self.config.num_accounts + self.config.num_merchants,
        ))

        for i in range(count):
            timestamp = burst_start + float(rng.uniform(0, 7200))  # Within 2 hours
            amount = float(rng.uniform(500, 5000))  # High-value
            dst = int(rng.choice(merchant_ids))
            features = self._encode_features(
                rng, amount, timestamp, channel="online", is_fraud=True
            )
            edges.append(Edge(
                src_id=victim, dst_id=dst, timestamp=timestamp,
                features=features, label=1, edge_type="purchase",
                metadata={"pattern": "account_takeover"},
            ))
        return edges

    def _pattern_card_testing(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """
        Card Testing: Rapid sequence of small transactions ($0.01-$5)
        to verify stolen card details before making larger purchases.
        """
        edges = []
        attacker = int(rng.integers(0, self.config.num_accounts))
        test_start = self.config.start_timestamp + float(
            rng.uniform(0.2, 0.7) * self.config.duration_days * 86400
        )
        merchant_ids = list(range(
            self.config.num_accounts,
            self.config.num_accounts + self.config.num_merchants,
        ))

        for i in range(count):
            timestamp = test_start + float(i * rng.uniform(5, 60))  # 5-60s apart
            amount = float(rng.uniform(0.01, 5.0))  # Tiny amounts
            dst = int(rng.choice(merchant_ids))
            features = self._encode_features(
                rng, amount, timestamp, channel="online", is_fraud=True
            )
            edges.append(Edge(
                src_id=attacker, dst_id=dst, timestamp=timestamp,
                features=features, label=1, edge_type="purchase",
                metadata={"pattern": "card_testing"},
            ))
        return edges

    def _pattern_money_laundering(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """
        Money Laundering: Layered transactions through intermediary accounts.
        Funds split and recombine through multiple hops.
        """
        edges = []
        # Create a chain: source -> intermediaries -> destination
        chain_length = min(count, 5)
        chain_accounts = rng.choice(self.config.num_accounts, size=chain_length + 1, replace=False)

        base_time = self.config.start_timestamp + float(
            rng.uniform(0.1, 0.6) * self.config.duration_days * 86400
        )
        total_amount = float(rng.uniform(10000, 50000))

        edges_per_hop = max(1, count // chain_length)
        for hop in range(chain_length):
            src = int(chain_accounts[hop])
            dst = int(chain_accounts[hop + 1])
            hop_amount = total_amount / (hop + 1)  # Split

            for j in range(min(edges_per_hop, count - len(edges))):
                timestamp = base_time + float((hop * 3600) + (j * rng.uniform(300, 1800)))
                amount = hop_amount * float(rng.uniform(0.8, 1.2))
                features = self._encode_features(
                    rng, amount, timestamp, channel="transfer", is_fraud=True
                )
                edges.append(Edge(
                    src_id=src, dst_id=dst, timestamp=timestamp,
                    features=features, label=1, edge_type="transfer",
                    metadata={"pattern": "money_laundering"},
                ))
                if len(edges) >= count:
                    break
            if len(edges) >= count:
                break
        return edges

    def _pattern_synthetic_identity(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """
        Synthetic Identity: Brand new accounts (young age) immediately make
        large purchases without building transaction history.
        """
        edges = []
        # Use a "new" account (high node_id range)
        fake_account = self.config.num_accounts - 1  # Latest account
        creation_time = self.config.start_timestamp + float(
            rng.uniform(0.5, 0.9) * self.config.duration_days * 86400
        )
        merchant_ids = list(range(
            self.config.num_accounts,
            self.config.num_accounts + self.config.num_merchants,
        ))

        for i in range(count):
            timestamp = creation_time + float(rng.uniform(0, 3600))  # Within first hour
            amount = float(rng.uniform(1000, 10000))  # Immediately large
            dst = int(rng.choice(merchant_ids))
            features = self._encode_features(
                rng, amount, timestamp, channel="online", is_fraud=True
            )
            edges.append(Edge(
                src_id=fake_account, dst_id=dst, timestamp=timestamp,
                features=features, label=1, edge_type="purchase",
                metadata={"pattern": "synthetic_identity"},
            ))
        return edges

    def _pattern_bust_out(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """
        Bust Out: Account builds legitimate history, then suddenly
        maxes out with multiple high-value fraudulent transactions.
        """
        edges = []
        bust_account = int(rng.integers(0, self.config.num_accounts))
        # Fraud burst at the end of the simulation
        bust_time = self.config.start_timestamp + float(
            0.85 * self.config.duration_days * 86400
        )
        merchant_ids = list(range(
            self.config.num_accounts,
            self.config.num_accounts + self.config.num_merchants,
        ))

        for i in range(count):
            timestamp = bust_time + float(i * rng.uniform(60, 600))
            amount = float(rng.uniform(2000, 15000))  # Max out
            dst = int(rng.choice(merchant_ids))
            features = self._encode_features(
                rng, amount, timestamp, channel="pos", is_fraud=True
            )
            edges.append(Edge(
                src_id=bust_account, dst_id=dst, timestamp=timestamp,
                features=features, label=1, edge_type="purchase",
                metadata={"pattern": "bust_out"},
            ))
        return edges

    # ------------------------------------------------------------------
    # Feature encoding helpers
    # ------------------------------------------------------------------

    def _encode_features(
        self,
        rng: np.random.Generator,
        amount: float,
        timestamp: float,
        channel: str = "pos",
        is_fraud: bool = False,
    ) -> np.ndarray:
        """Encode transaction features into a fixed-length vector.

        Feature layout (EDGE_FEAT_DIM=20):
            [0] log_amount
            [1] amount_normalized (capped at 1.0)
            [2] time_of_day_sin
            [3] time_of_day_cos
            [4] day_of_week_sin
            [5] day_of_week_cos
            [6] channel_encoding (pos=0.2, online=0.5, transfer=0.8, atm=0.9)
            [7] is_international (binary)
            [8] velocity_indicator (0-1)
            [9] amount_deviation (z-score proxy)
            [10-14] pattern indicators
            [15-19] noise/padding
        """
        feat = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)

        # Amount features
        feat[0] = np.log1p(amount)
        feat[1] = min(amount / 10000.0, 1.0)

        # Temporal features (cyclical encoding)
        hour_frac = (timestamp % 86400) / 86400
        feat[2] = np.sin(2 * np.pi * hour_frac)
        feat[3] = np.cos(2 * np.pi * hour_frac)
        day_frac = (timestamp % 604800) / 604800
        feat[4] = np.sin(2 * np.pi * day_frac)
        feat[5] = np.cos(2 * np.pi * day_frac)

        # Channel
        channel_map = {"pos": 0.2, "online": 0.5, "transfer": 0.8, "atm": 0.9}
        feat[6] = channel_map.get(channel, 0.5)

        # International flag
        feat[7] = float(rng.random() < 0.05)

        # Velocity indicator (higher for fraud)
        feat[8] = float(rng.beta(5, 2) if is_fraud else rng.beta(2, 5))

        # Amount deviation
        feat[9] = float(np.clip((np.log1p(amount) - 3.5) / 2.0, -2, 2))

        # Noise for remaining features
        feat[10:] = rng.normal(0, 0.1, size=EDGE_FEAT_DIM - 10).astype(np.float32)

        return feat

    def _sample_business_time(self, rng: np.random.Generator, duration_seconds: float) -> float:
        """Sample a timestamp biased toward business hours (9am-6pm)."""
        t = float(rng.uniform(0, duration_seconds))
        # Bias toward business hours with rejection sampling (simplified)
        hour = (t % 86400) / 3600
        if 9 <= hour <= 18:
            return t
        # 30% chance to keep off-hours transactions
        if rng.random() < 0.3:
            return t
        # Re-sample to business hours
        day_offset = t - (t % 86400)
        business_hour = float(rng.uniform(9, 18))
        return day_offset + business_hour * 3600
