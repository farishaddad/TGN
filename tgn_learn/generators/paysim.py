"""
PaySim Generator — Synthetic mobile money transaction data.

Simulates mobile payment networks with P2P transfers, cash-in/cash-out,
and payment transactions. Fraud patterns include:

- Money mule networks (rapid fund forwarding)
- Structuring (breaking large amounts into small pieces below thresholds)
- Account drainage (emptying compromised accounts)

Inspired by the PaySim simulator and adapted for TGN learning.
"""

from __future__ import annotations

import numpy as np

from tgn_learn.graph import Edge, Node, TemporalGraph, EDGE_FEAT_DIM
from .base import BaseGenerator, GeneratorConfig


class PaySimGenerator(BaseGenerator):
    """
    Synthetic mobile money transaction generator.

    Produces a temporal graph modeling a mobile payment ecosystem with
    P2P transfers, merchant payments, and cash operations.
    """

    name = "paysim"
    description = "Synthetic mobile money with P2P fraud patterns"
    fraud_patterns = ["money_mule", "structuring", "account_drainage"]

    def __init__(self, config: GeneratorConfig, patterns: list[str] | None = None):
        super().__init__(config)
        self.patterns = patterns or self.fraud_patterns

    def generate(self) -> TemporalGraph:
        """Generate a temporal graph with mobile money transactions."""
        rng = np.random.default_rng(self.config.seed)
        graph = TemporalGraph()

        # Create account nodes (agents in mobile money)
        for i in range(self.config.num_accounts):
            graph.add_node(Node(
                node_id=i,
                node_type="account",
                metadata={
                    "balance": float(rng.lognormal(7, 1.5)),
                    "account_type": rng.choice(["individual", "merchant", "agent"]),
                },
            ))

        # Generate legitimate transactions
        num_legit = int(self.config.num_transactions * (1 - self.config.fraud_rate))
        legit_edges = self._generate_legitimate(rng, num_legit)

        # Inject fraud
        num_fraud = self.config.num_transactions - num_legit
        fraud_edges = self._inject_fraud(rng, num_fraud)

        graph.add_edges(legit_edges)
        graph.add_edges(fraud_edges)
        return graph

    def _generate_legitimate(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """Generate legitimate mobile money transactions."""
        edges = []
        duration_seconds = self.config.duration_days * 86400
        tx_types = ["transfer", "payment", "cash_in", "cash_out"]
        tx_weights = [0.4, 0.3, 0.15, 0.15]

        for _ in range(count):
            src = int(rng.integers(0, self.config.num_accounts))
            dst = int(rng.integers(0, self.config.num_accounts))
            while dst == src:
                dst = int(rng.integers(0, self.config.num_accounts))

            timestamp = self.config.start_timestamp + float(
                rng.uniform(0, duration_seconds)
            )
            tx_type = str(rng.choice(tx_types, p=tx_weights))
            amount = float(rng.lognormal(4, 1.5))  # Median ~$55

            features = self._encode_features(rng, amount, timestamp, tx_type, False)
            edges.append(Edge(
                src_id=src, dst_id=dst, timestamp=timestamp,
                features=features, label=0, edge_type=tx_type,
            ))
        return edges

    def _inject_fraud(self, rng: np.random.Generator, total: int) -> list[Edge]:
        """Distribute fraud across patterns."""
        if not self.patterns or total == 0:
            return []

        per_pattern = max(1, total // len(self.patterns))
        edges: list[Edge] = []

        for pattern in self.patterns:
            count = min(per_pattern, total - len(edges))
            if count <= 0:
                break
            if pattern == "money_mule":
                edges.extend(self._pattern_money_mule(rng, count))
            elif pattern == "structuring":
                edges.extend(self._pattern_structuring(rng, count))
            elif pattern == "account_drainage":
                edges.extend(self._pattern_account_drainage(rng, count))

        return edges

    def _pattern_money_mule(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """Money Mule: Rapid forwarding of funds through mule accounts."""
        edges = []
        num_mules = min(3, self.config.num_accounts // 10)
        mule_chain = rng.choice(self.config.num_accounts, size=num_mules + 1, replace=False)
        base_time = self.config.start_timestamp + float(
            rng.uniform(0.3, 0.7) * self.config.duration_days * 86400
        )

        per_hop = max(1, count // num_mules)
        for hop in range(num_mules):
            src = int(mule_chain[hop])
            dst = int(mule_chain[hop + 1])
            for j in range(min(per_hop, count - len(edges))):
                timestamp = base_time + float(hop * 1800 + j * rng.uniform(30, 300))
                amount = float(rng.uniform(500, 5000))
                features = self._encode_features(rng, amount, timestamp, "transfer", True)
                edges.append(Edge(
                    src_id=src, dst_id=dst, timestamp=timestamp,
                    features=features, label=1, edge_type="transfer",
                    metadata={"pattern": "money_mule"},
                ))
                if len(edges) >= count:
                    return edges
        return edges

    def _pattern_structuring(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """Structuring: Break large amounts into pieces below $10K threshold."""
        edges = []
        structurer = int(rng.integers(0, self.config.num_accounts))
        base_time = self.config.start_timestamp + float(
            rng.uniform(0.2, 0.6) * self.config.duration_days * 86400
        )

        for i in range(count):
            dst = int(rng.integers(0, self.config.num_accounts))
            while dst == structurer:
                dst = int(rng.integers(0, self.config.num_accounts))
            timestamp = base_time + float(i * rng.uniform(3600, 86400))
            amount = float(rng.uniform(8000, 9999))  # Just under $10K
            features = self._encode_features(rng, amount, timestamp, "transfer", True)
            edges.append(Edge(
                src_id=structurer, dst_id=dst, timestamp=timestamp,
                features=features, label=1, edge_type="transfer",
                metadata={"pattern": "structuring"},
            ))
        return edges

    def _pattern_account_drainage(self, rng: np.random.Generator, count: int) -> list[Edge]:
        """Account Drainage: Compromised account emptied via cash_out."""
        edges = []
        victim = int(rng.integers(0, self.config.num_accounts))
        drain_time = self.config.start_timestamp + float(
            rng.uniform(0.5, 0.8) * self.config.duration_days * 86400
        )

        for i in range(count):
            dst = int(rng.integers(0, self.config.num_accounts))
            while dst == victim:
                dst = int(rng.integers(0, self.config.num_accounts))
            timestamp = drain_time + float(i * rng.uniform(60, 600))
            amount = float(rng.uniform(1000, 8000))
            features = self._encode_features(rng, amount, timestamp, "cash_out", True)
            edges.append(Edge(
                src_id=victim, dst_id=dst, timestamp=timestamp,
                features=features, label=1, edge_type="cash_out",
                metadata={"pattern": "account_drainage"},
            ))
        return edges

    def _encode_features(
        self, rng: np.random.Generator, amount: float,
        timestamp: float, tx_type: str, is_fraud: bool,
    ) -> np.ndarray:
        """Encode features for mobile money transactions."""
        feat = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)

        feat[0] = np.log1p(amount)
        feat[1] = min(amount / 10000.0, 1.0)

        hour_frac = (timestamp % 86400) / 86400
        feat[2] = np.sin(2 * np.pi * hour_frac)
        feat[3] = np.cos(2 * np.pi * hour_frac)
        day_frac = (timestamp % 604800) / 604800
        feat[4] = np.sin(2 * np.pi * day_frac)
        feat[5] = np.cos(2 * np.pi * day_frac)

        type_map = {"transfer": 0.3, "payment": 0.5, "cash_in": 0.7, "cash_out": 0.9}
        feat[6] = type_map.get(tx_type, 0.5)

        feat[7] = float(rng.random() < 0.02)  # Cross-border
        feat[8] = float(rng.beta(4, 2) if is_fraud else rng.beta(2, 5))
        feat[9] = float(np.clip((np.log1p(amount) - 4.0) / 2.0, -2, 2))
        feat[10:] = rng.normal(0, 0.1, size=EDGE_FEAT_DIM - 10).astype(np.float32)

        return feat
