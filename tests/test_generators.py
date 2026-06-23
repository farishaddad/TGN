"""Tests for synthetic fraud data generators."""

import pytest

from tgn_learn.generators import BankSimGenerator, PaySimGenerator, GeneratorRegistry
from tgn_learn.generators.base import GeneratorConfig
from tgn_learn.graph import EDGE_FEAT_DIM


class TestBankSimGenerator:
    """Tests for BankSimGenerator."""

    def test_generates_valid_graph(self):
        config = GeneratorConfig(num_accounts=50, num_merchants=10, num_transactions=500, seed=42)
        gen = BankSimGenerator(config)
        graph = gen.generate()

        assert graph.num_nodes > 0
        assert graph.num_edges > 0
        assert graph.num_edges == 500

    def test_fraud_rate_approximate(self):
        config = GeneratorConfig(
            num_accounts=100, num_merchants=20,
            num_transactions=1000, fraud_rate=0.05, seed=42
        )
        gen = BankSimGenerator(config)
        graph = gen.generate()

        # Fraud rate should be close to target (within tolerance due to pattern injection)
        actual_rate = graph.fraud_rate
        assert 0.01 <= actual_rate <= 0.15  # Generous tolerance

    def test_node_types(self):
        config = GeneratorConfig(num_accounts=30, num_merchants=5, num_transactions=200, seed=7)
        gen = BankSimGenerator(config)
        graph = gen.generate()

        types = graph.node_types()
        assert "account" in types
        assert "merchant" in types
        assert types["account"] == 30
        assert types["merchant"] == 5

    def test_deterministic_with_seed(self):
        config = GeneratorConfig(num_accounts=20, num_merchants=5, num_transactions=100, seed=123)
        g1 = BankSimGenerator(config).generate()
        g2 = BankSimGenerator(config).generate()

        assert g1.num_edges == g2.num_edges
        assert g1.num_fraud == g2.num_fraud
        # Same edges in same order
        for e1, e2 in zip(g1.edges, g2.edges):
            assert e1.src_id == e2.src_id
            assert e1.dst_id == e2.dst_id
            assert e1.label == e2.label

    def test_edges_have_correct_features(self):
        config = GeneratorConfig(num_accounts=10, num_merchants=5, num_transactions=50, seed=1)
        gen = BankSimGenerator(config)
        graph = gen.generate()

        for edge in graph.edges:
            assert edge.features.shape == (EDGE_FEAT_DIM,)
            assert edge.features.dtype.name == "float32"

    def test_specific_pattern_only(self):
        config = GeneratorConfig(
            num_accounts=50, num_merchants=10,
            num_transactions=200, fraud_rate=0.1, seed=5
        )
        gen = BankSimGenerator(config, patterns=["card_testing"])
        graph = gen.generate()

        fraud_edges = [e for e in graph.edges if e.label == 1]
        assert len(fraud_edges) > 0
        for e in fraud_edges:
            assert e.metadata.get("pattern") == "card_testing"

    def test_all_patterns_represented(self):
        config = GeneratorConfig(
            num_accounts=100, num_merchants=20,
            num_transactions=2000, fraud_rate=0.10, seed=99
        )
        gen = BankSimGenerator(config)
        graph = gen.generate()

        patterns_seen = set()
        for e in graph.edges:
            if e.label == 1 and "pattern" in e.metadata:
                patterns_seen.add(e.metadata["pattern"])

        assert len(patterns_seen) == 5


class TestPaySimGenerator:
    """Tests for PaySimGenerator."""

    def test_generates_valid_graph(self):
        config = GeneratorConfig(num_accounts=50, num_transactions=500, seed=42)
        gen = PaySimGenerator(config)
        graph = gen.generate()

        assert graph.num_nodes > 0
        # Allow slight variance from fraud pattern chain lengths
        assert 490 <= graph.num_edges <= 510

    def test_p2p_transactions(self):
        config = GeneratorConfig(num_accounts=30, num_transactions=300, seed=7)
        gen = PaySimGenerator(config)
        graph = gen.generate()

        # PaySim uses account-to-account transfers
        edge_types = graph.edge_types()
        assert "transfer" in edge_types

    def test_deterministic_with_seed(self):
        config = GeneratorConfig(num_accounts=20, num_transactions=100, seed=55)
        g1 = PaySimGenerator(config).generate()
        g2 = PaySimGenerator(config).generate()

        assert g1.num_edges == g2.num_edges
        assert g1.num_fraud == g2.num_fraud

    def test_fraud_patterns_injected(self):
        config = GeneratorConfig(
            num_accounts=50, num_transactions=500, fraud_rate=0.08, seed=33
        )
        gen = PaySimGenerator(config)
        graph = gen.generate()

        fraud_edges = [e for e in graph.edges if e.label == 1]
        assert len(fraud_edges) > 0


class TestGeneratorRegistry:
    """Tests for GeneratorRegistry."""

    def test_list_generators(self):
        generators = GeneratorRegistry.list_generators()
        assert len(generators) >= 2
        names = [g["name"] for g in generators]
        assert "banksim" in names
        assert "paysim" in names

    def test_create_banksim(self):
        config = GeneratorConfig(seed=1)
        gen = GeneratorRegistry.create("banksim", config)
        assert gen.name == "banksim"

    def test_create_paysim(self):
        config = GeneratorConfig(seed=1)
        gen = GeneratorRegistry.create("paysim", config)
        assert gen.name == "paysim"

    def test_unknown_generator_raises(self):
        config = GeneratorConfig()
        with pytest.raises(KeyError, match="Unknown generator"):
            GeneratorRegistry.create("nonexistent", config)
