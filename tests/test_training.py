"""Tests for training pipeline."""

import os
import tempfile

import numpy as np
import pytest
import torch

from tgn_learn.generators import BankSimGenerator
from tgn_learn.generators.base import GeneratorConfig
from tgn_learn.graph import Edge, TemporalGraph
from tgn_learn.model import TGNConfig
from tgn_learn.training import FraudMetrics, TGNTrainer, TrainingConfig


class TestFraudMetrics:
    """Tests for FraudMetrics."""

    def test_perfect_predictions(self):
        scores = np.array([0.9, 0.9, 0.1, 0.1])
        labels = np.array([1, 1, 0, 0])
        m = FraudMetrics.compute(scores, labels)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.auc_pr > 0.99

    def test_random_predictions(self):
        rng = np.random.default_rng(42)
        scores = rng.random(100)
        labels = rng.integers(0, 2, 100)
        m = FraudMetrics.compute(scores, labels)
        # Random should give metrics around 0.5
        assert 0.0 <= m.auc_pr <= 1.0
        assert 0.0 <= m.auc_roc <= 1.0

    def test_filters_unlabeled(self):
        scores = np.array([0.9, 0.1, 0.5, 0.8])
        labels = np.array([1, 0, -1, -1])  # Only first two are labeled
        m = FraudMetrics.compute(scores, labels)
        assert m.precision == 1.0
        assert m.recall == 1.0

    def test_all_unlabeled(self):
        scores = np.array([0.5, 0.5])
        labels = np.array([-1, -1])
        m = FraudMetrics.compute(scores, labels)
        assert m.auc_pr == 0.0

    def test_to_dict(self):
        m = FraudMetrics(precision=0.8, recall=0.7, f1=0.75, auc_pr=0.85, auc_roc=0.9)
        d = m.to_dict()
        assert d["precision"] == 0.8
        assert "auc_pr" in d

    def test_str_representation(self):
        m = FraudMetrics(auc_pr=0.85, auc_roc=0.9, precision=0.8, recall=0.7, f1=0.75)
        s = str(m)
        assert "0.8500" in s
        assert "0.9000" in s


class TestTGNTrainer:
    """Tests for training pipeline."""

    @pytest.fixture
    def small_graph(self):
        """Create a small synthetic graph for testing."""
        config = GeneratorConfig(
            num_accounts=30, num_merchants=10,
            num_transactions=300, fraud_rate=0.1, seed=42,
        )
        return BankSimGenerator(config).generate()

    def test_train_runs_without_error(self, small_graph):
        """Basic smoke test — training should complete."""
        train_config = TrainingConfig(epochs=3, batch_size=50, patience=5)
        model_config = TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)
        trainer = TGNTrainer(train_config, model_config)

        results = trainer.train(small_graph, verbose=False)

        assert "model" in results
        assert "history" in results
        assert "best_metrics" in results
        assert len(results["history"]) == 3

    def test_loss_is_finite(self, small_graph):
        """Training loss should be finite (no NaN/Inf)."""
        train_config = TrainingConfig(epochs=2, batch_size=100)
        model_config = TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)
        trainer = TGNTrainer(train_config, model_config)

        results = trainer.train(small_graph, verbose=False)

        for record in results["history"]:
            assert np.isfinite(record["train_loss"])

    def test_metrics_in_valid_range(self, small_graph):
        """All metrics should be in [0, 1]."""
        train_config = TrainingConfig(epochs=3, batch_size=50)
        model_config = TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)
        trainer = TGNTrainer(train_config, model_config)

        results = trainer.train(small_graph, verbose=False)
        m = results["best_metrics"]

        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.f1 <= 1.0
        assert 0.0 <= m.auc_pr <= 1.0

    def test_early_stopping(self, small_graph):
        """Early stopping should trigger before max epochs if patience exhausted."""
        train_config = TrainingConfig(epochs=100, batch_size=50, patience=2)
        model_config = TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)
        trainer = TGNTrainer(train_config, model_config)

        results = trainer.train(small_graph, verbose=False)

        # Should stop before 100 epochs
        assert len(results["history"]) < 100

    def test_checkpoint_save_load(self, small_graph):
        """Checkpoint save/load should preserve model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_config = TrainingConfig(
                epochs=2, batch_size=50, checkpoint_dir=tmpdir
            )
            model_config = TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)
            trainer = TGNTrainer(train_config, model_config)

            results = trainer.train(small_graph, verbose=False)

            # Load checkpoint
            checkpoint_path = os.path.join(tmpdir, "best_model.pt")
            assert os.path.exists(checkpoint_path)

            loaded_model = TGNTrainer.load_checkpoint(checkpoint_path, device="cpu")
            assert loaded_model.total_parameters == results["model"].total_parameters

    def test_callback_called(self, small_graph):
        """Callback should be called each epoch."""
        train_config = TrainingConfig(epochs=3, batch_size=100)
        model_config = TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)
        trainer = TGNTrainer(train_config, model_config)

        callback_records = []
        results = trainer.train(
            small_graph, verbose=False,
            callback=lambda r: callback_records.append(r),
        )

        assert len(callback_records) == 3
        assert "epoch" in callback_records[0]
        assert "train_loss" in callback_records[0]

    def test_temporal_split_preserves_ordering(self, small_graph):
        """Training data should be earlier than validation/test."""
        train_g, val_g, test_g = small_graph.temporal_split(0.7, 0.15)

        if train_g.num_edges > 0 and val_g.num_edges > 0:
            assert train_g.edges[-1].timestamp <= val_g.edges[0].timestamp
        if val_g.num_edges > 0 and test_g.num_edges > 0:
            assert val_g.edges[-1].timestamp <= test_g.edges[0].timestamp
