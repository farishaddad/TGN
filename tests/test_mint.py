"""Tests for MiNT multi-network training."""

import pytest

from tgn_learn.model import TGNConfig
from tgn_learn.training.mint import MiNTConfig, MiNTTrainer, MiNTResult


class TestMiNTTrainer:
    """Tests for MiNT multi-network training."""

    @pytest.fixture
    def small_config(self):
        """Minimal config for fast testing."""
        return MiNTConfig(
            num_train_networks=2,
            num_test_networks=1,
            epochs=2,
            batch_size=50,
            fine_tune_epochs=2,
            transactions_per_network=200,
            accounts_per_network=30,
            seed=42,
        )

    @pytest.fixture
    def small_model_config(self):
        return TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)

    def test_run_completes(self, small_config, small_model_config):
        """MiNT pipeline should run without errors."""
        trainer = MiNTTrainer(small_config, small_model_config)
        result = trainer.run(verbose=False)

        assert isinstance(result, MiNTResult)
        assert len(result.train_history) == 2
        assert len(result.zero_shot_metrics) == 1
        assert len(result.fine_tuned_metrics) == 1
        assert len(result.single_network_metrics) == 1

    def test_metrics_valid(self, small_config, small_model_config):
        """All output metrics should be in valid ranges."""
        trainer = MiNTTrainer(small_config, small_model_config)
        result = trainer.run(verbose=False)

        for name, m in result.zero_shot_metrics.items():
            assert 0.0 <= m.auc_pr <= 1.0
            assert 0.0 <= m.auc_roc <= 1.0

        for name, m in result.fine_tuned_metrics.items():
            assert 0.0 <= m.auc_pr <= 1.0

    def test_train_history_loss_finite(self, small_config, small_model_config):
        """Training losses should be finite."""
        trainer = MiNTTrainer(small_config, small_model_config)
        result = trainer.run(verbose=False)

        for record in result.train_history:
            assert record["avg_loss"] >= 0.0
            assert record["avg_loss"] < 100.0

    def test_summary_string(self, small_config, small_model_config):
        """Summary should produce readable output."""
        trainer = MiNTTrainer(small_config, small_model_config)
        result = trainer.run(verbose=False)

        s = result.summary()
        assert "MiNT" in s
        assert "Zero-shot" in s
        assert "fine-tuning" in s
