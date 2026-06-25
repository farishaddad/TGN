"""
Tests for TGN forward pass with use_profile_encoder=True.

Validates that the model completes a forward pass without error
when profile features are provided.
"""

import torch
import pytest

from tgn_learn.model import TGNConfig, TGNFraudDetector


class TestTGNWithProfile:
    """TGN forward pass with profile encoder enabled."""

    @pytest.fixture
    def model_with_profile(self):
        config = TGNConfig(
            memory_dim=32,
            embedding_dim=32,
            time_dim=16,
            edge_feat_dim=20,
            num_neighbors=5,
            num_heads=2,
            use_multiscale_time=False,
            time_encoder_type="pragma",
            use_profile_encoder=True,
            profile_dim=6,
            profile_encoder_dim=16,
        )
        num_nodes = 50
        model = TGNFraudDetector(num_nodes=num_nodes, config=config)

        # Set profile features for all nodes
        model.node_profiles = torch.randn(num_nodes, 6)

        return model

    @pytest.fixture
    def model_without_profile(self):
        config = TGNConfig(
            memory_dim=32,
            embedding_dim=32,
            time_dim=16,
            edge_feat_dim=20,
            num_neighbors=5,
            num_heads=2,
            use_multiscale_time=False,
            time_encoder_type="pragma",
            use_profile_encoder=False,
        )
        num_nodes = 50
        return TGNFraudDetector(num_nodes=num_nodes, config=config)

    def test_forward_with_profile(self, model_with_profile):
        """Forward pass completes without error."""
        model = model_with_profile
        batch_size = 8

        src = torch.randint(0, 30, (batch_size,))
        dst = torch.randint(30, 50, (batch_size,))
        t = torch.arange(batch_size, dtype=torch.float) * 100 + 1700000000
        msg = torch.randn(batch_size, 20)
        neg_dst = torch.randint(30, 50, (batch_size,))

        pos_score, neg_score, node_scores = model(src, dst, t, msg, neg_dst)

        assert pos_score.shape == (batch_size, 1)
        assert neg_score.shape == (batch_size, 1)
        assert node_scores.shape == (batch_size, 1)

    def test_forward_without_profile(self, model_without_profile):
        """Standard forward pass still works when profile is disabled."""
        model = model_without_profile
        batch_size = 8

        src = torch.randint(0, 30, (batch_size,))
        dst = torch.randint(30, 50, (batch_size,))
        t = torch.arange(batch_size, dtype=torch.float) * 100 + 1700000000
        msg = torch.randn(batch_size, 20)

        pos_score, _, node_scores = model(src, dst, t, msg)

        assert pos_score.shape == (batch_size, 1)
        assert node_scores.shape == (batch_size, 1)

    def test_gradients_flow_through_profile(self, model_with_profile):
        """Verify gradients flow through the profile encoder branch."""
        model = model_with_profile
        model.train()

        src = torch.randint(0, 30, (4,))
        dst = torch.randint(30, 50, (4,))
        t = torch.arange(4, dtype=torch.float) * 100 + 1700000000
        msg = torch.randn(4, 20)

        pos_score, _, node_scores = model(src, dst, t, msg)
        loss = pos_score.sum() + node_scores.sum()
        loss.backward()

        # Profile encoder parameters should have gradients
        for name, param in model.profile_encoder.named_parameters():
            assert param.grad is not None, f"No gradient for profile_encoder.{name}"

    def test_profile_affects_output(self, model_with_profile):
        """Different profile features should produce different scores."""
        model = model_with_profile
        model.eval()

        src = torch.tensor([0, 1])
        dst = torch.tensor([30, 31])
        t = torch.tensor([1700000000.0, 1700000100.0])
        msg = torch.randn(2, 20)

        # Run with original profiles
        model.reset_memory()
        pos1, _, _ = model(src, dst, t, msg)

        # Change profiles dramatically
        model.node_profiles = torch.randn(50, 6) * 10
        model.reset_memory()
        pos2, _, _ = model(src, dst, t, msg)

        # Outputs should differ (profile info changed)
        assert not torch.allclose(pos1, pos2, atol=1e-3)

    def test_backward_compat_no_profiles_set(self):
        """Model with use_profile_encoder=True but no profiles set should not crash."""
        config = TGNConfig(
            memory_dim=32,
            embedding_dim=32,
            time_dim=16,
            edge_feat_dim=20,
            num_neighbors=5,
            num_heads=2,
            time_encoder_type="fourier",
            use_profile_encoder=True,
            profile_dim=6,
            profile_encoder_dim=16,
        )
        model = TGNFraudDetector(num_nodes=50, config=config)
        # node_profiles is None — profile branch should be skipped gracefully

        src = torch.randint(0, 30, (4,))
        dst = torch.randint(30, 50, (4,))
        t = torch.arange(4, dtype=torch.float) * 100
        msg = torch.randn(4, 20)

        # Should complete without error (profile branch skipped when profiles is None)
        pos_score, _, node_scores = model(src, dst, t, msg)
        assert pos_score.shape == (4, 1)


class TestTGNWithPRAGMA:
    """TGN with PRAGMA time encoder (no profile encoder)."""

    def test_pragma_encoder_forward(self):
        config = TGNConfig(
            memory_dim=32,
            embedding_dim=32,
            time_dim=16,
            edge_feat_dim=20,
            num_neighbors=5,
            num_heads=2,
            time_encoder_type="pragma",
            use_profile_encoder=False,
        )
        model = TGNFraudDetector(num_nodes=50, config=config)

        src = torch.randint(0, 30, (8,))
        dst = torch.randint(30, 50, (8,))
        t = torch.arange(8, dtype=torch.float) * 100 + 1700000000
        msg = torch.randn(8, 20)

        pos_score, _, node_scores = model(src, dst, t, msg)
        assert pos_score.shape == (8, 1)
        assert node_scores.shape == (8, 1)

    def test_fourier_encoder_still_works(self):
        config = TGNConfig(
            memory_dim=32,
            embedding_dim=32,
            time_dim=16,
            edge_feat_dim=20,
            num_neighbors=5,
            num_heads=2,
            time_encoder_type="fourier",
            use_profile_encoder=False,
        )
        model = TGNFraudDetector(num_nodes=50, config=config)

        src = torch.randint(0, 30, (8,))
        dst = torch.randint(30, 50, (8,))
        t = torch.arange(8, dtype=torch.float) * 100
        msg = torch.randn(8, 20)

        pos_score, _, node_scores = model(src, dst, t, msg)
        assert pos_score.shape == (8, 1)
