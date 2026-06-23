"""Tests for TGN model components."""

import torch
import pytest

from tgn_learn.model import (
    TGNConfig,
    TimeEncoder,
    GraphAttentionEmbedding,
    LinkPredictor,
    NodeClassifier,
    LastNeighborLoader,
    TGNFraudDetector,
)


class TestTimeEncoder:
    """Tests for TimeEncoder."""

    def test_output_shape(self):
        enc = TimeEncoder(time_dim=32)
        t = torch.randn(10)
        out = enc(t)
        assert out.shape == (10, 32)

    def test_output_shape_2d_input(self):
        enc = TimeEncoder(time_dim=16)
        t = torch.randn(5, 1)
        out = enc(t)
        assert out.shape == (5, 16)

    def test_output_bounded(self):
        """Cosine activation output should be in [-1, 1]."""
        enc = TimeEncoder(time_dim=64)
        t = torch.randn(100) * 1e6
        out = enc(t)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_different_times_different_outputs(self):
        enc = TimeEncoder(time_dim=32)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1000.0])
        out1 = enc(t1)
        out2 = enc(t2)
        assert not torch.allclose(out1, out2)


class TestLinkPredictor:
    """Tests for LinkPredictor."""

    def test_output_shape(self):
        pred = LinkPredictor(in_channels=64)
        z_src = torch.randn(8, 64)
        z_dst = torch.randn(8, 64)
        out = pred(z_src, z_dst)
        assert out.shape == (8, 1)

    def test_gradient_flow(self):
        pred = LinkPredictor(in_channels=32)
        z_src = torch.randn(4, 32, requires_grad=True)
        z_dst = torch.randn(4, 32, requires_grad=True)
        out = pred(z_src, z_dst)
        loss = out.sum()
        loss.backward()
        assert z_src.grad is not None
        assert z_dst.grad is not None


class TestNodeClassifier:
    """Tests for NodeClassifier."""

    def test_output_shape(self):
        clf = NodeClassifier(in_channels=64, hidden=128)
        z = torch.randn(8, 64)
        out = clf(z)
        assert out.shape == (8, 1)

    def test_gradient_flow(self):
        clf = NodeClassifier(in_channels=32)
        z = torch.randn(4, 32, requires_grad=True)
        out = clf(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None


class TestLastNeighborLoader:
    """Tests for LastNeighborLoader."""

    def test_insert_and_query(self):
        loader = LastNeighborLoader(num_nodes=10, size=5)
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([3, 4, 5])
        loader.insert(src, dst)

        # Node 0 should have neighbor 3
        neighbors, edge_ids = loader(torch.tensor([0]))
        assert 3 in neighbors.tolist()

    def test_reset(self):
        loader = LastNeighborLoader(num_nodes=10, size=5)
        loader.insert(torch.tensor([0]), torch.tensor([1]))
        loader.reset()
        neighbors, _ = loader(torch.tensor([0]))
        assert len(neighbors) == 0

    def test_circular_buffer(self):
        loader = LastNeighborLoader(num_nodes=5, size=2)
        # Insert 3 neighbors for node 0 (buffer size = 2)
        loader.insert(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]))
        neighbors, _ = loader(torch.tensor([0]))
        # Should only keep 2 most recent
        assert len(neighbors) <= 2


class TestTGNFraudDetector:
    """Tests for the complete TGN model."""

    @pytest.fixture
    def model(self):
        config = TGNConfig(
            memory_dim=32, embedding_dim=32,
            time_dim=16, edge_feat_dim=20,
        )
        return TGNFraudDetector(num_nodes=100, config=config)

    def test_forward_shapes(self, model):
        batch_size = 8
        src = torch.randint(0, 50, (batch_size,))
        dst = torch.randint(50, 100, (batch_size,))
        t = torch.sort(torch.rand(batch_size) * 1000)[0]
        msg = torch.randn(batch_size, 20)
        neg_dst = torch.randint(0, 100, (batch_size,))

        pos_score, neg_score, node_scores = model(src, dst, t, msg, neg_dst)

        assert pos_score.shape == (batch_size, 1)
        assert neg_score.shape == (batch_size, 1)
        assert node_scores.shape == (batch_size, 1)

    def test_forward_without_negatives(self, model):
        batch_size = 4
        src = torch.randint(0, 50, (batch_size,))
        dst = torch.randint(50, 100, (batch_size,))
        t = torch.rand(batch_size) * 1000
        msg = torch.randn(batch_size, 20)

        pos_score, neg_score, node_scores = model(src, dst, t, msg)
        assert pos_score.shape == (batch_size, 1)
        assert neg_score is None
        assert node_scores.shape == (batch_size, 1)

    def test_memory_updates(self, model):
        """Memory should change after forward pass."""
        src = torch.tensor([0, 1])
        dst = torch.tensor([2, 3])
        t = torch.tensor([1.0, 2.0])
        msg = torch.randn(2, 20)

        # Get memory before
        z_before, _ = model.memory(torch.tensor([0, 1, 2, 3]))

        # Forward pass updates memory
        model(src, dst, t, msg)

        # Get memory after
        z_after, _ = model.memory(torch.tensor([0, 1, 2, 3]))

        # Memory should have changed for involved nodes
        assert not torch.allclose(z_before, z_after)

    def test_reset_memory(self, model):
        """Reset should restore memory to initial state."""
        # Get initial memory output (before any updates)
        z_initial, _ = model.memory(torch.tensor([0, 1]))

        src = torch.tensor([0])
        dst = torch.tensor([1])
        t = torch.tensor([1.0])
        msg = torch.randn(1, 20)

        model(src, dst, t, msg)

        # Memory should differ after update
        z_after, _ = model.memory(torch.tensor([0, 1]))
        assert not torch.allclose(z_initial, z_after)

        # After reset, memory should match initial state
        model.reset_memory()
        z_reset, _ = model.memory(torch.tensor([0, 1]))
        assert torch.allclose(z_initial, z_reset)

    def test_gradient_flow_through_model(self, model):
        """Gradients should flow through the complete model."""
        src = torch.randint(0, 50, (4,))
        dst = torch.randint(50, 100, (4,))
        t = torch.rand(4) * 100
        msg = torch.randn(4, 20)
        neg_dst = torch.randint(0, 100, (4,))

        pos_score, neg_score, node_scores = model(src, dst, t, msg, neg_dst)
        loss = pos_score.mean() + neg_score.mean() + node_scores.mean()
        loss.backward()

        # Check gradients exist for key parameters
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_total_parameters(self, model):
        """Model should report reasonable parameter count."""
        assert model.total_parameters > 0
        assert model.total_parameters < 1_000_000  # Should be small for learning config

    def test_overfit_tiny_graph(self, model):
        """Model should be able to reduce loss on a tiny dataset (sanity check)."""
        import torch.nn.functional as F

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Fixed tiny batch
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([4, 5, 6, 7])
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        msg = torch.randn(4, 20)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

        losses = []
        for epoch in range(30):
            model.reset_memory()
            optimizer.zero_grad()

            _, _, node_scores = model(src, dst, t, msg)
            loss = F.binary_cross_entropy_with_logits(
                node_scores.squeeze(), labels
            )
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            model.detach_memory()

        # Minimum loss across training should be less than first loss
        # (model should find at least one good configuration)
        assert min(losses) < losses[0]
