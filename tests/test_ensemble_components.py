"""Tests for ensemble TGN components (Phase 1).

Covers:
- MultiScaleTimeEncoder (Task 9)
- RFScoringHead (Task 10)
- Config + embedder wiring (Task 11)
"""

import numpy as np
import pytest
import torch

from tgn_learn.model import (
    TGNConfig,
    MultiScaleTimeEncoder,
    TimeEncoder,
    GraphAttentionEmbedding,
    RFScoringHead,
    TGNFraudDetector,
)


# ===========================================================================
# MultiScaleTimeEncoder Tests
# ===========================================================================


class TestMultiScaleTimeEncoder:
    """Tests for MultiScaleTimeEncoder."""

    def test_output_shape_1d(self):
        enc = MultiScaleTimeEncoder(time_dim=32)
        t = torch.randn(10)
        out = enc(t)
        assert out.shape == (10, 32)

    def test_output_shape_2d(self):
        enc = MultiScaleTimeEncoder(time_dim=16)
        t = torch.randn(5, 1)
        out = enc(t)
        assert out.shape == (5, 16)

    def test_output_same_dim_as_single_scale(self):
        """MultiScale output dim must match TimeEncoder for drop-in use."""
        time_dim = 32
        single = TimeEncoder(time_dim)
        multi = MultiScaleTimeEncoder(time_dim)
        t = torch.randn(7)
        assert single(t).shape == multi(t).shape

    def test_has_five_encoders(self):
        enc = MultiScaleTimeEncoder(time_dim=32)
        assert len(enc.encoders) == 5
        assert enc.num_scales == 5

    def test_scales_are_correct(self):
        assert MultiScaleTimeEncoder.SCALES == [
            60.0, 3600.0, 86400.0, 604800.0, 2592000.0
        ]

    def test_different_times_different_outputs(self):
        enc = MultiScaleTimeEncoder(time_dim=32)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([3600.0])  # 1 hour apart
        out1 = enc(t1)
        out2 = enc(t2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flow(self):
        enc = MultiScaleTimeEncoder(time_dim=32)
        t = torch.randn(4, requires_grad=True)
        out = enc(t)
        loss = out.sum()
        loss.backward()
        assert t.grad is not None
        assert t.grad.abs().sum() > 0

    def test_fusion_layer_shape(self):
        """Fusion linear should map (5 * time_dim) -> time_dim."""
        time_dim = 16
        enc = MultiScaleTimeEncoder(time_dim)
        assert enc.scale_fusion.in_features == 5 * time_dim
        assert enc.scale_fusion.out_features == time_dim

    def test_batch_consistency(self):
        """Same input should produce same output (deterministic)."""
        enc = MultiScaleTimeEncoder(time_dim=32)
        enc.eval()
        t = torch.tensor([100.0, 200.0, 300.0])
        out1 = enc(t)
        out2 = enc(t)
        assert torch.allclose(out1, out2)

    def test_large_timestamps(self):
        """Should handle timestamps spanning months without NaN."""
        enc = MultiScaleTimeEncoder(time_dim=32)
        t = torch.tensor([0.0, 86400.0, 604800.0, 2592000.0, 1e8])
        out = enc(t)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ===========================================================================
# RFScoringHead Tests
# ===========================================================================


class TestRFScoringHead:
    """Tests for RFScoringHead."""

    @pytest.fixture
    def sample_data(self):
        """Generate a simple linearly separable dataset."""
        rng = np.random.default_rng(42)
        n = 200
        # Fraud: high values in first 3 features
        X_fraud = rng.normal(loc=2.0, scale=0.5, size=(n // 10, 10))
        y_fraud = np.ones(n // 10)
        # Legit: low values
        X_legit = rng.normal(loc=-1.0, scale=0.5, size=(n - n // 10, 10))
        y_legit = np.zeros(n - n // 10)

        X = np.vstack([X_legit, X_fraud])
        y = np.concatenate([y_legit, y_fraud])
        return X, y

    def test_fit_and_predict(self, sample_data):
        X, y = sample_data
        head = RFScoringHead(n_estimators=50, random_state=42)
        head.fit(X, y)
        probs = head.predict_proba(X)
        assert probs.shape == (len(X),)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_is_fitted_flag(self):
        head = RFScoringHead()
        assert not head.is_fitted
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        head.fit(X, y)
        assert head.is_fitted

    def test_predict_before_fit_raises(self):
        head = RFScoringHead()
        with pytest.raises(RuntimeError, match="not been fitted"):
            head.predict_proba(np.random.randn(5, 10))

    def test_score_single_before_fit_raises(self):
        head = RFScoringHead()
        with pytest.raises(RuntimeError, match="not been fitted"):
            head.score_single(np.random.randn(10))

    def test_fit_empty_data_raises(self):
        head = RFScoringHead()
        with pytest.raises(ValueError, match="empty data"):
            head.fit(np.array([]).reshape(0, 5), np.array([]))

    def test_fit_mismatched_lengths_raises(self):
        head = RFScoringHead()
        with pytest.raises(ValueError, match="mismatch"):
            head.fit(np.random.randn(10, 5), np.random.randint(0, 2, 8))

    def test_feature_names_auto_generated(self, sample_data):
        X, y = sample_data
        head = RFScoringHead(n_estimators=10, random_state=42)
        head.fit(X, y)
        importances = head.get_feature_importances()
        assert len(importances) == 10
        assert "feat_0" in importances

    def test_custom_feature_names(self, sample_data):
        X, y = sample_data
        names = [f"custom_{i}" for i in range(10)]
        head = RFScoringHead(n_estimators=10, random_state=42)
        head.fit(X, y, feature_names=names)
        importances = head.get_feature_importances()
        assert "custom_0" in importances

    def test_feature_names_wrong_length_raises(self, sample_data):
        X, y = sample_data
        head = RFScoringHead()
        with pytest.raises(ValueError, match="feature_names length"):
            head.fit(X, y, feature_names=["a", "b"])

    def test_get_top_features(self, sample_data):
        X, y = sample_data
        head = RFScoringHead(n_estimators=50, random_state=42)
        head.fit(X, y)
        top = head.get_top_features(n=3)
        assert len(top) == 3
        # Should be sorted descending
        assert top[0][1] >= top[1][1] >= top[2][1]

    def test_predict_binary(self, sample_data):
        X, y = sample_data
        head = RFScoringHead(n_estimators=50, random_state=42)
        head.fit(X, y)
        preds = head.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_score_single(self, sample_data):
        X, y = sample_data
        head = RFScoringHead(n_estimators=50, random_state=42)
        head.fit(X, y)
        result = head.score_single(X[0])
        assert 0.0 <= result.fraud_probability <= 1.0
        assert len(result.feature_importances) == 10
        assert len(result.feature_names) == 10

    def test_class_imbalance_handling(self):
        """RF with balanced weights should handle 99:1 imbalance."""
        rng = np.random.default_rng(42)
        X_legit = rng.normal(0, 1, (990, 5))
        X_fraud = rng.normal(3, 1, (10, 5))
        X = np.vstack([X_legit, X_fraud])
        y = np.concatenate([np.zeros(990), np.ones(10)])

        head = RFScoringHead(n_estimators=100, random_state=42)
        head.fit(X, y)
        # Should detect at least some fraud (recall > 0)
        fraud_probs = head.predict_proba(X_fraud)
        assert fraud_probs.mean() > 0.3

    def test_save_and_load(self, sample_data, tmp_path):
        X, y = sample_data
        head = RFScoringHead(n_estimators=20, random_state=42)
        head.fit(X, y)

        path = tmp_path / "rf_head.joblib"
        head.save(path)

        loaded = RFScoringHead.load(path)
        assert loaded.is_fitted
        assert loaded.n_estimators == 20

        # Predictions should be identical
        orig_probs = head.predict_proba(X)
        loaded_probs = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(orig_probs, loaded_probs)

    def test_save_unfitted_raises(self, tmp_path):
        head = RFScoringHead()
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            head.save(tmp_path / "should_not_exist.joblib")

    def test_build_feature_names(self):
        names = RFScoringHead.build_feature_names(
            embedding_dim=64, edge_feat_dim=20
        )
        assert len(names) == 64 + 64 + 20  # z_src + z_dst + edge_feats
        assert names[0] == "z_src_0"
        assert names[63] == "z_src_63"
        assert names[64] == "z_dst_0"
        assert names[128] == "edge_feat_0"


# ===========================================================================
# Config + Wiring Tests
# ===========================================================================


class TestConfigWiring:
    """Tests for config flags and embedder wiring."""

    def test_config_defaults(self):
        config = TGNConfig()
        assert config.use_multiscale_time is True
        assert config.fit_rf_head is True
        assert config.rf_n_estimators == 200
        assert config.rf_max_depth == 10

    def test_config_override(self):
        config = TGNConfig(use_multiscale_time=False, fit_rf_head=False)
        assert config.use_multiscale_time is False
        assert config.fit_rf_head is False

    def test_embedder_uses_multiscale_when_enabled(self):
        emb = GraphAttentionEmbedding(
            in_channels=64, out_channels=64, msg_dim=20,
            time_dim=32, use_multiscale_time=True,
        )
        assert isinstance(emb.time_enc, MultiScaleTimeEncoder)

    def test_embedder_uses_single_scale_when_disabled(self):
        emb = GraphAttentionEmbedding(
            in_channels=64, out_channels=64, msg_dim=20,
            time_dim=32, use_multiscale_time=False,
        )
        assert isinstance(emb.time_enc, TimeEncoder)
        assert not isinstance(emb.time_enc, MultiScaleTimeEncoder)

    def test_tgn_detector_wires_multiscale(self):
        config = TGNConfig(
            memory_dim=32, embedding_dim=32,
            time_dim=16, edge_feat_dim=20,
            time_encoder_type="multiscale",
            use_multiscale_time=True,
        )
        model = TGNFraudDetector(num_nodes=50, config=config)
        assert isinstance(model.gnn.time_enc, MultiScaleTimeEncoder)

    def test_tgn_detector_wires_single_scale(self):
        config = TGNConfig(
            memory_dim=32, embedding_dim=32,
            time_dim=16, edge_feat_dim=20,
            time_encoder_type="fourier",
            use_multiscale_time=False,
        )
        model = TGNFraudDetector(num_nodes=50, config=config)
        assert isinstance(model.gnn.time_enc, TimeEncoder)
        assert not isinstance(model.gnn.time_enc, MultiScaleTimeEncoder)

    def test_tgn_forward_with_multiscale(self):
        """Full forward pass works with multiscale time encoder."""
        config = TGNConfig(
            memory_dim=32, embedding_dim=32,
            time_dim=16, edge_feat_dim=20,
            use_multiscale_time=True,
        )
        model = TGNFraudDetector(num_nodes=100, config=config)

        batch_size = 4
        src = torch.randint(0, 50, (batch_size,))
        dst = torch.randint(50, 100, (batch_size,))
        t = torch.rand(batch_size) * 1000
        msg = torch.randn(batch_size, 20)

        pos_score, neg_score, node_scores = model(src, dst, t, msg)
        assert pos_score.shape == (batch_size, 1)
        assert neg_score is None
        assert node_scores.shape == (batch_size, 1)

    def test_tgn_backward_with_multiscale(self):
        """Gradients flow through multiscale time encoder."""
        config = TGNConfig(
            memory_dim=32, embedding_dim=32,
            time_dim=16, edge_feat_dim=20,
            time_encoder_type="multiscale",
            use_multiscale_time=True,
        )
        model = TGNFraudDetector(num_nodes=100, config=config)

        src = torch.randint(0, 50, (4,))
        dst = torch.randint(50, 100, (4,))
        t = torch.rand(4) * 100
        msg = torch.randn(4, 20)
        neg_dst = torch.randint(0, 100, (4,))

        pos_score, neg_score, node_scores = model(src, dst, t, msg, neg_dst)
        loss = pos_score.mean() + neg_score.mean() + node_scores.mean()
        loss.backward()

        # Check that multiscale encoder parameters got gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.gnn.time_enc.parameters()
        )
        assert has_grad


# ===========================================================================
# Integration: RFScoringHead + TGN Embeddings
# ===========================================================================


class TestRFIntegration:
    """Integration test: RF head fitted on TGN-generated embeddings."""

    def test_end_to_end_rf_on_tgn_embeddings(self):
        """Simulate the post-training RF fitting workflow."""
        config = TGNConfig(
            memory_dim=32, embedding_dim=32,
            time_dim=16, edge_feat_dim=20,
            use_multiscale_time=True,
        )
        model = TGNFraudDetector(num_nodes=100, config=config)
        model.eval()

        # Simulate generating embeddings from model
        n_samples = 50
        with torch.no_grad():
            z_src = torch.randn(n_samples, config.embedding_dim)
            z_dst = torch.randn(n_samples, config.embedding_dim)
            edge_feats = torch.randn(n_samples, config.edge_feat_dim)

        # Build RF input: [z_src, z_dst, edge_features]
        X = np.hstack([
            z_src.numpy(), z_dst.numpy(), edge_feats.numpy()
        ])
        y = np.random.randint(0, 2, n_samples)

        # Fit RF head
        feature_names = RFScoringHead.build_feature_names(
            embedding_dim=config.embedding_dim,
            edge_feat_dim=config.edge_feat_dim,
        )
        head = RFScoringHead(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
        )
        head.fit(X, y, feature_names=feature_names)

        # Score
        probs = head.predict_proba(X)
        assert probs.shape == (n_samples,)
        assert 0.0 <= probs.min()
        assert probs.max() <= 1.0

        # Feature importances should include z_src, z_dst, edge_feat entries
        importances = head.get_feature_importances()
        assert "z_src_0" in importances
        assert "z_dst_0" in importances
        assert "edge_feat_0" in importances
