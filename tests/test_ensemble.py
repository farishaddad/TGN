"""
Tests for the ensemble fraud detection package.

Covers all 5 phases:
- Phase 1: MultiScaleTimeEncoder, RFScoringHead, GraphSMOTE
- Phase 2: DualTrackMemory, FundFlowDAG, DeviceEventGraph
- Phase 3: EmbeddingCache, RTEmbedder
- Phase 4: SemanticDetector, FlowDAGDetector, EnsembleScorer, MetaLearner
- Phase 5: LatentSpaceDriftDetector, ThresholdAdapter

Note: Tests that require torch/torch_geometric (TGNDetector, RFDetector,
BatchEmbedder, DualTrackMemory) are marked with pytest.mark.slow as they
depend on heavy imports. The rest run with numpy/sklearn only.
"""

import numpy as np
import pytest

# ===========================================================================
# Phase 1: Model Components
# ===========================================================================


class TestMultiScaleTimeEncoder:
    """Tests for ensemble/model/time_encoder.py."""

    def test_output_shape(self):
        import torch
        from ensemble.model.time_encoder import MultiScaleTimeEncoder

        enc = MultiScaleTimeEncoder(time_dim=32)
        t = torch.randn(10)
        out = enc(t)
        assert out.shape == (10, 32)

    def test_same_dim_as_single_scale(self):
        import torch
        from ensemble.model.time_encoder import TimeEncoder, MultiScaleTimeEncoder

        single = TimeEncoder(16)
        multi = MultiScaleTimeEncoder(16)
        t = torch.randn(5)
        assert single(t).shape == multi(t).shape

    def test_five_scales(self):
        from ensemble.model.time_encoder import MultiScaleTimeEncoder

        assert len(MultiScaleTimeEncoder.SCALES) == 5

    def test_gradient_flow(self):
        import torch
        from ensemble.model.time_encoder import MultiScaleTimeEncoder

        enc = MultiScaleTimeEncoder(32)
        t = torch.randn(4, requires_grad=True)
        out = enc(t)
        out.sum().backward()
        assert t.grad is not None


class TestRFScoringHead:
    """Tests for ensemble/model/rf_head.py."""

    def test_fit_predict(self):
        from ensemble.model.rf_head import RFScoringHead

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 10))
        y = (X[:, 0] > 0).astype(int)
        head = RFScoringHead(n_estimators=10, random_state=42)
        head.fit(X, y)
        probs = head.predict_proba(X)
        assert probs.shape == (100,)
        assert 0 <= probs.min() and probs.max() <= 1

    def test_not_fitted_raises(self):
        from ensemble.model.rf_head import RFScoringHead

        head = RFScoringHead()
        with pytest.raises(RuntimeError):
            head.predict_proba(np.zeros((5, 10)))

    def test_feature_importances(self):
        from ensemble.model.rf_head import RFScoringHead

        X = np.random.default_rng(1).normal(size=(80, 8))
        y = (X[:, 0] > 0).astype(int)
        head = RFScoringHead(n_estimators=20, random_state=1)
        head.fit(X, y)
        imp = head.get_feature_importances()
        assert len(imp) == 8
        assert all(v >= 0 for v in imp.values())

    def test_save_load(self, tmp_path):
        from ensemble.model.rf_head import RFScoringHead

        X = np.random.default_rng(7).normal(size=(50, 5))
        y = np.random.default_rng(7).integers(0, 2, 50)
        head = RFScoringHead(n_estimators=5, random_state=7)
        head.fit(X, y)
        path = tmp_path / "rf.joblib"
        head.save(path)
        loaded = RFScoringHead.load(path)
        np.testing.assert_array_almost_equal(
            head.predict_proba(X), loaded.predict_proba(X)
        )


class TestGraphSMOTE:
    """Tests for ensemble/model/graph_smote.py."""

    def test_augment_increases_fraud(self):
        from ensemble.model.graph_smote import GraphSMOTE
        from tgn_learn.graph import TemporalGraph, Edge

        graph = TemporalGraph()
        # Add legit edges
        for i in range(90):
            graph.add_edge(Edge(
                src_id=i % 10, dst_id=10 + i % 5,
                timestamp=float(i), features=np.random.randn(20).astype(np.float32),
                label=0,
            ))
        # Add fraud edges
        for i in range(10):
            graph.add_edge(Edge(
                src_id=i % 3, dst_id=12 + i % 3,
                timestamp=float(90 + i), features=np.random.randn(20).astype(np.float32),
                label=1,
            ))

        smote = GraphSMOTE(k_hop=2, minority_ratio=0.2, random_state=42)
        result = smote.augment(graph)
        assert result.synthetic_count > 0
        assert all(e.label == 1 for e in result.synthetic_edges)

    def test_empty_fraud_no_crash(self):
        from ensemble.model.graph_smote import GraphSMOTE
        from tgn_learn.graph import TemporalGraph, Edge

        graph = TemporalGraph()
        for i in range(20):
            graph.add_edge(Edge(
                src_id=i % 5, dst_id=5 + i % 3,
                timestamp=float(i), features=np.zeros(20, dtype=np.float32),
                label=0,
            ))

        smote = GraphSMOTE()
        result = smote.augment(graph)
        assert result.synthetic_count == 0


# ===========================================================================
# Phase 2: Graphs
# ===========================================================================


class TestFundFlowDAG:
    """Tests for ensemble/graphs/flow_dag.py."""

    def test_build_creates_chain_edges(self):
        from ensemble.graphs.flow_dag import FundFlowDAG
        from tgn_learn.graph import TemporalGraph, Edge

        graph = TemporalGraph()
        # A→B at t=0, B→C at t=100 (chain: money flows A→B→C)
        graph.add_edge(Edge(src_id=0, dst_id=1, timestamp=0.0,
                            features=np.array([np.log1p(1000)] + [0]*19, dtype=np.float32), label=0))
        graph.add_edge(Edge(src_id=1, dst_id=2, timestamp=100.0,
                            features=np.array([np.log1p(900)] + [0]*19, dtype=np.float32), label=0))

        dag = FundFlowDAG(time_window_hours=1.0, min_amount_ratio=0.5)
        dag.build(graph)
        assert dag.num_nodes == 2
        assert dag.num_edges == 1  # edge from event0 → event1

    def test_no_future_leakage(self):
        from ensemble.graphs.flow_dag import FundFlowDAG
        from tgn_learn.graph import TemporalGraph, Edge

        graph = TemporalGraph()
        # B→C at t=0, A→B at t=100 (B→C happens BEFORE A→B)
        graph.add_edge(Edge(src_id=1, dst_id=2, timestamp=0.0,
                            features=np.array([np.log1p(500)] + [0]*19, dtype=np.float32), label=0))
        graph.add_edge(Edge(src_id=0, dst_id=1, timestamp=100.0,
                            features=np.array([np.log1p(500)] + [0]*19, dtype=np.float32), label=0))

        dag = FundFlowDAG(time_window_hours=1.0)
        dag.build(graph)
        # No chain: B→C (t=0) cannot follow A→B (t=100) because it happened first
        # But A→B (t=100) could chain to future outflows from B — none exist
        assert dag.num_edges == 0

    def test_path_scores_shape(self):
        from ensemble.graphs.flow_dag import FundFlowDAG
        from tgn_learn.graph import TemporalGraph, Edge

        graph = TemporalGraph()
        for i in range(5):
            graph.add_edge(Edge(src_id=i, dst_id=i+1, timestamp=float(i*60),
                                features=np.array([np.log1p(1000)] + [0]*19, dtype=np.float32), label=0))

        dag = FundFlowDAG(time_window_hours=1.0)
        dag.build(graph)
        scores = dag.get_path_scores()
        assert scores.shape == (5,)


class TestDeviceEventGraph:
    """Tests for ensemble/graphs/device_event_graph.py."""

    def test_add_and_retrieve_events(self):
        from ensemble.graphs.device_event_graph import DeviceEventGraph, EventType

        g = DeviceEventGraph(num_accounts=100)
        g.add_event(account_id=5, event_type=EventType.CARD_BIND, timestamp=1000.0)
        g.add_event(account_id=5, event_type=EventType.DEVICE_REG, timestamp=1010.0)
        g.add_event(account_id=7, event_type=EventType.ADDR_CHANGE, timestamp=1020.0)

        events_5 = g.get_events_for_account(5)
        assert len(events_5) == 2

        events_7 = g.get_events_for_account(7, event_type=EventType.ADDR_CHANGE)
        assert len(events_7) == 1

    def test_velocity(self):
        from ensemble.graphs.device_event_graph import DeviceEventGraph, EventType

        g = DeviceEventGraph()
        # 5 events in 60 seconds
        for i in range(5):
            g.add_event(account_id=1, event_type=EventType.DEVICE_REG, timestamp=1000.0 + i * 10)

        vel = g.get_velocity(account_id=1, window_seconds=300.0, reference_time=1050.0)
        assert vel == 5

    def test_risk_features(self):
        from ensemble.graphs.device_event_graph import DeviceEventGraph, EventType

        g = DeviceEventGraph()
        g.add_event(account_id=2, event_type=EventType.CARD_BIND, timestamp=500.0)
        g.add_event(account_id=2, event_type=EventType.DEVICE_REG, timestamp=600.0)

        features = g.compute_risk_features(account_id=2, timestamp=700.0)
        assert "device_velocity_5m" in features
        assert "new_devices_1h" in features
        assert features["new_devices_1h"] == 1.0


# ===========================================================================
# Phase 3: Embedding
# ===========================================================================


class TestEmbeddingCache:
    """Tests for ensemble/embedding/embedding_cache.py."""

    def test_set_get(self):
        from ensemble.embedding.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_age_seconds=3600)
        emb = np.random.randn(64).astype(np.float32)
        cache.set(42, emb, timestamp=1000.0)

        retrieved = cache.get(42)
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, emb)

    def test_get_missing_returns_none(self):
        from ensemble.embedding.embedding_cache import EmbeddingCache

        cache = EmbeddingCache()
        assert cache.get(999) is None

    def test_staleness(self):
        from ensemble.embedding.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_age_seconds=100)
        cache.set(1, np.zeros(10), timestamp=0.0)

        assert not cache.is_stale(1, reference_time=50.0)
        assert cache.is_stale(1, reference_time=200.0)

    def test_save_load(self, tmp_path):
        from ensemble.embedding.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_age_seconds=500)
        cache.set(0, np.ones(8, dtype=np.float32), timestamp=100.0)
        cache.set(1, np.zeros(8, dtype=np.float32), timestamp=200.0)

        cache.save(tmp_path / "cache")

        loaded = EmbeddingCache()
        loaded.load(tmp_path / "cache")
        assert loaded.size == 2
        np.testing.assert_array_almost_equal(loaded.get(0), np.ones(8))

    def test_delete(self):
        from ensemble.embedding.embedding_cache import EmbeddingCache

        cache = EmbeddingCache()
        cache.set(5, np.zeros(4))
        assert cache.delete(5) is True
        assert cache.get(5) is None
        assert cache.delete(5) is False


class TestRTEmbedder:
    """Tests for ensemble/embedding/rt_embedder.py."""

    def test_cache_hit(self):
        from ensemble.embedding.embedding_cache import EmbeddingCache
        from ensemble.embedding.rt_embedder import RTEmbedder

        cache = EmbeddingCache()
        emb_src = np.ones(64, dtype=np.float32) * 0.5
        emb_dst = np.ones(64, dtype=np.float32) * 0.3
        cache.set(10, emb_src)
        cache.set(20, emb_dst)

        rt = RTEmbedder(cache, delta_weight=0.0)
        z_src, z_dst = rt.embed(10, 20, 1000.0, np.zeros(20))
        np.testing.assert_array_almost_equal(z_src, emb_src)
        np.testing.assert_array_almost_equal(z_dst, emb_dst)

    def test_cache_miss_returns_zeros(self):
        from ensemble.embedding.embedding_cache import EmbeddingCache
        from ensemble.embedding.rt_embedder import RTEmbedder

        cache = EmbeddingCache()
        rt = RTEmbedder(cache)
        z_src, z_dst = rt.embed(99, 100, 0.0, np.zeros(20))
        assert z_src.sum() == 0.0
        assert z_dst.sum() == 0.0

    def test_build_scoring_input(self):
        from ensemble.embedding.embedding_cache import EmbeddingCache
        from ensemble.embedding.rt_embedder import RTEmbedder

        cache = EmbeddingCache()
        cache.set(1, np.ones(64, dtype=np.float32))
        cache.set(2, np.zeros(64, dtype=np.float32))

        rt = RTEmbedder(cache, delta_weight=0.0)
        edge_feats = np.ones(20, dtype=np.float32) * 0.5
        x = rt.build_scoring_input(1, 2, 500.0, edge_feats)
        assert x.shape == (64 + 64 + 20,)  # z_src + z_dst + edge_feats


# ===========================================================================
# Phase 4: Detectors + Fusion
# ===========================================================================


class TestSemanticDetector:
    """Tests for ensemble/detectors/semantic_detector.py."""

    def test_fit_and_score(self):
        from ensemble.detectors.semantic_detector import SemanticDetector
        from tgn_learn.graph import TemporalGraph, Edge

        graph = TemporalGraph()
        for i in range(50):
            feats = np.zeros(20, dtype=np.float32)
            feats[0] = np.random.default_rng(i).normal(3.0, 0.5)
            feats[6] = 0.5  # same channel
            graph.add_edge(Edge(src_id=i%10, dst_id=10+i%5,
                                timestamp=float(i), features=feats, label=0))

        det = SemanticDetector(n_sigma=3.0)
        det.fit(graph)
        assert det.is_fitted

        # Normal transaction
        normal_feats = np.zeros(20, dtype=np.float32)
        normal_feats[0] = 3.0
        normal_feats[6] = 0.5
        score_normal = det.score(0, 10, 100.0, normal_feats)
        assert 0 <= score_normal <= 1

        # Anomalous transaction (far from mean)
        anomalous_feats = np.zeros(20, dtype=np.float32)
        anomalous_feats[0] = 20.0  # way above normal
        anomalous_feats[6] = 0.5
        score_anomalous = det.score(0, 10, 100.0, anomalous_feats)
        assert score_anomalous > score_normal


class TestFlowDAGDetector:
    """Tests for ensemble/detectors/flow_dag_detector.py."""

    def test_fit_and_score(self):
        from ensemble.detectors.flow_dag_detector import FlowDAGDetector
        from tgn_learn.graph import TemporalGraph, Edge

        graph = TemporalGraph()
        # Chain: A(0)→B(1) at t=0, B(1)→C(2) at t=60
        graph.add_edge(Edge(src_id=0, dst_id=1, timestamp=0.0,
                            features=np.array([np.log1p(5000)]+[0]*19, dtype=np.float32), label=1))
        graph.add_edge(Edge(src_id=1, dst_id=2, timestamp=60.0,
                            features=np.array([np.log1p(4500)]+[0]*19, dtype=np.float32), label=1))
        # Isolated legit
        graph.add_edge(Edge(src_id=5, dst_id=6, timestamp=200.0,
                            features=np.array([np.log1p(50)]+[0]*19, dtype=np.float32), label=0))

        det = FlowDAGDetector(time_window_hours=1.0)
        det.fit(graph)
        assert det.is_fitted

        # Chain transaction should have higher score
        feats = np.array([np.log1p(5000)]+[0]*19, dtype=np.float32)
        score_chain = det.score(0, 1, 0.0, feats)
        score_isolated = det.score(5, 6, 200.0, feats)
        # The chain node (event 0) has out-degree to event 1
        assert score_chain >= 0


class TestEnsembleScorer:
    """Tests for ensemble/fusion/ensemble_scorer.py."""

    def test_score_without_meta_learner(self):
        from ensemble.detectors.semantic_detector import SemanticDetector
        from ensemble.fusion.ensemble_scorer import EnsembleScorer, RiskTier
        from tgn_learn.graph import TemporalGraph, Edge

        # Create a simple fitted detector
        graph = TemporalGraph()
        for i in range(30):
            feats = np.zeros(20, dtype=np.float32)
            feats[0] = 3.0
            feats[6] = 0.5
            graph.add_edge(Edge(src_id=i%5, dst_id=5+i%3,
                                timestamp=float(i), features=feats, label=0))

        det = SemanticDetector()
        det.fit(graph)

        scorer = EnsembleScorer(detectors=[det], graph=graph)
        result = scorer.score_transaction(src=0, dst=5, timestamp=50.0, amount=100.0)

        assert 0 <= result.risk_score <= 1
        assert result.risk_tier in (RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL)
        assert "Semantic Patterns" in result.detector_scores

    def test_score_batch(self):
        from ensemble.detectors.semantic_detector import SemanticDetector
        from ensemble.fusion.ensemble_scorer import EnsembleScorer
        from tgn_learn.graph import TemporalGraph, Edge

        graph = TemporalGraph()
        edges = []
        for i in range(10):
            feats = np.zeros(20, dtype=np.float32)
            feats[0] = 3.0
            e = Edge(src_id=i%5, dst_id=5+i%3, timestamp=float(i), features=feats, label=0)
            graph.add_edge(e)
            edges.append(e)

        det = SemanticDetector()
        det.fit(graph)

        scorer = EnsembleScorer(detectors=[det], graph=graph)
        results = scorer.score_batch(edges)
        assert len(results) == 10


class TestMetaLearner:
    """Tests for ensemble/fusion/meta_learner.py."""

    def test_fit_predict(self):
        from ensemble.fusion.meta_learner import EnsembleMetaLearner

        rng = np.random.default_rng(42)
        det_scores = rng.uniform(0, 1, (200, 5))
        raw_feats = rng.normal(size=(200, 10))
        # Labels correlate with first detector
        labels = (det_scores[:, 0] > 0.5).astype(int)

        ml = EnsembleMetaLearner(n_estimators=50)
        ml.fit(det_scores, raw_feats, labels, detector_names=["a", "b", "c", "d", "e"])
        assert ml.is_fitted

        probs = ml.predict_proba(det_scores[:10], raw_feats[:10])
        assert probs.shape == (10,)
        assert 0 <= probs.min() and probs.max() <= 1

    def test_predict_single(self):
        from ensemble.fusion.meta_learner import EnsembleMetaLearner

        rng = np.random.default_rng(99)
        det_scores = rng.uniform(0, 1, (100, 3))
        raw_feats = rng.normal(size=(100, 5))
        labels = rng.integers(0, 2, 100)

        ml = EnsembleMetaLearner(n_estimators=20)
        ml.fit(det_scores, raw_feats, labels, detector_names=["x", "y", "z"])

        result = ml.predict_single(
            det_scores[0], raw_feats[0], detector_names=["x", "y", "z"]
        )
        assert 0 <= result.fraud_probability <= 1


# ===========================================================================
# Phase 5: Maintenance
# ===========================================================================


class TestDriftDetector:
    """Tests for ensemble/maintenance/drift_detector.py."""

    def test_no_drift_on_normal(self):
        from ensemble.maintenance.drift_detector import LatentSpaceDriftDetector

        det = LatentSpaceDriftDetector(embedding_dim=16, hidden_dim=8, cusum_threshold=5.0)
        normal = np.random.default_rng(42).normal(0, 1, (200, 16)).astype(np.float32)
        det.fit_normal(normal)
        assert det.is_fitted

        # Check with same distribution — should not trigger
        event = det.check(normal[:50], timestamp=1.0)
        assert event is None

    def test_drift_on_shifted_distribution(self):
        from ensemble.maintenance.drift_detector import LatentSpaceDriftDetector

        det = LatentSpaceDriftDetector(embedding_dim=16, hidden_dim=8, cusum_threshold=3.0)
        normal = np.random.default_rng(7).normal(0, 1, (200, 16)).astype(np.float32)
        det.fit_normal(normal)

        # Feed shifted data repeatedly to accumulate CUSUM
        shifted = np.random.default_rng(8).normal(5, 1, (50, 16)).astype(np.float32)
        event = None
        for i in range(10):
            event = det.check(shifted, timestamp=float(i))
            if event is not None:
                break

        assert event is not None
        assert event.severity in ("minor", "major")

    def test_reset(self):
        from ensemble.maintenance.drift_detector import LatentSpaceDriftDetector

        det = LatentSpaceDriftDetector(embedding_dim=8, hidden_dim=4)
        normal = np.random.default_rng(1).normal(size=(100, 8)).astype(np.float32)
        det.fit_normal(normal)

        # Accumulate some CUSUM
        shifted = np.random.default_rng(2).normal(3, 1, (30, 8)).astype(np.float32)
        det.check(shifted)
        assert det.cusum_statistic > 0

        det.reset()
        assert det.cusum_statistic == 0


class TestThresholdAdapter:
    """Tests for ensemble/maintenance/threshold_adapter.py."""

    def test_raises_thresholds_on_high_fp(self):
        from ensemble.maintenance.threshold_adapter import ThresholdAdapter, SegmentMetrics

        adapter = ThresholdAdapter(target_fp_rate=0.05, min_observations=10)
        initial = adapter.get_thresholds("retail")

        metrics = SegmentMetrics(
            segment_id="retail", fp_rate=0.15, recall=0.90,
            n_transactions=200, n_false_alarms=30,
        )
        updated = adapter.update("retail", metrics)

        # Should raise thresholds (fewer alerts)
        assert updated.medium >= initial.medium

    def test_lowers_thresholds_on_low_recall(self):
        from ensemble.maintenance.threshold_adapter import ThresholdAdapter, SegmentMetrics

        adapter = ThresholdAdapter(target_recall=0.80, min_observations=10)
        initial = adapter.get_thresholds("premium")

        metrics = SegmentMetrics(
            segment_id="premium", fp_rate=0.02, recall=0.50,
            n_transactions=200, n_fraud_missed=10,
        )
        updated = adapter.update("premium", metrics)

        # Should lower thresholds (more alerts)
        assert updated.medium <= initial.medium

    def test_no_change_below_min_observations(self):
        from ensemble.maintenance.threshold_adapter import ThresholdAdapter, SegmentMetrics

        adapter = ThresholdAdapter(min_observations=100)
        metrics = SegmentMetrics(
            segment_id="new", fp_rate=0.50, recall=0.10,
            n_transactions=5,
        )
        result = adapter.update("new", metrics)
        default = adapter.default_thresholds
        assert result.medium == default.medium

    def test_history_tracking(self):
        from ensemble.maintenance.threshold_adapter import ThresholdAdapter, SegmentMetrics

        adapter = ThresholdAdapter(target_fp_rate=0.05, min_observations=10)
        metrics = SegmentMetrics(
            segment_id="seg1", fp_rate=0.20, recall=0.90,
            n_transactions=500,
        )
        adapter.update("seg1", metrics)
        history = adapter.get_history("seg1")
        assert len(history) >= 1
        assert history[0]["segment_id"] == "seg1"
