"""Tests for inference and scoring module."""

import numpy as np
import pytest
import torch

from tgn_learn.graph import Edge, EDGE_FEAT_DIM
from tgn_learn.model import TGNConfig, TGNFraudDetector
from tgn_learn.scoring import Scorer, RiskTier, ScoringResult
from tgn_learn.scoring.scorer import RiskThresholds


@pytest.fixture
def scorer():
    """Create a scorer with a fresh model."""
    config = TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)
    model = TGNFraudDetector(num_nodes=100, config=config)
    return Scorer(model, device="cpu")


class TestScoringResult:
    """Tests for ScoringResult."""

    def test_to_dict(self):
        r = ScoringResult(risk_score=0.7, risk_tier=RiskTier.HIGH, raw_score=0.68)
        d = r.to_dict()
        assert d["risk_score"] == 0.7
        assert d["risk_tier"] == "HIGH"

    def test_str(self):
        r = ScoringResult(
            risk_score=0.7, risk_tier=RiskTier.HIGH, raw_score=0.68,
            confidence_lower=0.6, confidence_upper=0.8,
        )
        s = str(r)
        assert "HIGH" in s
        assert "0.7" in s


class TestScorer:
    """Tests for Scorer."""

    def test_score_single_transaction(self, scorer):
        result = scorer.score_transaction(
            src=0, dst=10, timestamp=1700000000.0, amount=100.0
        )
        assert isinstance(result, ScoringResult)
        assert 0.0 <= result.risk_score <= 1.0
        assert isinstance(result.risk_tier, RiskTier)

    def test_score_batch(self, scorer):
        edges = [
            Edge(src_id=0, dst_id=10, timestamp=1000.0),
            Edge(src_id=1, dst_id=11, timestamp=2000.0),
            Edge(src_id=2, dst_id=12, timestamp=3000.0),
        ]
        results = scorer.score_batch(edges)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.risk_score <= 1.0

    def test_score_batch_empty(self, scorer):
        results = scorer.score_batch([])
        assert results == []

    def test_batch_and_single_agree(self, scorer):
        """Batch and single scoring should produce the same results."""
        # Reset model memory for fair comparison
        scorer.model.reset_memory()
        edge = Edge(src_id=5, dst_id=15, timestamp=5000.0)
        single = scorer.score_transaction(
            src=5, dst=15, timestamp=5000.0, features=edge.features
        )

        scorer.model.reset_memory()
        batch = scorer.score_batch([edge])

        # Should be very close (floating point tolerance)
        assert abs(single.raw_score - batch[0].raw_score) < 1e-5

    def test_risk_tier_classification(self, scorer):
        """Verify tier boundaries."""
        # Manually test the classify method
        assert scorer._classify_tier(0.1) == RiskTier.LOW
        assert scorer._classify_tier(0.4) == RiskTier.MEDIUM
        assert scorer._classify_tier(0.7) == RiskTier.HIGH
        assert scorer._classify_tier(0.9) == RiskTier.CRITICAL

    def test_custom_thresholds(self):
        config = TGNConfig(memory_dim=16, embedding_dim=16, time_dim=8)
        model = TGNFraudDetector(num_nodes=50, config=config)
        thresholds = RiskThresholds(medium=0.2, high=0.5, critical=0.8)
        scorer = Scorer(model, thresholds=thresholds, device="cpu")

        assert scorer._classify_tier(0.15) == RiskTier.LOW
        assert scorer._classify_tier(0.3) == RiskTier.MEDIUM
        assert scorer._classify_tier(0.6) == RiskTier.HIGH
        assert scorer._classify_tier(0.85) == RiskTier.CRITICAL

    def test_calibration(self, scorer):
        """Calibration should make scores more monotonic."""
        # Simulate some validation data
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                          0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.98])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                          0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

        scorer.calibrate(scores, labels)
        assert scorer.calibrator is not None

        # Calibrated scores should still be in [0, 1]
        calibrated = scorer._calibrate_score(0.5)
        assert 0.0 <= calibrated <= 1.0

    def test_calibration_with_too_few_samples(self, scorer):
        """Calibration should be skipped with too few samples."""
        scores = np.array([0.5, 0.6])
        labels = np.array([0, 1])
        scorer.calibrate(scores, labels)
        assert scorer.calibrator is None  # Not enough data

    def test_confidence_bounds(self, scorer):
        """Confidence bounds should bracket the score."""
        result = scorer.score_transaction(src=0, dst=10, timestamp=1000.0, amount=500.0)
        assert result.confidence_lower <= result.risk_score
        assert result.confidence_upper >= result.risk_score
        assert result.confidence_lower >= 0.0
        assert result.confidence_upper <= 1.0

    def test_with_explicit_features(self, scorer):
        """Should accept explicit feature vectors."""
        features = np.random.randn(EDGE_FEAT_DIM).astype(np.float32)
        result = scorer.score_transaction(
            src=0, dst=10, timestamp=1000.0, features=features
        )
        assert 0.0 <= result.risk_score <= 1.0
