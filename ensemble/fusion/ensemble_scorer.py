"""
Ensemble Scorer — orchestrates the full multi-detector scoring pipeline.

Orchestrates: embedding cache → parallel detectors → meta-learner
→ two-hurdle filter → risk tier classification.

Drop-in replacement for tgn_learn.scoring.Scorer in ensemble mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from tgn_learn.graph import TemporalGraph, Edge, EDGE_FEAT_DIM

from ..config import EnsembleConfig, RiskThresholds
from ..detectors.base import BaseDetector
from .meta_learner import EnsembleMetaLearner


class RiskTier(str, Enum):
    """Risk classification tiers (mirrors tgn_learn.scoring.RiskTier)."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class EnsembleScoringResult:
    """Result from ensemble scoring.

    Attributes:
        risk_score: Final calibrated fraud probability [0, 1]
        risk_tier: Classification tier
        detector_scores: Per-detector raw scores
        meta_learner_score: Meta-learner output (if active)
        two_hurdle_result: FP filter result ('HIGH'/'MEDIUM'/'PASS')
        confidence_lower: Lower confidence bound
        confidence_upper: Upper confidence bound
    """

    risk_score: float
    risk_tier: RiskTier
    detector_scores: dict[str, float] = field(default_factory=dict)
    meta_learner_score: Optional[float] = None
    two_hurdle_result: str = "PASS"
    confidence_lower: float = 0.0
    confidence_upper: float = 1.0

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier.value,
            "detector_scores": self.detector_scores,
            "meta_learner_score": self.meta_learner_score,
            "two_hurdle_result": self.two_hurdle_result,
        }


class EnsembleScorer:
    """Full ensemble scoring pipeline.

    Orchestrates parallel detectors, optional meta-learner, FP filter,
    and risk tier classification.

    Args:
        detectors: List of fitted BaseDetector instances
        meta_learner: Optional fitted meta-learner (uses averaging if None)
        config: Ensemble configuration
        thresholds: Risk tier thresholds
        graph: Optional graph for context-aware scoring
    """

    def __init__(
        self,
        detectors: list[BaseDetector],
        meta_learner: Optional[EnsembleMetaLearner] = None,
        config: Optional[EnsembleConfig] = None,
        thresholds: Optional[RiskThresholds] = None,
        graph: Optional[TemporalGraph] = None,
    ):
        self.detectors = detectors
        self.meta_learner = meta_learner
        self.config = config or EnsembleConfig()
        self.thresholds = thresholds or RiskThresholds()
        self.graph = graph

    def score_transaction(
        self,
        src: int,
        dst: int,
        timestamp: float,
        features: Optional[np.ndarray] = None,
        amount: Optional[float] = None,
    ) -> EnsembleScoringResult:
        """Score a transaction through the full ensemble pipeline.

        Args:
            src: Source node ID
            dst: Destination node ID
            timestamp: Transaction timestamp
            features: Edge feature vector (built from amount if None)
            amount: Transaction amount

        Returns:
            EnsembleScoringResult with full breakdown
        """
        if features is None:
            features = self._build_features(amount or 0.0, timestamp)

        # --- Run all detectors in parallel ---
        detector_scores: dict[str, float] = {}
        for detector in self.detectors:
            try:
                score = detector.score(
                    src, dst, timestamp, features, self.graph
                )
                detector_scores[detector.name] = float(score)
            except Exception:
                detector_scores[detector.name] = 0.0

        # --- Fuse scores ---
        if self.meta_learner and self.meta_learner.is_fitted:
            scores_array = np.array(list(detector_scores.values())).reshape(1, -1)
            features_array = features.reshape(1, -1)
            meta_score = float(
                self.meta_learner.predict_proba(scores_array, features_array)[0]
            )
        else:
            # Simple weighted average (TGN gets higher weight)
            weights = {"TGN Memory": 0.35, "RF Structural": 0.2,
                       "Fund-Flow Graph": 0.25, "Semantic Patterns": 0.1,
                       "Drift Monitor": 0.1}
            total_weight = 0.0
            weighted_sum = 0.0
            for name, score in detector_scores.items():
                w = weights.get(name, 0.15)
                weighted_sum += w * score
                total_weight += w
            meta_score = weighted_sum / (total_weight + 1e-8)

        # --- Two-hurdle FP filter ---
        two_hurdle = "PASS"
        if self.config.use_two_hurdle_filter:
            two_hurdle = self._two_hurdle_filter(meta_score, detector_scores)

        # --- Risk tier classification ---
        final_score = meta_score
        tier = self._classify_tier(final_score)

        # Adjust tier down if two-hurdle says MEDIUM
        if two_hurdle == "MEDIUM" and tier in (RiskTier.HIGH, RiskTier.CRITICAL):
            tier = RiskTier.MEDIUM

        # Confidence bounds
        lower, upper = self._confidence_bounds(final_score, len(detector_scores))

        return EnsembleScoringResult(
            risk_score=final_score,
            risk_tier=tier,
            detector_scores=detector_scores,
            meta_learner_score=meta_score,
            two_hurdle_result=two_hurdle,
            confidence_lower=lower,
            confidence_upper=upper,
        )

    def score_batch(
        self,
        edges: list[Edge],
    ) -> list[EnsembleScoringResult]:
        """Score a batch of transactions."""
        return [
            self.score_transaction(
                src=e.src_id, dst=e.dst_id,
                timestamp=e.timestamp, features=e.features,
            )
            for e in edges
        ]

    def _two_hurdle_filter(
        self,
        reconstruction_score: float,
        detector_scores: dict[str, float],
    ) -> str:
        """TFLAG-inspired false-positive suppression.

        Only flag HIGH when BOTH reconstruction and deviation are anomalous.
        """
        # Use drift monitor as reconstruction signal
        drift_score = detector_scores.get("Drift Monitor", 0.0)
        # Use TGN as deviation signal
        tgn_score = detector_scores.get("TGN Memory", 0.0)

        recon_high = drift_score > self.config.recon_threshold
        deviation_high = tgn_score > (self.config.deviation_threshold / 10.0)

        if recon_high and deviation_high:
            return "HIGH"
        elif recon_high or reconstruction_score >= 0.60:
            return "MEDIUM"
        return "PASS"

    def _classify_tier(self, score: float) -> RiskTier:
        if score >= self.thresholds.critical:
            return RiskTier.CRITICAL
        elif score >= self.thresholds.high:
            return RiskTier.HIGH
        elif score >= self.thresholds.medium:
            return RiskTier.MEDIUM
        return RiskTier.LOW

    def _confidence_bounds(self, score: float, n_detectors: int) -> tuple[float, float]:
        """Confidence bounds based on detector agreement."""
        z = 1.96
        n = max(n_detectors, 3)
        margin = z * np.sqrt(score * (1 - score) / n)
        return max(0.0, score - margin), min(1.0, score + margin)

    def _build_features(self, amount: float, timestamp: float) -> np.ndarray:
        """Build basic feature vector from amount and timestamp."""
        feat = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
        feat[0] = np.log1p(amount)
        feat[1] = min(amount / 10000.0, 1.0)
        hour_frac = (timestamp % 86400) / 86400
        feat[2] = np.sin(2 * np.pi * hour_frac)
        feat[3] = np.cos(2 * np.pi * hour_frac)
        day_frac = (timestamp % 604800) / 604800
        feat[4] = np.sin(2 * np.pi * day_frac)
        feat[5] = np.cos(2 * np.pi * day_frac)
        feat[6] = 0.5
        return feat
