"""
Transaction scoring and risk assessment.

The Scorer class loads a trained TGN model and provides an API for
scoring individual transactions or batches. It outputs calibrated
risk scores, risk tier classifications, and confidence bounds.

Risk Tiers:
    - LOW:      score < 0.30 — Normal transaction
    - MEDIUM:   0.30 <= score < 0.60 — Needs review
    - HIGH:     0.60 <= score < 0.85 — Likely fraud, hold for investigation
    - CRITICAL: score >= 0.85 — Block immediately
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from tgn_learn.graph import Edge, TemporalGraph, EDGE_FEAT_DIM
from tgn_learn.model import TGNConfig, TGNFraudDetector
from tgn_learn.training.trainer import TGNTrainer


class RiskTier(str, Enum):
    """Risk classification tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ScoringResult:
    """Result of scoring a single transaction.

    Attributes:
        risk_score: Calibrated probability of fraud [0.0, 1.0]
        risk_tier: Classification into LOW/MEDIUM/HIGH/CRITICAL
        raw_score: Uncalibrated model output (sigmoid of logit)
        confidence_lower: Lower bound of confidence interval
        confidence_upper: Upper bound of confidence interval
    """
    risk_score: float
    risk_tier: RiskTier
    raw_score: float
    confidence_lower: float = 0.0
    confidence_upper: float = 1.0

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier.value,
            "raw_score": self.raw_score,
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
        }

    def __str__(self) -> str:
        return (
            f"Score={self.risk_score:.4f} [{self.risk_tier.value}] "
            f"({self.confidence_lower:.3f}-{self.confidence_upper:.3f})"
        )


@dataclass
class RiskThresholds:
    """Configurable thresholds for risk tier classification."""
    medium: float = 0.30
    high: float = 0.60
    critical: float = 0.85


class Scorer:
    """
    Transaction scoring engine.

    Loads a trained TGN model and scores transactions against the
    existing graph context. Supports calibration via isotonic regression
    and produces risk tier classifications.

    Example:
        >>> scorer = Scorer.from_checkpoint("checkpoints/best_model.pt")
        >>> result = scorer.score_transaction(src=0, dst=5, amount=1500.0, timestamp=1700001000.0)
        >>> print(result)
        Score=0.7234 [HIGH] (0.650-0.790)
    """

    def __init__(
        self,
        model: TGNFraudDetector,
        thresholds: Optional[RiskThresholds] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.thresholds = thresholds or RiskThresholds()
        self.calibrator: Optional[IsotonicRegression] = None

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str,
        thresholds: Optional[RiskThresholds] = None,
        device: str = "auto",
    ) -> "Scorer":
        """Load scorer from a training checkpoint.

        Args:
            filepath: Path to saved checkpoint
            thresholds: Risk tier thresholds (or use defaults)
            device: Device to run inference on

        Returns:
            Configured Scorer instance
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = TGNTrainer.load_checkpoint(filepath, device=device)
        return cls(model, thresholds, device)

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic calibration from validation scores and labels.

        After calibration, `risk_score` in ScoringResult will be the
        calibrated probability rather than raw sigmoid output.

        Args:
            scores: Raw model scores (sigmoid outputs) [n_samples]
            labels: True labels (0 or 1) [n_samples]
        """
        mask = labels >= 0
        scores = scores[mask]
        labels = labels[mask]

        if len(scores) < 10:
            return  # Not enough data for calibration

        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self.calibrator.fit(scores, labels)

    def _classify_tier(self, score: float) -> RiskTier:
        """Map a score to a risk tier."""
        if score >= self.thresholds.critical:
            return RiskTier.CRITICAL
        elif score >= self.thresholds.high:
            return RiskTier.HIGH
        elif score >= self.thresholds.medium:
            return RiskTier.MEDIUM
        else:
            return RiskTier.LOW

    def _calibrate_score(self, raw_score: float) -> float:
        """Apply calibration if available."""
        if self.calibrator is not None:
            return float(self.calibrator.predict(np.array([raw_score]))[0])
        return raw_score

    def _compute_confidence(self, score: float, n_context: int = 10) -> tuple[float, float]:
        """Estimate confidence bounds using beta distribution approximation.

        Wider bounds for lower context (fewer observed interactions).
        """
        # Use Wilson confidence interval approximation
        z = 1.96  # 95% confidence
        n = max(n_context, 5)
        center = score
        margin = z * np.sqrt(score * (1 - score) / n + z**2 / (4 * n**2))
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return lower, upper

    @torch.no_grad()
    def score_transaction(
        self,
        src: int,
        dst: int,
        timestamp: float,
        features: Optional[np.ndarray] = None,
        amount: Optional[float] = None,
    ) -> ScoringResult:
        """Score a single transaction.

        Args:
            src: Source node ID
            dst: Destination node ID
            timestamp: Transaction timestamp
            features: Edge feature vector (EDGE_FEAT_DIM). If None, uses amount to build basic features.
            amount: Transaction amount (used to build features if features is None)

        Returns:
            ScoringResult with score, tier, and confidence bounds
        """
        # Build feature vector if not provided
        if features is None:
            features = self._build_basic_features(amount or 0.0, timestamp)

        src_t = torch.tensor([src], dtype=torch.long, device=self.device)
        dst_t = torch.tensor([dst], dtype=torch.long, device=self.device)
        t_t = torch.tensor([timestamp], dtype=torch.float, device=self.device)
        msg_t = torch.tensor(features, dtype=torch.float, device=self.device).unsqueeze(0)

        _, _, node_scores = self.model(src_t, dst_t, t_t, msg_t)
        raw_score = float(node_scores.sigmoid().item())

        # Calibrate
        calibrated_score = self._calibrate_score(raw_score)

        # Confidence bounds
        lower, upper = self._compute_confidence(calibrated_score)

        # Classify
        tier = self._classify_tier(calibrated_score)

        return ScoringResult(
            risk_score=calibrated_score,
            risk_tier=tier,
            raw_score=raw_score,
            confidence_lower=lower,
            confidence_upper=upper,
        )

    @torch.no_grad()
    def score_batch(
        self,
        edges: list[Edge],
    ) -> list[ScoringResult]:
        """Score a batch of transactions.

        Args:
            edges: List of Edge objects to score

        Returns:
            List of ScoringResult, one per edge
        """
        if not edges:
            return []

        src_t = torch.tensor([e.src_id for e in edges], dtype=torch.long, device=self.device)
        dst_t = torch.tensor([e.dst_id for e in edges], dtype=torch.long, device=self.device)
        t_t = torch.tensor([e.timestamp for e in edges], dtype=torch.float, device=self.device)
        msg_t = torch.tensor(
            np.stack([e.features for e in edges]),
            dtype=torch.float, device=self.device,
        )

        _, _, node_scores = self.model(src_t, dst_t, t_t, msg_t)
        raw_scores = node_scores.sigmoid().cpu().numpy().flatten()

        results = []
        for raw in raw_scores:
            calibrated = self._calibrate_score(float(raw))
            lower, upper = self._compute_confidence(calibrated)
            tier = self._classify_tier(calibrated)
            results.append(ScoringResult(
                risk_score=calibrated,
                risk_tier=tier,
                raw_score=float(raw),
                confidence_lower=lower,
                confidence_upper=upper,
            ))

        return results

    def _build_basic_features(self, amount: float, timestamp: float) -> np.ndarray:
        """Build a basic feature vector from amount and timestamp."""
        feat = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
        feat[0] = np.log1p(amount)
        feat[1] = min(amount / 10000.0, 1.0)
        hour_frac = (timestamp % 86400) / 86400
        feat[2] = np.sin(2 * np.pi * hour_frac)
        feat[3] = np.cos(2 * np.pi * hour_frac)
        day_frac = (timestamp % 604800) / 604800
        feat[4] = np.sin(2 * np.pi * day_frac)
        feat[5] = np.cos(2 * np.pi * day_frac)
        feat[6] = 0.5  # Default channel
        return feat
