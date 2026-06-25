"""
Threshold Adapter — per-segment risk threshold calibration.

Monitors daily false positive rate and recall per card segment,
applies exponential smoothing to adjust the medium/high/critical
thresholds. Keeps FP rate within target while maximising recall.

Simplified from the RL-based approach in hi-26 — uses EMA-based
feedback loop instead of full reinforcement learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import RiskThresholds


@dataclass
class SegmentMetrics:
    """Performance metrics for a card/account segment.

    Attributes:
        segment_id: Identifier for the segment
        fp_rate: Current false positive rate
        recall: Current fraud recall (sensitivity)
        n_transactions: Number of transactions observed
        n_fraud_detected: True positives
        n_fraud_missed: False negatives
        n_false_alarms: False positives
    """

    segment_id: str
    fp_rate: float = 0.0
    recall: float = 0.0
    n_transactions: int = 0
    n_fraud_detected: int = 0
    n_fraud_missed: int = 0
    n_false_alarms: int = 0


class ThresholdAdapter:
    """Adaptive risk threshold calibration per segment.

    Adjusts thresholds based on observed FP rate and recall:
    - If FP rate too high → raise thresholds (fewer alerts)
    - If recall too low → lower thresholds (more alerts)
    - Uses exponential smoothing for stability

    Args:
        target_fp_rate: Maximum acceptable false positive rate
        target_recall: Minimum acceptable fraud recall
        alpha: EMA smoothing factor (higher = more responsive)
        min_observations: Minimum transactions before adapting
        max_adjustment: Maximum per-update threshold change
    """

    def __init__(
        self,
        target_fp_rate: float = 0.05,
        target_recall: float = 0.80,
        alpha: float = 0.1,
        min_observations: int = 100,
        max_adjustment: float = 0.05,
    ):
        self.target_fp_rate = target_fp_rate
        self.target_recall = target_recall
        self.alpha = alpha
        self.min_observations = min_observations
        self.max_adjustment = max_adjustment

        # Per-segment thresholds
        self._segment_thresholds: dict[str, RiskThresholds] = {}
        # Per-segment smoothed metrics
        self._segment_fp_ema: dict[str, float] = {}
        self._segment_recall_ema: dict[str, float] = {}
        # History for monitoring
        self._adaptation_history: list[dict] = []

    @property
    def default_thresholds(self) -> RiskThresholds:
        return RiskThresholds()

    def get_thresholds(self, segment_id: str = "default") -> RiskThresholds:
        """Get current thresholds for a segment.

        Args:
            segment_id: Segment identifier (default = global)

        Returns:
            RiskThresholds for this segment
        """
        return self._segment_thresholds.get(segment_id, RiskThresholds())

    def update(
        self,
        segment_id: str,
        metrics: SegmentMetrics,
    ) -> RiskThresholds:
        """Update thresholds based on observed performance.

        Args:
            segment_id: Segment being updated
            metrics: Observed performance metrics

        Returns:
            Updated RiskThresholds for this segment
        """
        if metrics.n_transactions < self.min_observations:
            return self.get_thresholds(segment_id)

        # EMA update of FP rate and recall
        fp_ema = self._segment_fp_ema.get(segment_id, metrics.fp_rate)
        recall_ema = self._segment_recall_ema.get(segment_id, metrics.recall)

        fp_ema = (1 - self.alpha) * fp_ema + self.alpha * metrics.fp_rate
        recall_ema = (1 - self.alpha) * recall_ema + self.alpha * metrics.recall

        self._segment_fp_ema[segment_id] = fp_ema
        self._segment_recall_ema[segment_id] = recall_ema

        # Get current thresholds
        current = self._segment_thresholds.get(segment_id, RiskThresholds())

        # Compute adjustments
        adjustment = 0.0

        if fp_ema > self.target_fp_rate:
            # Too many false positives → raise thresholds
            overshoot = fp_ema - self.target_fp_rate
            adjustment = min(self.max_adjustment, overshoot * 0.5)
        elif recall_ema < self.target_recall:
            # Missing too much fraud → lower thresholds
            undershoot = self.target_recall - recall_ema
            adjustment = -min(self.max_adjustment, undershoot * 0.5)

        if adjustment != 0.0:
            new_thresholds = RiskThresholds(
                medium=np.clip(current.medium + adjustment, 0.10, 0.50),
                high=np.clip(current.high + adjustment, 0.40, 0.80),
                critical=np.clip(current.critical + adjustment, 0.70, 0.95),
            )
            self._segment_thresholds[segment_id] = new_thresholds

            self._adaptation_history.append({
                "segment_id": segment_id,
                "fp_ema": fp_ema,
                "recall_ema": recall_ema,
                "adjustment": adjustment,
                "new_medium": new_thresholds.medium,
                "new_high": new_thresholds.high,
                "new_critical": new_thresholds.critical,
            })

            return new_thresholds

        return current

    def get_history(self, segment_id: Optional[str] = None) -> list[dict]:
        """Get adaptation history, optionally filtered by segment."""
        if segment_id is None:
            return self._adaptation_history
        return [h for h in self._adaptation_history if h["segment_id"] == segment_id]

    def reset_segment(self, segment_id: str) -> None:
        """Reset a segment to default thresholds."""
        self._segment_thresholds.pop(segment_id, None)
        self._segment_fp_ema.pop(segment_id, None)
        self._segment_recall_ema.pop(segment_id, None)

    def reset_all(self) -> None:
        """Reset all segments to defaults."""
        self._segment_thresholds.clear()
        self._segment_fp_ema.clear()
        self._segment_recall_ema.clear()
        self._adaptation_history.clear()
