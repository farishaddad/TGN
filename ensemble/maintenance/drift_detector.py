"""
Latent-Space Drift Detector (TGNN-CDD, 2025).

Monitors distribution shifts in TGN node embeddings using CUSUM
(Cumulative Sum) control charts on reconstruction error. More sensitive
than raw-feature monitoring — detects structural drift (new fraud
topology) and relational drift (new entity relationships).

Trigger conditions:
  - CUSUM statistic exceeds threshold → flag drift
  - On drift: recommend expanding temporal receptive field
  - On significant drift: recommend fine-tuning TGN

Three drift types tracked:
  - FEATURE: Node embedding distribution shifts
  - STRUCTURAL: Graph connectivity patterns change
  - RELATIONAL: Inter-entity relationship dynamics change
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class DriftType(str, Enum):
    """Types of concept drift detected."""
    FEATURE = "feature"
    STRUCTURAL = "structural"
    RELATIONAL = "relational"


@dataclass
class DriftEvent:
    """A detected drift event.

    Attributes:
        drift_type: Type of drift detected
        cusum_value: CUSUM statistic at detection
        timestamp: When drift was detected
        severity: 'minor' or 'major' based on threshold multiples
        recommendation: Suggested action
    """

    drift_type: DriftType
    cusum_value: float
    timestamp: float
    severity: str  # 'minor' | 'major'
    recommendation: str


class LatentSpaceDriftDetector:
    """Autoencoder-based concept drift detection in embedding space.

    Monitors reconstruction error distribution via CUSUM. When the
    error distribution shifts significantly, flags a DriftEvent.

    Args:
        embedding_dim: Dimension of embeddings being monitored
        hidden_dim: Autoencoder bottleneck dimension
        cusum_threshold: CUSUM statistic trigger threshold
        cusum_drift_factor: Allowable drift (k parameter in CUSUM)
        window_size: Number of recent observations to track
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 32,
        cusum_threshold: float = 5.0,
        cusum_drift_factor: float = 0.5,
        window_size: int = 100,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.cusum_threshold = cusum_threshold
        self.cusum_drift_factor = cusum_drift_factor
        self.window_size = window_size

        # Autoencoder (linear, fitted via SVD)
        self._encoder_w: Optional[np.ndarray] = None
        self._decoder_w: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None

        # Baseline statistics
        self._baseline_error_mean: float = 0.0
        self._baseline_error_std: float = 1.0

        # CUSUM state
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._observation_count: int = 0
        self._recent_errors: list[float] = []

        # History
        self._drift_events: list[DriftEvent] = []
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def drift_events(self) -> list[DriftEvent]:
        return self._drift_events

    @property
    def cusum_statistic(self) -> float:
        """Current maximum CUSUM value (higher = more drift)."""
        return max(self._cusum_pos, abs(self._cusum_neg))

    def fit_normal(self, normal_embeddings: np.ndarray) -> None:
        """Fit the autoencoder and baseline on normal embeddings.

        Args:
            normal_embeddings: [n_samples, embedding_dim] from legit transactions
        """
        if len(normal_embeddings) < 20:
            return

        X = normal_embeddings.astype(np.float32)
        self._mean = X.mean(axis=0)
        X_centered = X - self._mean

        # SVD-based linear autoencoder
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        k = min(self.hidden_dim, len(S))
        self._encoder_w = Vt[:k].T  # [emb_dim, hidden]
        self._decoder_w = Vt[:k]    # [hidden, emb_dim]

        # Compute baseline reconstruction errors
        reconstructed = (X_centered @ self._encoder_w) @ self._decoder_w
        errors = np.linalg.norm(X_centered - reconstructed, axis=1)

        self._baseline_error_mean = float(errors.mean())
        self._baseline_error_std = float(errors.std()) + 1e-8

        # Reset CUSUM
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._observation_count = 0
        self._recent_errors.clear()

        self._fitted = True

    def check(
        self,
        embeddings: np.ndarray,
        timestamp: float = 0.0,
    ) -> Optional[DriftEvent]:
        """Check a batch of new embeddings for drift.

        Args:
            embeddings: [n_samples, embedding_dim] new observations
            timestamp: Current time (for event metadata)

        Returns:
            DriftEvent if drift detected, None otherwise
        """
        if not self._fitted or self._encoder_w is None:
            return None

        X = embeddings.astype(np.float32)
        X_centered = X - self._mean

        # Reconstruction error
        reconstructed = (X_centered @ self._encoder_w) @ self._decoder_w
        errors = np.linalg.norm(X_centered - reconstructed, axis=1)
        mean_error = float(errors.mean())

        # Standardise against baseline
        z_score = (mean_error - self._baseline_error_mean) / self._baseline_error_std

        # CUSUM update
        k = self.cusum_drift_factor
        self._cusum_pos = max(0, self._cusum_pos + z_score - k)
        self._cusum_neg = min(0, self._cusum_neg + z_score + k)

        self._observation_count += 1
        self._recent_errors.append(mean_error)
        if len(self._recent_errors) > self.window_size:
            self._recent_errors.pop(0)

        # Check thresholds
        cusum_val = self.cusum_statistic
        if cusum_val > self.cusum_threshold:
            severity = "major" if cusum_val > 2 * self.cusum_threshold else "minor"
            drift_type = self._classify_drift(embeddings)

            recommendations = {
                DriftType.FEATURE: "Expand temporal receptive field; consider fine-tuning",
                DriftType.STRUCTURAL: "Rebuild graph neighbourhood; retrain GNN layers",
                DriftType.RELATIONAL: "Update entity embeddings; refresh embedding cache",
            }

            event = DriftEvent(
                drift_type=drift_type,
                cusum_value=cusum_val,
                timestamp=timestamp,
                severity=severity,
                recommendation=recommendations[drift_type],
            )
            self._drift_events.append(event)

            # Reset CUSUM after detection (one-shot detection)
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0

            return event

        return None

    def _classify_drift(self, embeddings: np.ndarray) -> DriftType:
        """Heuristically classify which type of drift occurred."""
        if self._mean is None:
            return DriftType.FEATURE

        X = embeddings.astype(np.float32)
        diff = X.mean(axis=0) - self._mean

        # Feature drift: uniform shift across dimensions
        # Structural drift: shift concentrated in specific dimensions
        # Relational drift: variance change without mean shift

        uniformity = float(np.std(np.abs(diff)))
        mean_shift = float(np.linalg.norm(diff))

        variance_before = self._baseline_error_std
        variance_after = float(np.std(np.linalg.norm(X - self._mean, axis=1)))
        variance_ratio = variance_after / (variance_before + 1e-8)

        if variance_ratio > 2.0 and mean_shift < self._baseline_error_std:
            return DriftType.RELATIONAL
        elif uniformity < 0.1 * mean_shift:
            return DriftType.FEATURE
        else:
            return DriftType.STRUCTURAL

    def reset(self) -> None:
        """Reset CUSUM state without losing the fitted autoencoder."""
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._observation_count = 0
        self._recent_errors.clear()
