"""
TGN Memory Detector — wraps the existing TGNFraudDetector.

Uses TGN's temporal memory to detect deviations from per-account
behavioural baselines. This is the core detector in the ensemble.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from tgn_learn.graph import TemporalGraph, EDGE_FEAT_DIM
from tgn_learn.model import TGNFraudDetector
from tgn_learn.scoring import Scorer

from .base import BaseDetector


class TGNDetector(BaseDetector):
    """TGN memory-based fraud detector.

    Wraps the existing Scorer for uniform BaseDetector interface.

    Args:
        model: Trained TGNFraudDetector
        device: Torch device
    """

    def __init__(self, model: TGNFraudDetector, device: str = "cpu"):
        self._model = model
        self._scorer = Scorer(model, device=device)
        self._device = device
        self._fitted = True  # Assumes model is already trained

    @property
    def name(self) -> str:
        return "TGN Memory"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, graph: TemporalGraph) -> None:
        """No-op — TGN is trained externally via TGNTrainer."""
        pass

    def score(
        self,
        src: int,
        dst: int,
        timestamp: float,
        features: np.ndarray,
        graph: Optional[TemporalGraph] = None,
    ) -> float:
        """Score using TGN's memory-based inference."""
        result = self._scorer.score_transaction(
            src=src, dst=dst, timestamp=timestamp, features=features,
        )
        return result.risk_score
