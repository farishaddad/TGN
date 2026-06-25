"""
Base detector interface for the ensemble.

All specialised detectors implement this ABC so the ensemble
meta-learner can call them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from tgn_learn.graph import TemporalGraph


class BaseDetector(ABC):
    """Standard interface for all fraud detectors in the ensemble."""

    @abstractmethod
    def score(
        self,
        src: int,
        dst: int,
        timestamp: float,
        features: np.ndarray,
        graph: Optional[TemporalGraph] = None,
    ) -> float:
        """Return fraud probability in [0.0, 1.0]."""
        ...

    @abstractmethod
    def fit(self, graph: TemporalGraph) -> None:
        """Train/fit the detector on a labelled graph."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has been fitted."""
        return False
