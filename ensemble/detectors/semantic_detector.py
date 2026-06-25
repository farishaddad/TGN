"""
Semantic Detector — per-relation-type encoding (HTGNN, ICAART 2025).

Scores transactions based on semantic anomalies in the relation type.
Different transaction types (CNP, contactless, online, recurring) have
different normal patterns; this detector flags when behaviour within
a relation type deviates from its expected distribution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tgn_learn.graph import TemporalGraph

from .base import BaseDetector


class SemanticDetector(BaseDetector):
    """Per-relation-type semantic anomaly detector.

    Learns the typical feature distribution for each transaction
    type/channel and flags deviations within type.

    Args:
        n_sigma: Number of standard deviations for anomaly threshold
    """

    def __init__(self, n_sigma: float = 3.0):
        self._n_sigma = n_sigma
        self._type_stats: dict[int, dict[str, np.ndarray]] = {}
        self._fitted = False

    @property
    def name(self) -> str:
        return "Semantic Patterns"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, graph: TemporalGraph) -> None:
        """Learn per-type feature statistics from training data."""
        # Group edges by channel/type (feature index 6 in standard encoding)
        type_features: dict[int, list[np.ndarray]] = {}

        for edge in graph.edges:
            if edge.label == 1:
                continue  # Learn normal from legit only
            if edge.features is None:
                continue

            # Use discretised channel feature as type key
            type_key = int(edge.features[6] * 10) if len(edge.features) > 6 else 0
            type_features.setdefault(type_key, []).append(edge.features)

        # Compute mean and std per type
        self._type_stats.clear()
        for type_key, feats_list in type_features.items():
            if len(feats_list) < 5:
                continue
            feats = np.stack(feats_list)
            self._type_stats[type_key] = {
                "mean": feats.mean(axis=0),
                "std": feats.std(axis=0) + 1e-8,
                "count": len(feats_list),
            }

        self._fitted = len(self._type_stats) > 0

    def score(
        self,
        src: int,
        dst: int,
        timestamp: float,
        features: np.ndarray,
        graph: Optional[TemporalGraph] = None,
    ) -> float:
        """Score based on per-type deviation."""
        if not self._fitted:
            return 0.0

        type_key = int(features[6] * 10) if len(features) > 6 else 0
        stats = self._type_stats.get(type_key)

        if stats is None:
            # Unknown type — mildly suspicious
            return 0.3

        # Compute z-score across all features
        z_scores = np.abs(features - stats["mean"]) / stats["std"]
        max_z = float(z_scores.max())
        mean_z = float(z_scores.mean())

        # Convert to probability-like score
        # High z-score = more anomalous
        score = 1.0 - np.exp(-mean_z / self._n_sigma)
        return min(1.0, max(0.0, float(score)))
