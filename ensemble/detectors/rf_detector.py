"""
RF Structural Detector — weighted Random Forest on TGN embeddings.

Uses cached embeddings + structural features to produce fraud scores
with feature importances. Complements TGN by capturing explicit
structural patterns (bridge edges, triangle closure) that attention
may underweight.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tgn_learn.graph import TemporalGraph

from ..embedding.embedding_cache import EmbeddingCache
from ..model.rf_head import RFScoringHead
from .base import BaseDetector


class RFDetector(BaseDetector):
    """Random Forest detector on TGN embeddings + structural features.

    Args:
        rf_head: Fitted RFScoringHead (or unfitted — will fit on .fit())
        cache: EmbeddingCache for fast embedding lookup
        embedding_dim: Dimension of cached embeddings
        edge_feat_dim: Dimension of edge features
    """

    def __init__(
        self,
        rf_head: Optional[RFScoringHead] = None,
        cache: Optional[EmbeddingCache] = None,
        embedding_dim: int = 64,
        edge_feat_dim: int = 20,
    ):
        self._rf = rf_head or RFScoringHead()
        self._cache = cache or EmbeddingCache()
        self._embedding_dim = embedding_dim
        self._edge_feat_dim = edge_feat_dim

    @property
    def name(self) -> str:
        return "RF Structural"

    @property
    def is_fitted(self) -> bool:
        return self._rf.is_fitted

    def fit(self, graph: TemporalGraph) -> None:
        """Fit RF on graph — requires embeddings already in cache.

        Builds feature matrix from cached embeddings + edge features
        and fits the RF on labelled edges.
        """
        X_list = []
        y_list = []

        for edge in graph.edges:
            if edge.label < 0:
                continue  # Skip unlabelled

            z_src = self._cache.get(edge.src_id)
            z_dst = self._cache.get(edge.dst_id)

            if z_src is None:
                z_src = np.zeros(self._embedding_dim, dtype=np.float32)
            if z_dst is None:
                z_dst = np.zeros(self._embedding_dim, dtype=np.float32)

            x = np.concatenate([z_src, z_dst, edge.features])
            X_list.append(x)
            y_list.append(edge.label)

        if len(X_list) < 10:
            return  # Not enough data

        X = np.stack(X_list)
        y = np.array(y_list)

        feature_names = RFScoringHead.build_feature_names(
            self._embedding_dim, self._edge_feat_dim
        )
        self._rf.fit(X, y, feature_names=feature_names)

    def score(
        self,
        src: int,
        dst: int,
        timestamp: float,
        features: np.ndarray,
        graph: Optional[TemporalGraph] = None,
    ) -> float:
        """Score using RF on cached embeddings."""
        if not self._rf.is_fitted:
            return 0.0

        z_src = self._cache.get(src)
        z_dst = self._cache.get(dst)

        if z_src is None:
            z_src = np.zeros(self._embedding_dim, dtype=np.float32)
        if z_dst is None:
            z_dst = np.zeros(self._embedding_dim, dtype=np.float32)

        x = np.concatenate([z_src, z_dst, features])
        return float(self._rf.predict_proba(x.reshape(1, -1))[0])
