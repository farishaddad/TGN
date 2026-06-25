"""
Real-Time Embedder (BRIGHT Lambda architecture, CIKM 2022).

Lightweight inference-time embedding lookup with optional delta.
At scoring time:
  1. Retrieve z_src, z_dst from EmbeddingCache (pre-computed by BatchEmbedder)
  2. Optionally apply a lightweight temporal delta (last N transactions)
  3. Pass to scoring heads

No full neighbourhood traversal at inference time.
P99 latency target: <20ms for this step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tgn_learn.graph import TemporalGraph

from .embedding_cache import EmbeddingCache


class RTEmbedder:
    """Real-time embedding lookup with lightweight temporal delta.

    For each new transaction, retrieves pre-computed embeddings from
    the cache and optionally adjusts with a recency-weighted delta
    based on the last few transactions.

    Args:
        cache: EmbeddingCache with pre-computed embeddings
        delta_window: Number of recent transactions to consider for delta
        delta_weight: How much to weight the delta vs cached embedding
    """

    def __init__(
        self,
        cache: EmbeddingCache,
        delta_window: int = 5,
        delta_weight: float = 0.1,
    ):
        self.cache = cache
        self.delta_window = delta_window
        self.delta_weight = delta_weight

    def embed(
        self,
        src: int,
        dst: int,
        timestamp: float,
        edge_features: np.ndarray,
        graph: Optional[TemporalGraph] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get embeddings for a source-destination pair.

        Fast path: cache lookup only (no graph traversal).
        If graph is provided and cache misses, falls back to zero vectors.

        Args:
            src: Source node ID
            dst: Destination node ID
            timestamp: Current transaction timestamp
            edge_features: Edge feature vector
            graph: Optional graph for delta computation

        Returns:
            (z_src, z_dst) embedding vectors
        """
        z_src = self._get_embedding(src, timestamp, graph)
        z_dst = self._get_embedding(dst, timestamp, graph)
        return z_src, z_dst

    def _get_embedding(
        self,
        node_id: int,
        timestamp: float,
        graph: Optional[TemporalGraph] = None,
    ) -> np.ndarray:
        """Retrieve embedding for a single node with optional delta.

        Args:
            node_id: Node to look up
            timestamp: Current time
            graph: Optional graph for recency delta

        Returns:
            Embedding vector
        """
        cached = self.cache.get(node_id)

        if cached is None:
            # Cache miss — return zero vector
            # In production this would trigger async recomputation
            return np.zeros(64, dtype=np.float32)  # Default embedding dim

        if graph is None or self.delta_weight == 0:
            return cached

        # Apply lightweight temporal delta based on recent activity
        delta = self._compute_delta(node_id, timestamp, graph, cached.shape[0])
        if delta is not None:
            return (1 - self.delta_weight) * cached + self.delta_weight * delta

        return cached

    def _compute_delta(
        self,
        node_id: int,
        timestamp: float,
        graph: TemporalGraph,
        embedding_dim: int,
    ) -> Optional[np.ndarray]:
        """Compute a lightweight recency delta from recent transactions.

        Aggregates the last N edge features with exponential time decay.
        This captures very recent activity that the batch embedder hasn't
        seen yet (since it runs periodically, not per-transaction).

        Args:
            node_id: Node to compute delta for
            timestamp: Current time
            graph: Graph containing transaction history
            embedding_dim: Target dimension

        Returns:
            Delta vector or None if insufficient history
        """
        # Get recent edges for this node
        node_edges = graph.edges_for_node(node_id)
        recent = [
            e for e in node_edges
            if e.timestamp < timestamp  # Only past events
        ]

        if len(recent) < 2:
            return None

        # Take last N edges
        recent = sorted(recent, key=lambda e: e.timestamp)[-self.delta_window:]

        # Exponential time decay weighting
        time_diffs = np.array([timestamp - e.timestamp for e in recent])
        # Decay with half-life of 1 hour
        weights = np.exp(-time_diffs / 3600.0)
        weights /= weights.sum() + 1e-8

        # Weighted average of edge features
        features = np.stack([e.features for e in recent])
        weighted_feat = (features * weights[:, np.newaxis]).sum(axis=0)

        # Pad or truncate to embedding dim
        if len(weighted_feat) >= embedding_dim:
            return weighted_feat[:embedding_dim].astype(np.float32)
        else:
            delta = np.zeros(embedding_dim, dtype=np.float32)
            delta[:len(weighted_feat)] = weighted_feat
            return delta

    def build_scoring_input(
        self,
        src: int,
        dst: int,
        timestamp: float,
        edge_features: np.ndarray,
        graph: Optional[TemporalGraph] = None,
    ) -> np.ndarray:
        """Build the full feature vector for scoring heads.

        Concatenates [z_src, z_dst, edge_features] into a single
        vector suitable for the RF head or meta-learner.

        Args:
            src: Source node ID
            dst: Destination node ID
            timestamp: Transaction timestamp
            edge_features: Raw edge features
            graph: Optional graph for delta

        Returns:
            Concatenated feature vector [z_src | z_dst | edge_features]
        """
        z_src, z_dst = self.embed(src, dst, timestamp, edge_features, graph)
        return np.concatenate([z_src, z_dst, edge_features])
