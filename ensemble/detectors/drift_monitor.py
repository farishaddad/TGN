"""
Drift Monitor Detector — autoencoder reconstruction error.

Scores transactions based on how poorly their embedding can be
reconstructed by an autoencoder trained on normal data. High
reconstruction error indicates the embedding is "out of distribution"
— a concept drift or novel fraud pattern signal.

Based on TGNN-CDD latent-space monitoring approach.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tgn_learn.graph import TemporalGraph

from ..embedding.embedding_cache import EmbeddingCache
from .base import BaseDetector


class DriftMonitorDetector(BaseDetector):
    """Autoencoder-based drift/novelty detector.

    Trained on embeddings from normal transactions. At inference,
    high reconstruction error → novel or drifted behaviour.

    Args:
        cache: EmbeddingCache for embedding lookup
        embedding_dim: Expected embedding dimension
        hidden_dim: Autoencoder hidden dimension
        threshold_percentile: Percentile of training errors to use as threshold
    """

    def __init__(
        self,
        cache: Optional[EmbeddingCache] = None,
        embedding_dim: int = 64,
        hidden_dim: int = 32,
        threshold_percentile: float = 95.0,
    ):
        self._cache = cache or EmbeddingCache()
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._threshold_percentile = threshold_percentile

        # Simple autoencoder weights (numpy — no torch dependency for scoring)
        self._encoder_w: Optional[np.ndarray] = None
        self._encoder_b: Optional[np.ndarray] = None
        self._decoder_w: Optional[np.ndarray] = None
        self._decoder_b: Optional[np.ndarray] = None
        self._threshold: float = 1.0
        self._fitted = False

    @property
    def name(self) -> str:
        return "Drift Monitor"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, graph: TemporalGraph) -> None:
        """Train autoencoder on normal (legit) transaction embeddings."""
        # Collect embeddings for legit transactions
        embeddings = []
        for edge in graph.edges:
            if edge.label != 0:
                continue
            emb = self._cache.get(edge.src_id)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < 50:
            # Not enough data — use simple random init
            self._init_random()
            return

        X = np.stack(embeddings).astype(np.float32)

        # Train simple linear autoencoder via SVD (closed-form, fast)
        # X ≈ X @ W_enc @ W_dec
        # Using truncated SVD for the hidden bottleneck
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(self._hidden_dim, len(S))

        # Encoder: project to hidden dim
        self._encoder_w = Vt[:k].T  # [emb_dim, hidden]
        self._encoder_b = np.zeros(k, dtype=np.float32)

        # Decoder: project back
        self._decoder_w = Vt[:k]  # [hidden, emb_dim] (transpose of encoder)
        self._decoder_b = X.mean(axis=0)

        # Compute reconstruction errors on training data for threshold
        reconstructed = (X @ self._encoder_w) @ self._decoder_w
        errors = np.linalg.norm(X - reconstructed, axis=1)
        self._threshold = float(np.percentile(errors, self._threshold_percentile))

        self._fitted = True

    def score(
        self,
        src: int,
        dst: int,
        timestamp: float,
        features: np.ndarray,
        graph: Optional[TemporalGraph] = None,
    ) -> float:
        """Score based on reconstruction error."""
        if not self._fitted or self._encoder_w is None:
            return 0.0

        emb = self._cache.get(src)
        if emb is None:
            return 0.0

        # Reconstruct
        hidden = emb @ self._encoder_w + self._encoder_b
        reconstructed = hidden @ self._decoder_w + self._decoder_b
        error = float(np.linalg.norm(emb - reconstructed))

        # Normalise by threshold
        score = min(1.0, error / (self._threshold + 1e-8))
        return score

    def _init_random(self) -> None:
        """Initialise with random weights (fallback)."""
        rng = np.random.default_rng(42)
        self._encoder_w = rng.normal(0, 0.1, (self._embedding_dim, self._hidden_dim)).astype(np.float32)
        self._encoder_b = np.zeros(self._hidden_dim, dtype=np.float32)
        self._decoder_w = rng.normal(0, 0.1, (self._hidden_dim, self._embedding_dim)).astype(np.float32)
        self._decoder_b = np.zeros(self._embedding_dim, dtype=np.float32)
        self._threshold = 1.0
        self._fitted = True
