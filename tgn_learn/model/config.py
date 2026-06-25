"""
TGN model configuration.

Provides sensible defaults for a learning context — smaller dimensions
than production to enable fast iteration on a laptop.
"""

from dataclasses import dataclass


@dataclass
class TGNConfig:
    """Configuration for the TGN fraud detection model.

    Attributes:
        memory_dim: Dimension of node memory vectors (TGN state)
        embedding_dim: Output dimension of graph attention embedding
        time_dim: Dimension of temporal encoding
        edge_feat_dim: Dimension of edge feature vectors
        num_neighbors: Number of temporal neighbors to sample
        num_heads: Number of attention heads in TransformerConv
        dropout: Dropout rate in attention layers
        use_multiscale_time: Use MultiScaleTimeEncoder (TempReasoner 2026)
            instead of single-scale TimeEncoder. Detects fraud patterns
            at minute, hour, day, week, and month scales simultaneously.
        fit_rf_head: Fit a Random Forest scoring head post-training
            (NID-TGN, SPACE 2024). When True, an RFScoringHead is fitted
            on validation embeddings after TGN training completes.
        rf_n_estimators: Number of trees in the RF scoring head.
        rf_max_depth: Max depth for RF trees (None for unlimited).
    """

    memory_dim: int = 64
    embedding_dim: int = 64
    time_dim: int = 32
    edge_feat_dim: int = 20
    num_neighbors: int = 10
    num_heads: int = 2
    dropout: float = 0.1

    # Phase 1A — multi-scale time encoding (TempReasoner, 2026)
    use_multiscale_time: bool = True

    # Phase 1B — RF scoring head (NID-TGN, 2024)
    fit_rf_head: bool = True
    rf_n_estimators: int = 200
    rf_max_depth: int | None = 10
