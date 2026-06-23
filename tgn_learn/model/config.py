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
    """

    memory_dim: int = 64
    embedding_dim: int = 64
    time_dim: int = 32
    edge_feat_dim: int = 20
    num_neighbors: int = 10
    num_heads: int = 2
    dropout: float = 0.1
