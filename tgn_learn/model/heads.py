"""
Scoring heads for link prediction and node classification.

The TGN uses dual scoring heads:
- LinkPredictor: Scores edges (transactions) for anomaly detection
- NodeClassifier: Scores nodes (accounts) for overall risk level

Both heads are simple MLPs that operate on learned embeddings.
"""

import torch
import torch.nn as nn


class LinkPredictor(nn.Module):
    """
    MLP link predictor for transaction-level anomaly scoring.

    Takes source and destination embeddings, combines them, and outputs
    a scalar anomaly score. High scores indicate suspicious transactions.

    Args:
        in_channels: Embedding dimension
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Score source-destination pairs.

        Args:
            z_src: Source embeddings [batch, dim]
            z_dst: Destination embeddings [batch, dim]

        Returns:
            Anomaly logits [batch, 1]
        """
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class NodeClassifier(nn.Module):
    """
    MLP node classifier for account-level risk scoring.

    Takes a node embedding and predicts the probability of the
    account being involved in fraud.

    Args:
        in_channels: Embedding dimension
        hidden: Hidden layer size
    """

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Classify nodes.

        Args:
            z: Node embeddings [batch, dim]

        Returns:
            Risk logits [batch, 1]
        """
        return self.net(z)
