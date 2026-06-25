"""
Graph Attention Embedding module.

Computes node embeddings by aggregating information from temporal
neighbours using multi-head attention (TransformerConv). Edge features
include both the raw message and a temporal encoding of the time
difference, allowing the model to weigh recent vs. older interactions.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

from .time_encoder import MultiScaleTimeEncoder, TimeEncoder


class GraphAttentionEmbedding(nn.Module):
    """
    Temporal graph attention embedding layer.

    Takes node memory states and aggregates over temporal neighbours
    using TransformerConv with edge features.

    Args:
        in_channels: Input node feature dimension (memory_dim)
        out_channels: Output embedding dimension
        msg_dim: Raw message/edge feature dimension
        time_dim: Time encoding dimension
        num_heads: Number of attention heads
        dropout: Attention dropout rate
        use_multiscale_time: If True, use MultiScaleTimeEncoder
            (TempReasoner 2026) instead of single-scale TimeEncoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        msg_dim: int,
        time_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        use_multiscale_time: bool = False,
    ):
        super().__init__()

        if use_multiscale_time:
            self.time_enc = MultiScaleTimeEncoder(time_dim)
        else:
            self.time_enc = TimeEncoder(time_dim)

        edge_dim = msg_dim + time_dim

        # TransformerConv with multi-head attention
        self.conv = TransformerConv(
            in_channels,
            out_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        last_update: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal embeddings.

        Args:
            x: Node memory states [num_nodes, memory_dim]
            last_update: Last update time per node [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            t: Edge timestamps [num_edges]
            msg: Edge features/messages [num_edges, msg_dim]

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Compute relative time encoding
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t)

        # Concatenate time encoding with message features
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        return self.conv(x, edge_index, edge_attr)
