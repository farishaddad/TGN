"""
Graph Attention Embedding module.

Computes node embeddings by aggregating information from temporal
neighbours using multi-head attention (TransformerConv). Edge features
include both the raw message and a temporal encoding of the time
difference, allowing the model to weigh recent vs. older interactions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

from .config import TGNConfig
from .time_encoder import MultiScaleTimeEncoder, PRAGMATimeEncoder, TimeEncoder


def _make_time_encoder(config: TGNConfig) -> nn.Module:
    """Factory for time encoders based on config.time_encoder_type.

    Args:
        config: TGNConfig with time_encoder_type field.

    Returns:
        Instantiated time encoder module.

    Priority logic:
        1. If time_encoder_type is explicitly set (not None), use it.
        2. Else fall back to use_multiscale_time bool for backwards compat.
    """
    enc_type = getattr(config, "time_encoder_type", None)

    if enc_type == "pragma":
        return PRAGMATimeEncoder(config.time_dim)
    elif enc_type == "multiscale":
        return MultiScaleTimeEncoder(config.time_dim)
    elif enc_type == "fourier":
        return TimeEncoder(config.time_dim)
    else:
        # Legacy fallback: use use_multiscale_time bool (None or unrecognised)
        if getattr(config, "use_multiscale_time", False):
            return MultiScaleTimeEncoder(config.time_dim)
        return TimeEncoder(config.time_dim)


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
            Deprecated — prefer passing config to use time_encoder_type.
        config: Optional TGNConfig for time_encoder_type selection.
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
        config: TGNConfig | None = None,
    ):
        super().__init__()

        # Select time encoder
        if config is not None:
            self.time_enc = _make_time_encoder(config)
        elif use_multiscale_time:
            self.time_enc = MultiScaleTimeEncoder(time_dim)
        else:
            self.time_enc = TimeEncoder(time_dim)

        self._is_pragma = isinstance(self.time_enc, PRAGMATimeEncoder)

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
        t_abs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute temporal embeddings.

        Args:
            x: Node memory states [num_nodes, memory_dim]
            last_update: Last update time per node [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            t: Edge timestamps [num_edges]
            msg: Edge features/messages [num_edges, msg_dim]
            t_abs: Absolute timestamps [num_edges] (used by PRAGMATimeEncoder
                   for calendar features; ignored by other encoders)

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Compute relative time delta
        rel_t = last_update[edge_index[0]] - t

        # Encode time (PRAGMA needs both delta and absolute)
        if self._is_pragma:
            rel_t_enc = self.time_enc(rel_t, t_abs=t_abs)
        else:
            rel_t_enc = self.time_enc(rel_t)

        # Concatenate time encoding with message features
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        return self.conv(x, edge_index, edge_attr)
