"""
Static account context encoder (PRAGMA, Revolut arXiv:2604.08649, 2026).

Encodes time-invariant profile attributes as a fixed-dimensional vector
that is concatenated with the TGN node memory before the graph attention
embedding step.

This separation mirrors PRAGMA's finding: static attributes (account age,
plan, spending quantile) carry signals complementary to the dynamic event
stream. Fusing them at the embedding level rather than as edge features
gives the model explicit access to stable context.

Ablation from the paper shows +2.1% AUC on fraud tasks when profile state
is included vs. events-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProfileStateEncoder(nn.Module):
    """Static account context encoder (PRAGMA, Revolut 2026).

    Encodes time-invariant profile attributes as a fixed-dimensional
    vector that is concatenated with the TGN node memory before the
    graph attention embedding step.

    Input features (default 6 dims):
      - account_age_norm:    float  [0, 1]  (days since creation / 3650)
      - balance_quantile:    float  [0, 1]  (account balance percentile)
      - limit_quantile:      float  [0, 1]  (credit/spending limit percentile)
      - is_business:         float  {0, 1}  (business vs. consumer account)
      - region_emb_sin:      float  [-1,1]  (geographic cluster sin component)
      - region_emb_cos:      float  [-1,1]  (geographic cluster cos component)

    Args:
        profile_dim: Input feature dimension (default 6)
        out_dim: Output embedding dimension (should match memory_dim or be
                 specified via config.profile_encoder_dim)
    """

    def __init__(self, profile_dim: int = 6, out_dim: int = 32):
        super().__init__()
        self.profile_dim = profile_dim
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Linear(profile_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, profile_features: torch.Tensor) -> torch.Tensor:
        """Encode static profile attributes.

        Args:
            profile_features: [batch, profile_dim] or [num_nodes, profile_dim]

        Returns:
            Profile embedding [batch, out_dim]
        """
        return self.encoder(profile_features)
