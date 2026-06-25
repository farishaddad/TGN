"""
Learnable Fourier time encoding (Xu et al., 2020).

Maps scalar timestamps into high-dimensional representations using
learnable frequency parameters. This allows the model to discover
which temporal patterns matter for fraud detection.

Key insight: The initial frequencies span many orders of magnitude
(1 to 10^-9), letting the model capture patterns from seconds to years.

Also provides MultiScaleTimeEncoder (TempReasoner, Scientific Reports 2026)
which runs separate encoders at five temporal granularities and fuses with
learned scale weights.
"""

import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Learnable Fourier time encoding.

    Transforms a scalar time delta into a time_dim-dimensional vector
    using cosine activations with learnable frequencies.

    Args:
        time_dim: Output dimension of the time encoding
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize frequencies spanning multiple time scales."""
        with torch.no_grad():
            # Frequencies from 1 (fast) to 10^-9 (slow)
            freqs = 1.0 / 10 ** np.linspace(0, 9, self.time_dim, dtype=np.float32)
            self.w.weight.copy_(torch.from_numpy(freqs).reshape(self.time_dim, 1))
            self.w.bias.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Args:
            t: Tensor of timestamps, shape [batch] or [batch, 1]

        Returns:
            Time encodings, shape [batch, time_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cos(self.w(t.float()))


class MultiScaleTimeEncoder(nn.Module):
    """Multi-scale temporal encoding (TempReasoner, Scientific Reports 2026).

    Runs separate Fourier encoders at five temporal granularities and
    fuses with learned scale weights. Detects fraud patterns that
    manifest at different timescales simultaneously — minute-level
    card-testing bursts AND week-level ring coordination.

    Scales (seconds per unit):
        - Minute:  60s
        - Hour:    3,600s
        - Day:     86,400s
        - Week:    604,800s
        - Month:   2,592,000s

    The fusion layer learns which temporal scales are most informative,
    allowing the model to attend to rapid bursts (card testing) and
    slow coordination (money laundering) simultaneously.

    Args:
        time_dim: Output dimension of the fused time encoding.
            Each individual encoder also produces time_dim features,
            which are concatenated (5 * time_dim) then projected back
            to time_dim via a learned linear layer.
    """

    SCALES = [60.0, 3600.0, 86400.0, 604800.0, 2592000.0]

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.num_scales = len(self.SCALES)

        # One encoder per temporal scale
        self.encoders = nn.ModuleList([
            TimeEncoder(time_dim) for _ in self.SCALES
        ])

        # Learned fusion: concatenated scale encodings → fused output
        self.scale_fusion = nn.Linear(self.num_scales * time_dim, time_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps at multiple temporal scales.

        Args:
            t: Tensor of timestamps in seconds, shape [batch] or [batch, 1]

        Returns:
            Fused multi-scale time encoding, shape [batch, time_dim]
        """
        # Normalise time by each scale and encode separately
        scale_encs = [
            encoder(t / scale)
            for encoder, scale in zip(self.encoders, self.SCALES)
        ]

        # Concatenate and fuse with learned projection
        fused = torch.cat(scale_encs, dim=-1)
        return self.scale_fusion(fused)
