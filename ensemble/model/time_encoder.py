"""
Multi-scale temporal encoding (TempReasoner, Scientific Reports 2026).

Runs separate Fourier encoders at five temporal granularities and fuses
with learned scale weights. Detects fraud patterns at different timescales
simultaneously — minute-level card-testing bursts AND week-level ring
coordination.

This is the ensemble's own copy — tgn_learn/ is read-only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Single-scale learnable Fourier time encoding (Xu et al., 2020).

    Args:
        time_dim: Output dimension
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            freqs = 1.0 / 10 ** np.linspace(0, 9, self.time_dim, dtype=np.float32)
            self.w.weight.copy_(torch.from_numpy(freqs).reshape(self.time_dim, 1))
            self.w.bias.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cos(self.w(t.float()))


class MultiScaleTimeEncoder(nn.Module):
    """Multi-scale temporal encoding with learned fusion.

    Scales (seconds per unit):
        - Minute:  60s
        - Hour:    3,600s
        - Day:     86,400s
        - Week:    604,800s
        - Month:   2,592,000s

    Args:
        time_dim: Output dimension of the fused encoding.
    """

    SCALES = [60.0, 3600.0, 86400.0, 604800.0, 2592000.0]

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.num_scales = len(self.SCALES)
        self.encoders = nn.ModuleList([
            TimeEncoder(time_dim) for _ in self.SCALES
        ])
        self.scale_fusion = nn.Linear(self.num_scales * time_dim, time_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps at multiple temporal scales.

        Args:
            t: Timestamps in seconds, shape [batch] or [batch, 1]

        Returns:
            Fused encoding, shape [batch, time_dim]
        """
        scale_encs = [
            encoder(t / scale)
            for encoder, scale in zip(self.encoders, self.SCALES)
        ]
        fused = torch.cat(scale_encs, dim=-1)
        return self.scale_fusion(fused)
