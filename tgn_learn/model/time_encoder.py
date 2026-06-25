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

Also provides PRAGMATimeEncoder (Revolut, arXiv 2604.08649, 2026)
which combines log-transform inter-event gaps with fixed calendar cycle
features — validated at production scale on 26M users / 24B events.
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


class PRAGMATimeEncoder(nn.Module):
    """Production-validated dual time encoding from PRAGMA (Revolut, 2026).

    Combines two orthogonal time signals, validated at scale on 26M users /
    24B events (arXiv:2604.08649):

    1. Log-transform inter-event gap: 8·ln(1 + Δt/8)
       — Preserves linear resolution for recent events, compresses old gaps.
       Solves aliasing that occurs with raw timestamps in standard Fourier
       encoding when the dynamic range spans seconds to years.

    2. Calendar cycle features (fixed, not learned):
       — hour_sin, hour_cos  (24h cycle)
       — dow_sin,  dow_cos   (7-day cycle)
       — dom_sin,  dom_cos   (30-day cycle)
       These capture daily (card-testing bursts at night) and weekly
       (bust-out activation on Mondays) fraud rhythms.

    The log-gap and calendar features are concatenated and projected to
    time_dim via a learned linear layer.

    Args:
        time_dim: Output dimension of the time encoding.

    Note:
        Unlike TempReasoner's MultiScaleTimeEncoder (which divides Δt by
        fixed scales), PRAGMA's log-transform works on the raw gap in
        seconds and is scale-agnostic — no hyperparameter to tune.
    """

    N_CALENDAR_FEATURES = 6  # 3 cycles × 2 (sin/cos)

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        # Project log-gap (1 dim) + calendar (6 dims) → time_dim
        self.proj = nn.Linear(1 + self.N_CALENDAR_FEATURES, time_dim)

    @staticmethod
    def _log_gap(delta_t: torch.Tensor) -> torch.Tensor:
        """PRAGMA log-transform: 8·ln(1 + Δt/8). Input in seconds."""
        delta_t = delta_t.float().abs()
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        return 8.0 * torch.log1p(delta_t / 8.0)

    @staticmethod
    def _calendar_features(t_abs: torch.Tensor) -> torch.Tensor:
        """Periodic calendar features from absolute timestamp (Unix seconds)."""
        t = t_abs.float()
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        TWO_PI = 2.0 * 3.141592653589793
        hour_angle = (t % 86400) / 86400 * TWO_PI
        dow_angle  = (t % 604800) / 604800 * TWO_PI
        dom_angle  = (t % 2592000) / 2592000 * TWO_PI

        return torch.cat([
            torch.sin(hour_angle), torch.cos(hour_angle),
            torch.sin(dow_angle),  torch.cos(dow_angle),
            torch.sin(dom_angle),  torch.cos(dom_angle),
        ], dim=-1)

    def forward(
        self,
        delta_t: torch.Tensor,
        t_abs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode time delta (and optionally absolute timestamp).

        Args:
            delta_t: Inter-event gaps in seconds [batch] or [batch, 1].
            t_abs:   Absolute Unix timestamps [batch] or [batch, 1].
                     If None, calendar features are zeroed (graceful fallback
                     when absolute timestamps are unavailable).

        Returns:
            Time encoding [batch, time_dim].
        """
        log_gap = self._log_gap(delta_t)                        # [batch, 1]

        if t_abs is not None:
            cal = self._calendar_features(t_abs)                # [batch, 6]
        else:
            batch = log_gap.size(0)
            cal = torch.zeros(batch, self.N_CALENDAR_FEATURES,
                              device=log_gap.device, dtype=log_gap.dtype)

        combined = torch.cat([log_gap, cal], dim=-1)            # [batch, 7]
        return self.proj(combined)                              # [batch, time_dim]
